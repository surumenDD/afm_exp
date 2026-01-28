# -*- coding: utf-8 -*-
"""
Craftium Dreamer 学習済みエージェントの評価スクリプト（単発seed）
- exp_dreamer/train.py と同じ前処理・同じ Dreamer 構造で評価する
- 複数エピソードを実行し、return / length / chops を集計する
- 動画録画（RecordVideo）対応：全エピソード保存
- agent_state_dict形式チェックポイント / state_dict単体 の両方を自動判別してロード

想定:
  - このスクリプトは craftium_exp/experiments/exp_dreamer/ に置く
  - craftium_exp/dreamerv3-torch が存在する

実行例:
  - cd /home/jovyan/work/srv11/craftium_exp/experiments/exp_dreamer

python single_seed_dreamer_evaluate.py \
  --agent-path /home/jovyan/work/srv11/craftium_exp/output/exp_dreamer/2026-01-23_15-32-01/agents/agent_step_999424.pt \
  --num-episodes 5 \
  --seed 77 \
  --frameskip 4 \
  --obs-size 64 \
  --mt-wd ./eval_runs \
  --mt-port 49210 \
  --device cuda \
  --output-json ../output/dreamer_eval_seed77.json

"""

import os
import sys
import json
import time
import argparse
import random
import pathlib
import functools
from types import SimpleNamespace
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf, DictConfig

import craftium  # noqa: F401


# ---------------------------------------------------------------------------
# Path setup for dreamerv3-torch modules (Hydra chdir でも壊れない)
# ---------------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).resolve().parent
# exp_dreamer/ の親は experiments/、その親が craftium_exp/
WORKSPACE_ROOT = THIS_DIR.parents[1] if len(THIS_DIR.parents) >= 2 else THIS_DIR
DREAMER_ROOT = WORKSPACE_ROOT / "dreamerv3-torch"
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER_ROOT))

import dreamer as dv3  # type: ignore
import tools  # type: ignore
import envs.wrappers as wrappers  # type: ignore
from parallel import Damy  # type: ignore


# ---------------------------------------------------------------------------
# Gymnasium -> gym compatibility and Craftium dict observation wrapper
# exp_dreamer/train.py と同等
# ---------------------------------------------------------------------------
def as_float_scalar(x) -> float:
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    if a.size == 1:
        return float(a.reshape(()))
    return float(a.sum())


class GymV26ToV21(gym.Wrapper):
    """gymnasium (obs, reward, terminated, truncated, info) -> gym style (obs, reward, done, info)"""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        info = dict(info) if info is not None else {}
        info.setdefault("terminated", terminated)
        info.setdefault("truncated", truncated)
        info.setdefault("discount", np.array(0.0 if terminated else 1.0, dtype=np.float32))

        # ChopTree: episode total reward = chops（終了時の統計に入れる）
        if done and isinstance(info.get("episode"), dict):
            ep_r = info["episode"].get("r", None)
            if ep_r is not None:
                info["episode"]["chops"] = as_float_scalar(ep_r)

        return obs, reward, done, info

    def reset(self, **kwargs):
        # Gymnasium reset() -> (obs, info) を obs のみに潰す
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
            return obs
        return result


class CraftiumObsDictWrapper(gym.Wrapper):
    """image観測を Dreamer が期待する dict に変換する"""

    def __init__(self, env, obs_key: str = "image", channel_last: bool = True):
        super().__init__(env)
        self._obs_key = obs_key
        self._channel_last = channel_last

        raw_space = self.env.observation_space
        if not isinstance(raw_space, gym.spaces.Box):
            raise ValueError("CraftiumObsDictWrapper expects Box observation space")

        img_shape = raw_space.shape
        # FrameStack(1) 後を想定： (C,H,W)
        if len(img_shape) != 3:
            raise ValueError(f"Expected (C,H,W) after FrameStack, got {img_shape}")

        if channel_last:
            c, h, w = img_shape[0], img_shape[1], img_shape[2]
            image_shape = (h, w, c)
        else:
            image_shape = img_shape

        self.observation_space = gym.spaces.Dict(
            {
                obs_key: gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
                "is_first": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                "is_terminal": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            }
        )

    def _to_image(self, obs):
        arr = np.array(obs)
        if self._channel_last:
            # (C,H,W) -> (H,W,C)
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return {
            self._obs_key: self._to_image(obs),
            "is_first": True,
            "is_terminal": False,
        }

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(info.get("terminated", done))
        obs_dict = {
            self._obs_key: self._to_image(obs),
            "is_first": False,
            "is_terminal": terminated,
        }
        return obs_dict, reward, done, info


class OneHotActionCompat(gym.ActionWrapper):
    """Discrete -> onehot Box に見せる（Dreamerが onehot を期待する系の互換）"""

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(env.action_space, "n"):
            raise TypeError(f"OneHotActionCompat expects Discrete-like action_space, got: {env.action_space}")
        self._n = int(env.action_space.n)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self._n,), dtype=np.float32)

    def action(self, action):
        # Dreamerが dict を返すケース対応
        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]
            else:
                raise TypeError(f"Unsupported action dict keys: {list(action.keys())}")

        # intならそのまま
        if isinstance(action, (int, np.integer)):
            return int(action)

        # torch tensor / list / np.array を許容
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        a = np.asarray(action)

        # scalarならそのまま
        if a.ndim == 0:
            return int(a)

        # (n,) one-hot / logits -> argmax
        idx = np.argmax(a, axis=-1)
        if isinstance(idx, np.ndarray):
            idx = idx.item()
        return int(idx)


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------
def _find_latest_pt_in_dir(dir_path: str) -> Optional[str]:
    if not dir_path or (not os.path.isdir(dir_path)):
        return None
    pts = [f for f in os.listdir(dir_path) if f.endswith(".pt")]
    if not pts:
        return None
    pts = sorted(pts)
    return os.path.join(dir_path, pts[-1])


def resolve_agent_path(agent_path: str) -> str:
    p = str(agent_path).strip()
    if os.path.isdir(p):
        latest = _find_latest_pt_in_dir(p)
        if latest is None:
            raise FileNotFoundError(f"No .pt found in directory: {p}")
        return latest
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"agent_path not found: {p}")


def load_agent_payload(path: str, device: torch.device) -> Tuple[Dict[str, Any], DictConfig]:
    """
    戻り値:
      payload: loadしたdict（state_dict単体 or checkpoint辞書）
      cfg: DictConfig（あれば復元、なければ空）
    """
    obj = torch.load(path, map_location=device)

    # checkpoint形式
    if isinstance(obj, dict) and "agent_state_dict" in obj:
        cfg_container = obj.get("cfg", None)
        cfg = OmegaConf.create(cfg_container) if isinstance(cfg_container, (dict, list)) else OmegaConf.create({})
        return obj, cfg

    # state_dict単体
    if isinstance(obj, dict):
        return {"agent_state_dict": obj}, OmegaConf.create({})

    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def find_hydra_config_yaml(agent_path: str) -> Optional[str]:
    """
    agents/agent_step_*.pt から親をたどって .hydra/config.yaml を探す
    """
    p = pathlib.Path(agent_path).resolve()
    for parent in [p.parent] + list(p.parents):
        cand = parent / ".hydra" / "config.yaml"
        if cand.is_file():
            return str(cand)
    return None


def merge_cfg(base_cfg: DictConfig, hydra_cfg: Optional[DictConfig]) -> DictConfig:
    if hydra_cfg is None:
        return base_cfg
    # base_cfg に hydra_cfg を上書きマージ
    merged = OmegaConf.merge(base_cfg, hydra_cfg)
    return merged


def default_cfg_dict() -> Dict[str, Any]:
    # exp_dreamer/config/config.yaml の重要部分 + 評価に必要な最低限
    return dict(
        exp_name="exp_dreamer",
        seed=42,
        torch_deterministic=True,
        cuda=True,
        track=False,
        wandb_project_name="craftium-dreamer",
        wandb_entity=None,
        wandb_resume=True,
        env_id="Craftium/ChopTree-v0",
        mt_wd="./eval_runs",
        frameskip=4,
        mt_port=49200,
        obs_size=64,
        num_envs=1,
        time_limit=0,
        precision=32,
        # Dreamer core / model（学習時に合うように、上書き可能）
        steps=1,
        eval_every=1,
        log_every=1,
        reset_every=0,
        device="cuda",
        compile=False,
        debug=False,
        video_pred_log=False,
        action_repeat=1,
        prefill=0,
        dataset_size=1000,
        parallel=False,
        deterministic_run=False,
        reward_EMA=True,
        dyn_hidden=128,
        dyn_deter=128,
        units=128,
        encoder=dict(cnn_depth=8),
        decoder=dict(cnn_depth=8),
        actor=dict(layers=1),
        critic=dict(layers=1),
        batch_size=16,
        batch_length=32,
        train_ratio=32,
        pretrain=0,
    )


def to_dreamer_config(cfg: DictConfig, device_str: str, logdir: pathlib.Path) -> SimpleNamespace:
    container = OmegaConf.to_container(cfg, resolve=True)
    container = dict(container)

    # 必須に寄せる（train.py と同じ思想）
    container.setdefault("device", device_str)
    container.setdefault("size", [int(cfg.obs_size), int(cfg.obs_size)])
    container.setdefault("logdir", str(logdir))
    container.setdefault("traindir", None)
    container.setdefault("evaldir", None)
    container.setdefault("offline_traindir", "")
    container.setdefault("offline_evaldir", "")
    container.setdefault("envs", int(cfg.num_envs))
    container.setdefault("action_repeat", int(getattr(cfg, "action_repeat", cfg.frameskip)))
    container.setdefault("time_limit", int(getattr(cfg, "time_limit", 0)))
    return SimpleNamespace(**container)


# ---------------------------------------------------------------------------
# Env builder (評価用)
# exp_dreamer/train.py と一致させる
# ---------------------------------------------------------------------------
def make_eval_env(
    cfg: DictConfig,
    run_name: str,
    record_video: bool,
    video_dir: Optional[str],
    name_prefix: str,
):
    craftium_kwargs = dict(
        run_dir_prefix=str(cfg.mt_wd),
        mt_port=int(cfg.mt_port),
        frameskip=int(cfg.frameskip),
        rgb_observations=True,
        mt_listen_timeout=300_000,
        seed=int(cfg.seed),
    )

    if record_video:
        if video_dir is None:
            raise ValueError("record_video=True requires video_dir")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(cfg.env_id, render_mode="rgb_array", **craftium_kwargs)

        safe_prefix = str(name_prefix).replace("/", "__")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix=safe_prefix,
            episode_trigger=lambda episode_id: True,  # 全エピソード保存
        )
    else:
        env = gym.make(cfg.env_id, **craftium_kwargs)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, int(cfg.obs_size))
    env = gym.wrappers.FrameStack(env, 1)

    env = GymV26ToV21(env)
    env = CraftiumObsDictWrapper(env, obs_key="image", channel_last=True)
    if int(getattr(cfg, "time_limit", 0)) > 0:
        env = wrappers.TimeLimit(env, int(cfg.time_limit))

    env = OneHotActionCompat(env)
    env = wrappers.UUID(env)

    return env


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    agent_path: str,
    num_episodes: int,
    seed: int,
    frameskip: int,
    record_video: bool,
    video_dir: Optional[str],
    device_str: str,
    obs_size: int,
    mt_wd: str,
    mt_port: int,
    env_id: str,
    config_yaml: Optional[str],
) -> Dict[str, Any]:
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")
    print(f"[Eval] device={device}")
    agent_path = resolve_agent_path(agent_path)
    print(f"[Eval] agent_path={agent_path}")
    print(f"[Eval] env_id={env_id}")
    print(f"[Eval] episodes={num_episodes}")
    print(f"[Eval] seed={seed}")
    print(f"[Eval] frameskip={frameskip}")
    print(f"[Eval] obs_size={obs_size}")
    print(f"[Eval] mt_wd={mt_wd}")
    print(f"[Eval] mt_port={mt_port}")
    if record_video:
        print(f"[Eval] video_dir={video_dir}")

    # base cfg
    base_cfg = OmegaConf.create(default_cfg_dict())

    # load cfg from checkpoint if present
    payload, ckpt_cfg = load_agent_payload(agent_path, device)

    # load cfg from hydra config.yaml if found (ckpt_cfg が空のときの救済)
    hydra_cfg = None
    if (ckpt_cfg is None) or (len(ckpt_cfg.keys()) == 0):
        if config_yaml is None:
            config_yaml = find_hydra_config_yaml(agent_path)
        if config_yaml is not None:
            try:
                hydra_cfg = OmegaConf.load(config_yaml)
                print(f"[Eval] hydra config loaded: {config_yaml}")
            except Exception as e:
                print(f"[Eval] failed to load hydra config: {config_yaml} ({e})")

    cfg = merge_cfg(base_cfg, ckpt_cfg if len(ckpt_cfg.keys()) > 0 else hydra_cfg)

    # overwrite by CLI (明示指定を優先)
    cfg.env_id = env_id
    cfg.seed = int(seed)
    cfg.frameskip = int(frameskip)
    cfg.obs_size = int(obs_size)
    cfg.num_envs = 1
    cfg.mt_wd = str(mt_wd)
    cfg.mt_port = int(mt_port)

    # deterministic
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    torch.backends.cudnn.deterministic = bool(getattr(cfg, "torch_deterministic", True))

    # local logdir for evaluation
    ts = int(time.time())
    eval_root = pathlib.Path(f"./eval_logs/{ts}")
    eval_root.mkdir(parents=True, exist_ok=True)

    dreamer_config = to_dreamer_config(cfg, "cuda" if device.type == "cuda" else "cpu", eval_root)
    logdir = pathlib.Path(dreamer_config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    dreamer_config.traindir = logdir / "train_eps"
    dreamer_config.evaldir = logdir / "eval_eps"
    dreamer_config.traindir.mkdir(parents=True, exist_ok=True)
    dreamer_config.evaldir.mkdir(parents=True, exist_ok=True)

    # logger
    logger = tools.Logger(logdir, 0)

    # datasets (空でも train.py と同様に作る)
    train_eps = tools.load_episodes(dreamer_config.traindir, limit=int(getattr(dreamer_config, "dataset_size", 1000)))
    train_dataset = dv3.make_dataset(train_eps, dreamer_config)

    # env
    run_name = f"{cfg.env_id}__eval__{cfg.seed}__{ts}"
    name_prefix = f"eval_seed{cfg.seed}"
    env = make_eval_env(
        cfg=cfg,
        run_name=run_name,
        record_video=record_video,
        video_dir=video_dir,
        name_prefix=name_prefix,
    )
    env = Damy(env)

    # action size
    acts = env.action_space
    dreamer_config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # agent
    agent = dv3.Dreamer(
        env.observation_space,
        env.action_space,
        dreamer_config,
        logger,
        train_dataset,
    ).to(device)
    agent.requires_grad_(requires_grad=False)

    # load weights
    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    agent.eval()
    print("[Eval] agent loaded")

    # simulate N episodes (train.py の eval と同様)
    eval_eps = tools.load_episodes(dreamer_config.evaldir, limit=1_000_000)
    prev_keys = set(eval_eps.keys())

    eval_policy = functools.partial(agent, training=False)
    tools.simulate(
        eval_policy,
        [env],
        eval_eps,
        dreamer_config.evaldir,
        logger,
        is_eval=True,
        episodes=int(num_episodes),
    )

    # extract new episodes
    new_keys = [k for k in eval_eps.keys() if k not in prev_keys]
    if not new_keys:
        print("[Eval] no new episodes were collected (unexpected).")
        new_eps = []
    else:
        new_eps = [eval_eps[k] for k in new_keys]

    episode_returns: List[float] = []
    episode_lengths_env: List[int] = []
    episode_lengths_real: List[int] = []
    episode_chops: List[float] = []

    for i, ep in enumerate(new_eps):
        rew = ep.get("reward", None)
        if rew is None:
            rsum = 0.0
            elen = 0
        else:
            rew_arr = np.asarray(rew, dtype=np.float32)
            rsum = float(rew_arr.sum())
            elen = int(rew_arr.shape[0])

        chops = rsum  # ChopTree は reward 合計が chops
        episode_returns.append(rsum)
        episode_lengths_env.append(elen)
        episode_lengths_real.append(int(elen * int(cfg.frameskip)))
        episode_chops.append(chops)

        print(f"  Episode {i+1}/{len(new_eps)}: return={rsum:.1f}, len_env={elen}, len_real={elen*int(cfg.frameskip)}, chops={chops:.1f}")

    # close
    try:
        env.close()
    except Exception:
        pass

    if len(episode_returns) == 0:
        # 空防止
        results = dict(
            agent_path=agent_path,
            env_id=cfg.env_id,
            num_episodes=int(num_episodes),
            seed=int(cfg.seed),
            frameskip=int(cfg.frameskip),
            obs_size=int(cfg.obs_size),
            mt_wd=str(cfg.mt_wd),
            mt_port=int(cfg.mt_port),
            episode_returns=[],
            episode_lengths_env=[],
            episode_lengths_real=[],
            episode_chops=[],
            mean_return=0.0,
            std_return=0.0,
            mean_length_env=0.0,
            mean_length_real=0.0,
            mean_chops=0.0,
            std_chops=0.0,
            video_dir=video_dir if record_video else None,
        )
        return results

    results = {
        "agent_path": agent_path,
        "env_id": cfg.env_id,
        "num_episodes": int(num_episodes),
        "seed": int(cfg.seed),
        "frameskip": int(cfg.frameskip),
        "obs_size": int(cfg.obs_size),
        "mt_wd": str(cfg.mt_wd),
        "mt_port": int(cfg.mt_port),
        "episode_returns": episode_returns,
        "episode_lengths_env": episode_lengths_env,
        "episode_lengths_real": episode_lengths_real,
        "episode_chops": episode_chops,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "mean_length_env": float(np.mean(episode_lengths_env)),
        "mean_length_real": float(np.mean(episode_lengths_real)),
        "std_length_real": float(np.std(episode_lengths_real)),
        "mean_chops": float(np.mean(episode_chops)),
        "std_chops": float(np.std(episode_chops)),
        "video_dir": video_dir if record_video else None,
    }

    print("\n" + "=" * 60)
    print("Evaluation Results (Dreamer)")
    print("=" * 60)
    print(f"  Mean Return:  {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Min/Max:      {results['min_return']:.1f} / {results['max_return']:.1f}")
    print(f"  Mean Length:  {results['mean_length_real']:.0f} ± {results['std_length_real']:.0f} (real steps)")
    print(f"  Mean Chops:   {results['mean_chops']:.2f} ± {results['std_chops']:.2f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Craftium Dreamer agent (single seed)")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to agent .pt (state_dict or checkpoint) OR directory containing .pt")
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--video-dir", type=str, default=None, help="Video output directory")
    parser.add_argument("--mt-wd", type=str, default="./eval_runs", help="Minetest run directory prefix")
    parser.add_argument("--mt-port", type=int, default=49200, help="Minetest port")
    parser.add_argument("--frameskip", type=int, default=4, help="Frame skip (must match training)")
    parser.add_argument("--obs-size", type=int, default=64, help="Resize observation size (must match training)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--config-yaml", type=str, default=None, help="Optional: path to Hydra config.yaml (/.hydra/config.yaml)")
    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation results to JSON")
    args = parser.parse_args()

    # video dir default
    if args.video_dir is None:
        ts = int(time.time())
        args.video_dir = f"./eval_videos/eval_{ts}"

    if not args.no_video:
        os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.mt_wd, exist_ok=True)

    results = evaluate(
        agent_path=args.agent_path,
        num_episodes=args.num_episodes,
        seed=args.seed,
        frameskip=args.frameskip,
        record_video=(not args.no_video),
        video_dir=(args.video_dir if not args.no_video else None),
        device_str=args.device,
        obs_size=args.obs_size,
        mt_wd=args.mt_wd,
        mt_port=args.mt_port,
        env_id=args.env_id,
        config_yaml=args.config_yaml,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Eval] results saved to: {args.output_json}")

    if not args.no_video:
        print(f"[Eval] videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
