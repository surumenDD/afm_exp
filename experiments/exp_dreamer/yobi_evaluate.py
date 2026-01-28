# -*- coding: utf-8 -*-
"""
Craftium Dreamer 学習済みエージェントの評価スクリプト（単発seed）
- exp_dreamer/train.py で保存した checkpoint / agent重み をロード
- 複数エピソード回して return / length / chops を集計
- Gymnasium RecordVideo で全エピソードの動画を保存
- Dreamer実行は dreamerv3-torch の tools.simulate を利用（trainと同系統の実行経路）

配置:
  craftium_exp/experiments/exp_dreamer/
    - train.py
    - single_seed_dreamer_evaluate.py   <- これ
    - config/config.yaml

実行例:
    python single_seed_dreamer_evaluate.py \
  --agent-path /home/jovyan/work/srv11/craftium_exp/output/exp_dreamer/2026-01-XX_XX-XX-XX/checkpoints/latest.pt \
  --num-episodes 5 \
  --seed 77 \
  --mt-port 49200 \
  --output-json ./eval_results.json

"""

import os
import sys
import json
import time
import random
import argparse
import pathlib
import functools
import logging
from types import SimpleNamespace
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf, DictConfig

import craftium

# ---------------------------------------------------------------------------
# Path setup for dreamerv3-torch modules (keeps working regardless of cwd)
# ---------------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1] if len(THIS_DIR.parents) >= 4 else THIS_DIR
DREAMER_ROOT = WORKSPACE_ROOT / "dreamerv3-torch"
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER_ROOT))

import dreamer as dv3  # type: ignore
import tools  # type: ignore
import envs.wrappers as wrappers  # type: ignore
from parallel import Damy  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if len(logger.handlers) > 0:
        return

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def as_float_scalar(x) -> float:
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    if a.size == 1:
        return float(a.reshape(()))
    return float(a.sum())


def _safe_prefix(name: str) -> str:
    return str(name).replace("/", "__")


def find_hydra_config_near(agent_path: str, max_up: int = 6) -> Optional[pathlib.Path]:
    """
    Hydra出力ディレクトリ配下にある .hydra/config.yaml を探索する。
    agents/ や checkpoints/ からでも拾える想定。
    """
    p = pathlib.Path(agent_path).resolve()
    cur = p.parent
    for _ in range(max_up):
        cand = cur / ".hydra" / "config.yaml"
        if cand.exists():
            return cand
        cur = cur.parent
    return None


def load_cfg_anyhow(agent_path: str, cfg_path_arg: Optional[str]) -> DictConfig:
    """
    config決定優先順位:
      1) --config で渡されたyaml
      2) agent_path近傍の .hydra/config.yaml
      3) exp_dreamer/config/config.yaml（ローカルのデフォルト）
    """
    if cfg_path_arg is not None and str(cfg_path_arg).strip() != "":
        cfgp = pathlib.Path(cfg_path_arg).expanduser().resolve()
        if not cfgp.exists():
            raise FileNotFoundError(f"--config not found: {cfgp}")
        return OmegaConf.load(str(cfgp))

    hydra_cfg = find_hydra_config_near(agent_path)
    if hydra_cfg is not None:
        return OmegaConf.load(str(hydra_cfg))

    default_cfg = THIS_DIR / "config" / "config.yaml"
    if not default_cfg.exists():
        raise FileNotFoundError(f"default config not found: {default_cfg}")
    return OmegaConf.load(str(default_cfg))


def load_agent_state_dict_and_maybe_cfg(agent_path: str, device: torch.device) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    train.py の保存形式に対応する。
      - checkpoint形式: {"agent_state_dict": ..., "cfg": ..., ...}
      - agent保存形式: state_dict単体
    """
    obj = torch.load(agent_path, map_location=device)

    if isinstance(obj, dict) and "agent_state_dict" in obj:
        sd = obj["agent_state_dict"]
        cfg = obj.get("cfg", None)
        return sd, cfg

    if isinstance(obj, dict):
        # state_dict単体
        return obj, None

    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def to_dreamer_config(cfg: DictConfig, device_str: str, num_envs_eval: int = 1) -> SimpleNamespace:
    """
    train.py と同様に DictConfig -> SimpleNamespace へ変換する。
    評価用なので envs は基本 1 に寄せる。
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    container = dict(container)

    container.setdefault("device", device_str)
    container.setdefault("size", [int(container.get("obs_size", 84)), int(container.get("obs_size", 84))])

    # dreamerv3-torch が参照するパス類
    container.setdefault("logdir", "logs")
    container.setdefault("traindir", None)
    container.setdefault("evaldir", None)
    container.setdefault("offline_traindir", "")
    container.setdefault("offline_evaldir", "")

    container.setdefault("envs", int(num_envs_eval))
    container.setdefault("action_repeat", int(container.get("frameskip", 4)))
    container.setdefault("time_limit", int(container.get("time_limit", 0)))

    return SimpleNamespace(**container)


# ---------------------------------------------------------------------------
# Gym wrappers (train.py と揃える)
# ---------------------------------------------------------------------------
class OneHotActionCompat(gym.ActionWrapper):
    """
    Dreamer側は one-hot(Box) を流すことが多い。
    Craftium環境は Discrete を想定するので、one-hot -> int に戻す。
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(env.action_space, "n"):
            raise TypeError(f"OneHotActionCompat expects Discrete-like action_space, got: {env.action_space}")
        self._n = int(env.action_space.n)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self._n,), dtype=np.float32)

    def action(self, action):
        # Dreamerが dict を返すことがある
        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]
            else:
                raise TypeError(f"Unsupported action dict keys: {list(action.keys())}")

        if isinstance(action, (int, np.integer)):
            return int(action)

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        a = np.asarray(action)
        if a.ndim == 0:
            return int(a)

        idx = np.argmax(a, axis=-1)
        if isinstance(idx, np.ndarray):
            idx = idx.item()
        return int(idx)


class GymV26ToV21(gym.Wrapper):
    """
    gymnasium (obs, reward, terminated, truncated, info)
      -> gym互換 (obs, reward, done, info)
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        info = dict(info) if info is not None else {}
        info.setdefault("terminated", terminated)
        info.setdefault("truncated", truncated)
        info.setdefault("discount", np.array(0.0 if terminated else 1.0, dtype=np.float32))

        # ChopTree: episode total reward = chops（仮定）
        if done and isinstance(info.get("episode"), dict):
            ep_r = info["episode"].get("r", None)
            if ep_r is not None:
                info["episode"]["chops"] = as_float_scalar(ep_r)

        return obs, reward, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
            return obs
        return result


class CraftiumObsDictWrapper(gym.Wrapper):
    """
    Dreamerが要求する dict観測へ変換する。
      obs_dict["image"] = HWC uint8
      obs_dict["is_first"] / ["is_terminal"]
    """

    def __init__(self, env, obs_key: str = "image", channel_last: bool = True):
        super().__init__(env)
        self._obs_key = obs_key
        self._channel_last = channel_last

        raw_space = self.env.observation_space
        if not isinstance(raw_space, gym.spaces.Box):
            raise ValueError("CraftiumObsDictWrapper expects Box observation space")

        img_shape = raw_space.shape
        if channel_last:
            # input is CHW -> output is HWC
            h, w = img_shape[1], img_shape[2]
            c = img_shape[0]
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


# ---------------------------------------------------------------------------
# Env builder (train.py と同様の前処理)
# ---------------------------------------------------------------------------
def make_eval_env(
    env_id: str,
    mt_wd: str,
    mt_port: int,
    frameskip: int,
    obs_size: int,
    seed: int,
    record_video: bool,
    video_dir: Optional[str],
    name_prefix: str,
    time_limit: int = 0,
):
    craftium_kwargs = dict(
        run_dir_prefix=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        rgb_observations=True,
        mt_listen_timeout=300_000,
        seed=int(seed),
    )

    if record_video:
        if video_dir is None:
            raise ValueError("record_video=True requires video_dir")
        os.makedirs(video_dir, exist_ok=True)

        env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix=_safe_prefix(name_prefix),
            episode_trigger=lambda episode_id: True,  # 全エピソード保存
        )
    else:
        env = gym.make(env_id, **craftium_kwargs)

    # train.py と同じ
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, int(obs_size))
    env = gym.wrappers.FrameStack(env, 1)

    env = GymV26ToV21(env)
    env = CraftiumObsDictWrapper(env, obs_key="image", channel_last=True)
    if int(time_limit) > 0:
        env = wrappers.TimeLimit(env, int(time_limit))
    env = OneHotActionCompat(env)
    env = wrappers.UUID(env)
    return env


# ---------------------------------------------------------------------------
# Dataset (可能なら既存train_epsから作る)
# ---------------------------------------------------------------------------
def find_train_eps_dir_near(agent_path: str, max_up: int = 6) -> Optional[pathlib.Path]:
    """
    Hydra run dir の logs/train_eps を探す。
    例: output/exp_dreamer/xxxx/agents/agent_step_....pt
        output/exp_dreamer/xxxx/logs/train_eps
    """
    p = pathlib.Path(agent_path).resolve().parent
    cur = p
    for _ in range(max_up):
        cand = cur / "logs" / "train_eps"
        if cand.exists() and cand.is_dir():
            return cand
        cur = cur.parent
    return None


def make_train_dataset_for_init(
    dreamer_config: SimpleNamespace,
    agent_path: str,
) -> Any:
    """
    Dreamer初期化用に dataset を用意する。
    既存 replay があればそれを使う。なければダミーで回避する。
    """
    eps_dir = find_train_eps_dir_near(agent_path)
    if eps_dir is not None:
        try:
            eps = tools.load_episodes(eps_dir, limit=int(getattr(dreamer_config, "dataset_size", 100000)))
            if isinstance(eps, dict) and len(eps) > 0:
                return dv3.make_dataset(eps, dreamer_config)
        except Exception:
            pass

    # fallback: 無限ダミー（評価だけなら基本参照されない想定）
    def _dummy():
        while True:
            yield {}

    return _dummy()


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------
def evaluate(
    agent_path: str,
    cfg_path: Optional[str],
    env_id: str,
    num_episodes: int,
    seed: int,
    device_str: str,
    mt_wd: str,
    mt_port: int,
    record_video: bool,
    video_dir: Optional[str],
    output_json: Optional[str],
    deterministic: bool,
):
    log = logging.getLogger(__name__)

    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")
    log.info(f"device={device}")
    log.info(f"agent_path={agent_path}")
    log.info(f"env_id={env_id}")
    log.info(f"episodes={num_episodes}")
    log.info(f"seed={seed}")
    log.info(f"mt_port={mt_port}")
    log.info(f"video={record_video} video_dir={video_dir}")

    # load weights (+cfg if checkpoint)
    sd, ckpt_cfg = load_agent_state_dict_and_maybe_cfg(agent_path, device)

    # cfg決定
    if ckpt_cfg is not None:
        cfg = OmegaConf.create(ckpt_cfg)
    else:
        cfg = load_cfg_anyhow(agent_path, cfg_path)

    # 評価側で上書きしたいパラメータ
    # （アーキ形状に影響しない範囲のみ）
    cfg.env_id = env_id
    cfg.seed = int(seed)
    cfg.num_envs = 1
    cfg.capture_video = bool(record_video)

    # Dreamer config
    device_str2 = "cuda" if device.type == "cuda" else "cpu"
    dreamer_config = to_dreamer_config(cfg, device_str2, num_envs_eval=1)

    # 評価なので学習用挙動を止める（アーキ形状は変えない）
    try:
        dreamer_config.pretrain = 0
    except Exception:
        pass
    try:
        dreamer_config.prefill = 0
    except Exception:
        pass

    # env
    obs_size = int(getattr(cfg, "obs_size", 84))
    frameskip = int(getattr(cfg, "frameskip", 4))
    time_limit = int(getattr(cfg, "time_limit", 0))
    name_prefix = f"eval_seed{seed}"

    env = make_eval_env(
        env_id=env_id,
        mt_wd=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        obs_size=obs_size,
        seed=seed,
        record_video=record_video,
        video_dir=video_dir,
        name_prefix=name_prefix,
        time_limit=time_limit,
    )

    # dreamer_config.num_actions を合わせる
    acts = env.action_space
    dreamer_config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # logger (最低限)
    # tools.Logger は logdir を Path で持つので eval用にローカル作成する
    eval_logdir = pathlib.Path(getattr(dreamer_config, "logdir", "logs")).expanduser()
    eval_logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(eval_logdir, 0)

    # dataset（初期化のために可能なら replay を拾う）
    train_dataset = make_train_dataset_for_init(dreamer_config, agent_path)

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
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    log.info(f"loaded_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        log.info(f"missing_keys(sample): {missing[:5]}")
    if len(unexpected) > 0:
        log.info(f"unexpected_keys(sample): {unexpected[:5]}")

    # ここが True のままだと「最初の1回だけ pretrain」等が走る実装があるので潰す
    if hasattr(agent, "_should_pretrain") and hasattr(agent._should_pretrain, "_once"):
        agent._should_pretrain._once = False

    # 評価実行は train.py と同じ simulate ルートを使う
    # envは Damy で包む（trainと同じ）
    eval_envs = [Damy(env)]

    # eval_eps を作って simulate で episodes=num_episodes 回す
    eval_eps_dir = pathlib.Path("eval_eps")
    eval_eps_dir.mkdir(parents=True, exist_ok=True)
    eval_eps = tools.load_episodes(eval_eps_dir, limit=1)

    prev_keys = set(eval_eps.keys()) if isinstance(eval_eps, dict) else set()

    # deterministic は dreamer側の実装依存なので「best-effort」で反映する
    # 基本は training=False + is_eval=True で greedy 寄りになる想定
    eval_policy = functools.partial(agent, training=False)

    tools.simulate(
        eval_policy,
        eval_envs,
        eval_eps,
        eval_eps_dir,
        logger,
        is_eval=True,
        episodes=int(num_episodes),
    )

    # 新規エピソード抽出
    new_keys = [k for k in eval_eps.keys() if k not in prev_keys] if isinstance(eval_eps, dict) else []
    new_keys = sorted(new_keys)

    episode_returns = []
    episode_lengths = []
    episode_chops = []

    for i, k in enumerate(new_keys[:num_episodes]):
        ep = eval_eps.get(k, None)
        if not isinstance(ep, dict):
            continue

        rew = ep.get("reward", None)
        if rew is None:
            continue

        rsum = float(np.asarray(rew).sum())
        tlen = int(np.asarray(rew).shape[0])

        # ChopTree仮定: total reward == chops
        chops = rsum

        episode_returns.append(rsum)
        episode_lengths.append(tlen * frameskip)  # 実ステップ換算
        episode_chops.append(chops)

        log.info(f"episode {i+1}/{num_episodes}: return={rsum:.1f} length={tlen*frameskip} chops={chops:.1f}")

    # close
    try:
        env.close()
    except Exception:
        pass

    # stats
    if len(episode_returns) == 0:
        raise RuntimeError("No episodes were recorded. env launch or simulate may have failed.")

    results = {
        "agent_path": agent_path,
        "env_id": env_id,
        "num_episodes": int(num_episodes),
        "deterministic_best_effort": bool(deterministic),
        "seed": int(seed),
        "frameskip": int(frameskip),
        "obs_size": int(obs_size),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_chops": episode_chops,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_chops": float(np.mean(episode_chops)),
        "std_chops": float(np.std(episode_chops)),
        "video_dir": video_dir if record_video else None,
        "eval_eps_dir": str(eval_eps_dir),
    }

    print("\n" + "=" * 60)
    print("Evaluation Results (Dreamer)")
    print("=" * 60)
    print(f"  Mean Return:  {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Min/Max:      {results['min_return']:.1f} / {results['max_return']:.1f}")
    print(f"  Mean Length:  {results['mean_length']:.0f} ± {results['std_length']:.0f}")
    print(f"  Mean Chops:   {results['mean_chops']:.2f} ± {results['std_chops']:.2f}")
    print("=" * 60)

    # json
    if output_json is not None and str(output_json).strip() != "":
        outp = pathlib.Path(output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Eval] results saved: {outp}")

    if record_video and video_dir is not None:
        print(f"[Eval] videos saved: {video_dir}")

    return results


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate Craftium Dreamer agent (single seed)")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to .pt (checkpoint or state_dict)")
    parser.add_argument("--config", type=str, default=None, help="Optional config yaml (if hydra config not found)")
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")

    parser.add_argument("--mt-wd", type=str, default="./eval_minetest_runs", help="Minetest run directory prefix")
    parser.add_argument("--mt-port", type=int, default=49200, help="Minetest port for eval")

    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--video-dir", type=str, default=None, help="Video output directory")

    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation results to JSON")
    parser.add_argument("--deterministic", action="store_true", help="Best-effort deterministic evaluation")

    args = parser.parse_args()

    # video dir default
    record_video = (not args.no_video)
    if record_video and args.video_dir is None:
        ts = int(time.time())
        args.video_dir = f"./eval_videos/dreamer_eval_{ts}"
    if record_video:
        os.makedirs(args.video_dir, exist_ok=True)

    os.makedirs(args.mt_wd, exist_ok=True)

    evaluate(
        agent_path=args.agent_path,
        cfg_path=args.config,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        seed=args.seed,
        device_str=args.device,
        mt_wd=args.mt_wd,
        mt_port=args.mt_port,
        record_video=record_video,
        video_dir=args.video_dir if record_video else None,
        output_json=args.output_json,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
