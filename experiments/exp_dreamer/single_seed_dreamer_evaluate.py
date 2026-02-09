# -*- coding: utf-8 -*-
"""
Dreamer 学習済みエージェント評価 (Single-seed / Multi-seed)

主な機能:
- checkpoint 内の `cfg` (Training時のHydra設定) を優先して読み込み、同一構造の Dreamer を再構築してロードします。
- `cfg` が含まれない checkpoint の場合でも、形状が一致する重みのみをロードするフォールバック機能 (strict=False) を備えています。

出力:
- 各エピソードの `chops` 数 (報酬合計)
- シード内の `chops` の最小/最大エピソード番号 (同率の場合は全て表示)
"""

import envs.wrappers as dv3_wrappers  # type: ignore
import dreamer as dv3  # type: ignore
import os
import sys
import json
import time
import argparse
import pathlib
import socket
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
import torch

import craftium

# ---------------------------------------------------------------------------
# Path setup for dreamerv3-torch modules
# ---------------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1] if len(
    THIS_DIR.parents) >= 4 else THIS_DIR
DREAMER_ROOT = WORKSPACE_ROOT / "dreamerv3-torch"
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pick_free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def as_bool(x: Any, default: bool) -> bool:
    try:
        return bool(x)
    except Exception:
        return bool(default)


def extract_state_dict_and_cfg(loaded_obj: Any) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """
    返すもの:
      - state_dict: model.load_state_dict に入れる辞書
      - cfg_dict: 訓練時 Hydra cfg
    """
    if isinstance(loaded_obj, dict):
        cfg_dict = loaded_obj.get("cfg", None)
        if cfg_dict is not None and not isinstance(cfg_dict, dict):
            cfg_dict = None

        if "agent_state_dict" in loaded_obj and isinstance(loaded_obj["agent_state_dict"], dict):
            return loaded_obj["agent_state_dict"], cfg_dict

        # items_to_save 形式 (latest.pt 等): {"agent_state_dict":..., "optims_state_dict":...} は上で拾う
        # それ以外: dict がそのまま state_dict である可能性
        looks_like_state_dict = all(isinstance(k, str) and isinstance(
            v, torch.Tensor) for k, v in loaded_obj.items())
        if looks_like_state_dict:
            return loaded_obj, cfg_dict

    raise ValueError(
        "checkpoint の形式を解釈できない（agent_state_dict も state_dict 形式でもない）")


def make_dreamer_config_from_training_cfg(
    training_cfg: Dict[str, Any],
    device_str: str,
    num_envs: int,
    num_actions: int,
    deterministic_eval: bool,
) -> SimpleNamespace:
    """
    train.py の to_dreamer_config と同様のロジック:
      - training_cfg をベースにし、Dreamer 初期化に必要なキーを補完します。
    """
    cfg = dict(training_cfg)

    # device / shape
    obs_size = as_int(cfg.get("obs_size", cfg.get("size", [84, 84])[
                      0] if isinstance(cfg.get("size"), list) else 84), 84)
    cfg.setdefault("device", device_str)
    cfg.setdefault("size", [obs_size, obs_size])

    # dirs (評価では実ファイル保存を必須にしない)
    cfg.setdefault("logdir", "logs")
    cfg.setdefault("traindir", None)
    cfg.setdefault("evaldir", None)
    cfg.setdefault("offline_traindir", "")
    cfg.setdefault("offline_evaldir", "")

    # envs / actions
    cfg.setdefault("envs", int(num_envs))
    cfg["num_actions"] = int(num_actions)

    # 訓練 cfg の action_repeat を尊重する（env 側で frameskip 済みの運用があり得る）
    if "action_repeat" in cfg:
        cfg["action_repeat"] = as_int(cfg["action_repeat"], 1)
    else:
        cfg["action_repeat"] = 1

    cfg.setdefault("time_limit", as_int(cfg.get("time_limit", 0), 0))

    # 評価モードにおける決定性 (Deterministic Evaluation)
    if deterministic_eval:
        cfg["eval_state_mean"] = True

    return SimpleNamespace(**cfg)


def load_state_dict_no_crash(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    まず strict=True を試す。
    失敗したら shape が合うキーだけにフィルタして strict=False で落とさない。
    """
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
        return {
            "mode": "strict",
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
            "filtered_out": [],
        }
    except Exception as first_error:
        model_sd = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        filtered_out = []
        for key, value in state_dict.items():
            if key not in model_sd:
                filtered_out.append(key)
                continue
            if tuple(model_sd[key].shape) != tuple(value.shape):
                filtered_out.append(key)
                continue
            filtered[key] = value
        incompatible = model.load_state_dict(filtered, strict=False)
        return {
            "mode": "filtered",
            "first_error": str(first_error),
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
            "filtered_out": filtered_out,
        }


# ---------------------------------------------------------------------------
# Gymnasium -> gym compatibility and Craftium dict observation wrapper
# ---------------------------------------------------------------------------
class OneHotActionCompat(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not hasattr(env.action_space, "n"):
            raise TypeError(
                f"OneHotActionCompat expects Discrete-like action_space, got: {env.action_space}")
        self._n = int(env.action_space.n)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._n,), dtype=np.float32)

    def action(self, action):
        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]
            else:
                raise TypeError(
                    f"Unsupported action dict keys: {list(action.keys())}")

        if isinstance(action, (int, np.integer)):
            return int(action)

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        arr = np.asarray(action)

        # shape=(1,) や (1,1) を「離散index」として扱う
        if arr.ndim == 0 or arr.size == 1:
            return int(arr.reshape(()))

        # one-hot / logits (n,) または (1,n) は argmax
        idx = np.argmax(arr, axis=-1)
        if isinstance(idx, np.ndarray):
            idx = idx.item()
        return int(idx)


class GymV26ToV21(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        info = dict(info) if info is not None else {}
        info.setdefault("terminated", terminated)
        info.setdefault("truncated", truncated)
        info.setdefault("discount", np.array(
            0.0 if terminated else 1.0, dtype=np.float32))
        return obs, reward, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
            return obs
        return result


class CraftiumObsDictWrapper(gym.Wrapper):
    def __init__(self, env, obs_key: str = "image", channel_last: bool = True):
        super().__init__(env)
        self._obs_key = obs_key
        self._channel_last = channel_last

        raw_space = self.env.observation_space
        if not isinstance(raw_space, gym.spaces.Box):
            raise ValueError(
                "CraftiumObsDictWrapper expects Box observation space")

        img_shape = raw_space.shape
        if channel_last:
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
        return {"image": self._to_image(obs), "is_first": True, "is_terminal": False}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(info.get("terminated", done))
        obs_dict = {"image": self._to_image(
            obs), "is_first": False, "is_terminal": terminated}
        return obs_dict, reward, done, info


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------
def make_eval_env(
    env_id: str,
    mt_wd: str,
    mt_port: int,
    frameskip: int,
    seed: int,
    record_video: bool,
    video_dir: str,
    obs_size: int,
):
    if mt_port <= 0:
        mt_port = pick_free_tcp_port()

    craftium_kwargs = dict(
        run_dir_prefix=mt_wd,
        mt_port=int(mt_port),
        frameskip=int(frameskip),
        rgb_observations=True,
        seed=int(seed),
    )

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, episode_trigger=lambda _ep: True)
    else:
        env = gym.make(env_id, **craftium_kwargs)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, int(obs_size))
    env = gym.wrappers.FrameStack(env, 1)

    env = GymV26ToV21(env)
    env = CraftiumObsDictWrapper(env, obs_key="image", channel_last=True)

    env = OneHotActionCompat(env)
    env = dv3_wrappers.UUID(env)  # 訓練と揃える（ID 付与）
    return env


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------
def evaluate_one_seed(
    agent_path: str,
    env_id: str,
    num_episodes: int,
    seed: int,
    device_str: str,
    record_video: bool,
    video_dir: str,
    mt_wd: str,
    mt_port: int,
    deterministic: bool,
    prefer_ckpt_cfg: bool,
    frameskip_arg: int,
    obs_size_arg: int,
    probe_actions: bool,
):
    device = torch.device(device_str if (
        torch.cuda.is_available() and str(device_str).startswith("cuda")) else "cpu")

    loaded_ckpt = torch.load(
        agent_path, map_location="cpu", weights_only=False)
    agent_state_dict, training_cfg = extract_state_dict_and_cfg(loaded_ckpt)

    # cfg を優先して env 設定を決める
    if prefer_ckpt_cfg and isinstance(training_cfg, dict):
        frameskip = as_int(training_cfg.get(
            "frameskip", frameskip_arg), frameskip_arg)
        obs_size = as_int(training_cfg.get(
            "obs_size", obs_size_arg), obs_size_arg)
        action_repeat = as_int(training_cfg.get("action_repeat", 1), 1)
        time_limit = as_int(training_cfg.get("time_limit", 0), 0)
    else:
        frameskip = int(frameskip_arg)
        obs_size = int(obs_size_arg)
        action_repeat = 1
        time_limit = 0

    # env
    env = make_eval_env(
        env_id=env_id,
        mt_wd=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        seed=seed,
        record_video=record_video,
        video_dir=video_dir,
        obs_size=obs_size,
    )

    def to_scalar(x) -> float:
        return float(np.asarray(x).reshape(()))

    def angle_delta(prev: float, cur: float) -> float:
        wrap = 360.0 if (abs(prev) > 10.0 or abs(cur)
                         > 10.0) else (2.0 * np.pi)
        d = cur - prev
        d = (d + wrap / 2.0) % wrap - wrap / 2.0
        return float(d)

    def to_vec3(x) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        return v[:3] if v.size >= 3 else np.pad(v, (0, 3 - v.size), constant_values=0.0)

    def run_action_probe(env, steps: int = 20):
        action_count = int(env.action_space.shape[0]) if hasattr(
            env.action_space, "shape") else 0
        print(f"[Probe] action_count={action_count} steps={steps}")
        if action_count <= 0:
            print("[Probe] action_space が想定外なので中止")
            return

        for action_index in range(action_count):
            obs = env.reset()
            prev_img = obs["image"].astype(np.int16)

            prev_yaw = None
            prev_pitch = None
            prev_pos = None

            sum_absdiff = 0.0
            sum_reward = 0.0
            sum_abs_dyaw = 0.0
            sum_abs_dpitch = 0.0
            sum_dpos = 0.0
            count_delta = 0

            done = False

            onehot = np.zeros((action_count,), dtype=np.float32)
            onehot[action_index] = 1.0

            for t in range(steps):
                obs, reward, done, info = env.step(onehot)

                cur_img = obs["image"].astype(np.int16)
                sum_absdiff += float(np.mean(np.abs(cur_img - prev_img)))
                prev_img = cur_img

                sum_reward += float(reward)

                yaw = to_scalar(info.get("player_yaw", 0.0))
                pitch = to_scalar(info.get("player_pitch", 0.0))
                pos = to_vec3(
                    info.get("player_pos", np.zeros(3, dtype=np.float32)))

                if prev_yaw is not None:
                    sum_abs_dyaw += abs(angle_delta(prev_yaw, yaw))
                    sum_abs_dpitch += abs(angle_delta(prev_pitch, pitch))
                    sum_dpos += float(np.linalg.norm(pos - prev_pos))
                    count_delta += 1

                prev_yaw, prev_pitch, prev_pos = yaw, pitch, pos

                if done:
                    break

            denom_img = max(1, (t + 1))
            denom_delta = max(1, count_delta)

            mean_absdiff = sum_absdiff / denom_img
            mean_abs_dyaw = sum_abs_dyaw / denom_delta
            mean_abs_dpitch = sum_abs_dpitch / denom_delta
            mean_dpos = sum_dpos / denom_delta

            print(
                f"[Probe] action={action_index} "
                f"mean_absdiff={mean_absdiff:.2f} "
                f"mean_abs_dyaw={mean_abs_dyaw:.4f} "
                f"mean_abs_dpitch={mean_abs_dpitch:.4f} "
                f"mean_dpos={mean_dpos:.4f} "
                f"reward_sum={sum_reward:.2f} done={done}"
            )

    if probe_actions:
        run_action_probe(env, steps=20)
        env.close()
        return {"probe_only": True}

    if time_limit > 0:
        env = dv3_wrappers.TimeLimit(env, int(time_limit))

    obs_space = env.observation_space
    act_space = env.action_space
    num_actions = int(act_space.shape[0]) if hasattr(
        act_space, "shape") else int(getattr(act_space, "n"))

    # Dreamer config は cfg から復元して形状一致を最優先する
    if isinstance(training_cfg, dict):
        dreamer_config = make_dreamer_config_from_training_cfg(
            training_cfg=training_cfg,
            device_str=str(device_str),
            num_envs=1,
            num_actions=num_actions,
            deterministic_eval=deterministic,
        )
    else:
        dreamer_config = SimpleNamespace(
            device=str(device_str),
            size=[obs_size, obs_size],
            action_repeat=int(action_repeat),
            compile=False,
            precision=32,
            debug=False,
            eval_state_mean=bool(deterministic),
            num_actions=int(num_actions),
        )

    # logger と dataset は評価では使わない前提でダミー
    class DummyLogger:
        def __init__(self, step=0):
            self.step = step

        def scalar(self, *_args, **_kwargs):
            return None

        def image(self, *_args, **_kwargs):
            return None

        def video(self, *_args, **_kwargs):
            return None

        def write(self, *_args, **_kwargs):
            return None

    dummy_logger = DummyLogger(step=0)
    dummy_dataset = iter([])

    agent = dv3.Dreamer(obs_space, act_space, dreamer_config,
                        dummy_logger, dummy_dataset).to(device)
    agent.requires_grad_(requires_grad=False)

    # 重みロード
    agent_state_dict = {k: v.to(device) for k, v in agent_state_dict.items()}
    load_report = load_state_dict_no_crash(agent, agent_state_dict)

    lr = load_report
    print(
        f"[Load] mode={lr.get('mode')} "
        f"filtered_out={len(lr.get('filtered_out', []))} "
        f"missing={len(lr.get('missing_keys', []))} "
        f"unexpected={len(lr.get('unexpected_keys', []))}"
    )

    episode_rewards = []
    episode_lengths = []
    episode_chops = []
    episode_records = []  # エピソード単位の詳細

    def safe_reset(env, seed_value: Optional[int] = None):
        try:
            if seed_value is None:
                return env.reset()
            return env.reset(seed=int(seed_value))
        except TypeError:
            return env.reset()

    for episode_index in range(int(num_episodes)):
        obs = safe_reset(env, int(seed) + episode_index)

        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_chop_count = 0
        agent_state = None
        action_index_hist = []

        # episode 初期化: kinematics 集計用
        prev_yaw = None
        prev_pitch = None
        prev_pos = None
        action_counts = np.zeros((num_actions,), dtype=np.int64)
        sum_abs_dyaw = np.zeros((num_actions,), dtype=np.float64)
        sum_abs_dpitch = np.zeros((num_actions,), dtype=np.float64)
        sum_dpos = np.zeros((num_actions,), dtype=np.float64)
        count_delta = np.zeros((num_actions,), dtype=np.int64)

        while not done:
            obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            reset_flag = np.asarray(
                [bool(obs.get("is_first", False))], dtype=np.bool_)

            with torch.no_grad():
                policy_output, agent_state = agent(
                    obs_batch, reset_flag, agent_state, training=False)

            # 予測 reward と実 reward のズレを見る（最初の数 step だけ）
            if episode_length < 30:
                latent_torch, action_torch = agent_state

                # (A) 現在 latent の reward 予測
                try:
                    feat = agent._wm.dynamics.get_feat(latent_torch)
                    reward_pred = agent._wm.heads["reward"](feat).mean()
                    reward_pred_value = float(
                        reward_pred.detach().cpu().item())
                except Exception:
                    reward_pred_value = None

                # (B) 選んだ action を入れた「次 latent」の reward 予測（actor が実際に最適化している側に近い）
                try:
                    next_latent = agent._wm.dynamics.img_step(
                        latent_torch, action_torch)
                    next_feat = agent._wm.dynamics.get_feat(next_latent)
                    reward_pred_next = agent._wm.heads["reward"](
                        next_feat).mean()
                    reward_pred_next_value = float(
                        reward_pred_next.detach().cpu().item())
                except Exception:
                    reward_pred_next_value = None

            # actor 分布を観測（最初の数 step だけ）
            if episode_length < 10:
                # state=(latent, action)
                latent_torch, _prev_action = agent_state
                feat = agent._wm.dynamics.get_feat(latent_torch)
                actor_dist = agent._task_behavior.actor(feat)

                entropy_value = None
                if hasattr(actor_dist, "entropy"):
                    try:
                        entropy_value = float(
                            actor_dist.entropy().mean().detach().cpu().item())
                    except Exception:
                        entropy_value = None

                logits = getattr(actor_dist, "logits", None)
                if logits is None and hasattr(actor_dist, "base_dist") and hasattr(actor_dist.base_dist, "logits"):
                    logits = actor_dist.base_dist.logits

                if logits is not None:
                    probs = torch.softmax(logits, dim=-1)
                    topk = min(3, probs.shape[-1])
                    topv, topi = torch.topk(probs, k=topk, dim=-1)
                    top_idx = topi[0].detach().cpu().tolist()
                    top_p = [float(x) for x in topv[0].detach().cpu().tolist()]
                    print(
                        f"[ActorDist] entropy={entropy_value} top{topk}_idx={top_idx} top{topk}_p={top_p}")
                else:
                    print(
                        f"[ActorDist] entropy={entropy_value} dist_type={type(actor_dist)}")

            action = policy_output.get("action", policy_output)

            # --- DEBUG: action の中身を観測する ---
            if torch.is_tensor(action):
                action_np = action.detach().cpu().numpy()
            else:
                action_np = np.asarray(action)

            if action_np.ndim == 0 or action_np.size == 1:
                action_index = int(action_np.reshape(()))
                action_max = float(action_np.reshape(()))
            else:
                action_index = int(np.argmax(action_np, axis=-1).item() if hasattr(
                    np.argmax(action_np, axis=-1), "item") else np.argmax(action_np, axis=-1))
                action_max = float(np.max(action_np))

            action_index_hist.append(action_index)

            # step を先に行い、reward/info を確定させる
            next_obs, reward, done, info = env.step(action)

            # RewardCheck は reward が確定した後にだけ出す
            if episode_length < 30:
                print(
                    f"[RewardCheck] step={episode_length:03d} "
                    f"reward_pred={reward_pred_value} reward_pred_next={reward_pred_next_value} "
                    f"reward_env={float(reward)} action_index={action_index}"
                )

            # --- kinematics trace end ---

            episode_reward += float(reward)
            episode_length += 1

            if float(reward) >= 1.0:
                episode_chop_count += 1

            obs = next_obs

        episode_rewards.append(float(episode_reward))
        episode_lengths.append(int(episode_length * frameskip))
        episode_chops.append(int(episode_chop_count))

        episode_records.append(
            {
                "episode_index": int(episode_index),
                "reset_seed": int(seed) + int(episode_index),
                "chops": int(episode_chop_count),
                "reward_sum": float(episode_reward),
                "length_steps": int(episode_length),
                "length_env_frames": int(episode_length * frameskip),
            }
        )

        # エピソードごとの chops を表示
        print(
            f"[EpisodeChops] episode={episode_index} "
            f"seed={seed}+{episode_index}={int(seed)+int(episode_index)} "
            f"chops={episode_chop_count} "
            f"reward_sum={episode_reward:.2f} "
            f"len_steps={episode_length} len_frames={int(episode_length * frameskip)}"
        )

        unique_actions = sorted(set(action_index_hist))
        print(
            f"[EpisodeSummary] unique_actions={len(unique_actions)} first20={action_index_hist[:20]} last20={action_index_hist[-20:]}")
        print("[ActionHistogram]", {i: int(c)
              for i, c in enumerate(action_counts) if c > 0})

        for i in range(num_actions):
            if action_counts[i] <= 0:
                continue
            denom = max(1, int(count_delta[i]))
            print(
                f"[ActionKinematics] action={i} "
                f"count={int(action_counts[i])} "
                f"mean_abs_dyaw={float(sum_abs_dyaw[i]/denom):.4f} "
                f"mean_abs_dpitch={float(sum_abs_dpitch[i]/denom):.4f} "
                f"mean_dpos={float(sum_dpos[i]/denom):.4f}"
            )

    # seed 内の min/max を表示
    if episode_chops:
        min_ep = int(np.argmin(np.asarray(episode_chops)))
        max_ep = int(np.argmax(np.asarray(episode_chops)))
        print("\n" + "-" * 70)
        print("[ChopsMinMax]")
        print(
            f"  min: episode={min_ep} chops={episode_chops[min_ep]} reward_sum={episode_rewards[min_ep]:.2f} len_frames={episode_lengths[min_ep]}")
        print(
            f"  max: episode={max_ep} chops={episode_chops[max_ep]} reward_sum={episode_rewards[max_ep]:.2f} len_frames={episode_lengths[max_ep]}")
        min_val = int(np.min(np.asarray(episode_chops)))
        max_val = int(np.max(np.asarray(episode_chops)))
        min_eps = [i for i, v in enumerate(episode_chops) if v == min_val]
        max_eps = [i for i, v in enumerate(episode_chops) if v == max_val]
        print(f"[ChopsTies] min_chops={min_val} episodes={min_eps}")
        print(f"[ChopsTies] max_chops={max_val} episodes={max_eps}")
        print("-" * 70)

    env.close()

    return {
        "seed": int(seed),
        "agent_path": str(agent_path),
        "env_id": str(env_id),
        "num_episodes": int(num_episodes),
        "device": str(device),
        "used_obs_size": int(obs_size),
        "used_frameskip": int(frameskip),
        "used_action_repeat": int(getattr(dreamer_config, "action_repeat", 1)),
        "load_report": load_report,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_chops": episode_chops,
        "episode_records": episode_records,  # 追加
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_chops": float(np.mean(episode_chops)) if episode_chops else 0.0,
        "std_chops": float(np.std(episode_chops)) if episode_chops else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-num", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)

    parser.add_argument("--mt-wd", type=str, default="./eval_runs")
    parser.add_argument("--mt-port", type=int, default=0,
                        help="0 なら空きポートを自動選択する")

    parser.add_argument("--frameskip", type=int, default=4,
                        help="cfg が無い場合のフォールバック")
    parser.add_argument("--obs-size", type=int, default=84,
                        help="cfg が無い場合のフォールバック")

    parser.add_argument("--deterministic", action="store_true",
                        help="eval_state_mean を True にする")
    parser.add_argument("--prefer-ckpt-cfg", action="store_true",
                        help="checkpoint 内 cfg を優先する（推奨）")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--probe-actions", action="store_true",
                        help="各actionを固定入力して挙動を計測し終了")

    args = parser.parse_args()

    if args.video_dir is None:
        args.video_dir = f"./eval_videos/eval_{int(time.time())}"

    os.makedirs(args.mt_wd, exist_ok=True)
    if not args.no_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # seed の扱いは引数 seed を起点に +1
    eval_seeds = [int(args.seed) + i for i in range(int(args.seed_num))]

    all_results = []
    for seed_value in eval_seeds:
        per_seed_video_dir = args.video_dir
        if not args.no_video:
            per_seed_video_dir = os.path.join(
                args.video_dir, f"seed{seed_value}")
            os.makedirs(per_seed_video_dir, exist_ok=True)

        result = evaluate_one_seed(
            agent_path=args.agent_path,
            env_id=args.env_id,
            num_episodes=int(args.num_episodes),
            seed=int(seed_value),
            device_str=str(args.device),
            record_video=(not args.no_video),
            video_dir=per_seed_video_dir,
            mt_wd=str(args.mt_wd),
            mt_port=int(args.mt_port),
            deterministic=bool(args.deterministic),
            prefer_ckpt_cfg=bool(args.prefer_ckpt_cfg) or True,
            frameskip_arg=int(args.frameskip),
            obs_size_arg=int(args.obs_size),
            probe_actions=bool(args.probe_actions),
        )
        all_results.append(result)

        if bool(args.probe_actions):
            print("[Probe] finished")
            return

        lr = result["load_report"]
        print("\n" + "#" * 70)
        print(f"[Eval] seed={seed_value}")
        print(
            f"  used_obs_size={result['used_obs_size']} used_frameskip={result['used_frameskip']} action_repeat={result['used_action_repeat']}")
        print(f"  load_mode={lr.get('mode')}")
        if lr.get("mode") != "strict":
            print(f"  first_error={lr.get('first_error')}")
            print(
                f"  filtered_out={len(lr.get('filtered_out', []))} missing={len(lr.get('missing_keys', []))} unexpected={len(lr.get('unexpected_keys', []))}")
        print(
            f"  mean_reward={result['mean_reward']:.2f} mean_chops={result['mean_chops']:.2f}")
        print("#" * 70)

    mean_rewards = [r["mean_reward"] for r in all_results]
    mean_chops = [r["mean_chops"] for r in all_results]

    print("\n" + "=" * 60)
    print("Multi-Seed Summary")
    print("=" * 60)
    for r in all_results:
        print(
            f"  seed={r['seed']}: mean_reward={r['mean_reward']:.2f}, mean_chops={r['mean_chops']:.2f}")
    print("-" * 60)
    print(
        f"  over_seeds: mean_reward={float(np.mean(mean_rewards)):.2f} ± {float(np.std(mean_rewards)):.2f}")
    print(
        f"  over_seeds: mean_chops={float(np.mean(mean_chops)):.2f} ± {float(np.std(mean_chops)):.2f}")
    print("=" * 60)

    if args.output_json:
        out = {
            "runs": all_results,
            "agg": {
                "seed_num": int(args.seed_num),
                "seeds": eval_seeds,
                "mean_reward_over_seeds": float(np.mean(mean_rewards)),
                "std_reward_over_seeds": float(np.std(mean_rewards)),
                "mean_chops_over_seeds": float(np.mean(mean_chops)),
                "std_chops_over_seeds": float(np.std(mean_chops)),
            },
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[Eval] saved: {args.output_json}")


if __name__ == "__main__":
    main()
