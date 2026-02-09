# -*- coding: utf-8 -*-
"""
Craftium Dreamer Experiment Runner (Hydra + WandB)

主な機能:
- DreamerV3 エージェントを Craftium 環境で学習させるためのメインスクリプト。
- Hydra による設定管理、WandB による実験トラッキング、チェックポイント管理、ロギング機能を提供。

References:
    - craftium_exp/experiments/exp__lstmppo (logging/checkpoint/resume style)
    - dreamerv3-torch (https://github.com/NM512/dreamerv3-torch)
"""

import os
import sys
import json
import random
import time
import datetime
import logging
import pathlib
import functools
from types import SimpleNamespace
from typing import Optional, Dict, Any

import hydra
import numpy as np
import torch
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

import craftium

# ---------------------------------------------------------------------------
# Path setup for dreamerv3-torch modules (keeps working even after Hydra chdir)
# ---------------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).resolve().parent
print(THIS_DIR)
WORKSPACE_ROOT = THIS_DIR.parents[1] if len(THIS_DIR.parents) >= 4 else THIS_DIR
DREAMER_ROOT = WORKSPACE_ROOT / "dreamerv3-torch"
print(DREAMER_ROOT)
if str(DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER_ROOT))

import dreamer as dv3  # type: ignore
import models  # type: ignore
import tools  # type: ignore
import exploration as expl  # type: ignore
import envs.wrappers as wrappers  # type: ignore
from parallel import Parallel, Damy  # type: ignore

to_np = lambda x: x.detach().cpu().numpy()
"""PyTorch TensorをNumPy配列に変換するヘルパー関数。"""


def setup_logging():
    """ロギングシステムをセットアップする。

    logsディレクトリを作成し、コンソールとファイルの両方にログを出力するよう設定する。
    ファイルログは logs/train.log に保存される。
    """
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if len(logger.handlers) > 0:
        return

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler("logs/train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def log_new_episode_chops(
    log: logging.Logger,
    logger: "WandbLogger",
    eps: Dict[str, Any],
    prev_keys: set,
    prefix: str = "train",
    flush_now: bool = False,
):
    """
    tools.load_episodes() が作る eps(dict) に追加された「新しいエピソード」を検出し、
    そのエピソード報酬合計を chops としてログに出す。

    前提:
      ChopTree-v0 は「episode total reward == chops」になっている（あなたの仮定通り）
    """
    if eps is None:
        return

    new_keys = [k for k in eps.keys() if k not in prev_keys]
    if not new_keys:
        return

    chops_list = []
    for k in new_keys:
        ep = eps.get(k, None)
        if not isinstance(ep, dict):
            continue

        # dreamerv3-torch episode には reward 配列が入ることが多い
        rew = ep.get("reward", None)
        if rew is None:
            continue

        try:
            ep_chops = float(np.asarray(rew).sum())
        except Exception:
            continue

        chops_list.append(ep_chops)

    if not chops_list:
        return

    mean_chops = float(np.mean(chops_list))
    max_chops = float(np.max(chops_list))

    # loggerに入れる（次の logger.write() で表示される）
    logger.scalar(f"{prefix}/episode_chops_mean", mean_chops)
    logger.scalar(f"{prefix}/episode_chops_max", max_chops)

    # コンソールにも出す
    log.info(f"{prefix}_new_episode_chops: mean={mean_chops:.1f} max={max_chops:.1f} n={len(chops_list)}")

    # すぐ表示したいなら flush
    if flush_now:
        logger.write(step=int(logger.step))


# ---------------------------------------------------------------------------
# Gymnasium -> gym compatibility and Craftium dict observation wrapper
# ---------------------------------------------------------------------------
class OneHotActionCompat(gym.ActionWrapper):
    """One-hotエンコードされたアクションをDiscreteアクションに変換するラッパー。

    DreamerV3がone-hotまたはlogitsベクトルとしてアクションを出力する場合に、
    それをCraftiumが期待する整数アクションに変換する。

    Attributes:
        _n (int): アクション空間のサイズ（選択肢の数）
    """

    def __init__(self, env):
        """OneHotActionCompatを初期化する。

        Args:
            env: ラップするgym環境（Discrete-likeなaction_spaceを持つこと）

        Raises:
            TypeError: 環境のaction_spaceがDiscreteでない場合
        """
        super().__init__(env)
        if not hasattr(env.action_space, "n"):
            raise TypeError(f"OneHotActionCompat expects Discrete-like action_space, got: {env.action_space}")
        self._n = int(env.action_space.n)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self._n,), dtype=np.float32)

    def action(self, action):
        """アクションを整数インデックスに変換する。

        Args:
            action: 変換するアクション。以下の形式をサポート:
                - int/np.integer: そのまま返す
                - dict: action["action"]を抽出して処理
                - torch.Tensor/np.ndarray/list: argmaxを取って整数に変換

        Returns:
            int: 環境に渡す整数アクションインデックス

        Raises:
            TypeError: サポートされていないアクション形式の場合
        """
        # Helper: Dreamerが dict を返すケースへの対応
        if isinstance(action, dict):
            # Dreamerの出力はだいたい action["action"] に入っている
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

        # idxが配列(例 shape=(1,)) の場合でも安全にint化
        if isinstance(idx, np.ndarray):
            idx = idx.item()

        return int(idx)


def as_float_scalar(x) -> float:
    """様々な数値型を安全にfloatスカラーに変換する。

    Args:
        x: 変換する値。float, np.scalar, shape=(1,)配列, listなどをサポート

    Returns:
        float: 変換されたfloat値。ベクトルの場合は合計を返す。
    """
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    if a.size == 1:
        return float(a.reshape(()))  # or float(a.item())
    return float(a.sum())  # 万一ベクトルなら合計を採用


class GymV26ToV21(gym.Wrapper):
    """Gymnasium v26 APIをgym v21スタイルに変換するラッパー。

    Gymnasium の (obs, reward, terminated, truncated, info) 形式を
    gym の (obs, reward, done, info) 形式に変換する。
    """

    def step(self, action):
        """環境を1ステップ進め、v21形式で結果を返す。

        Args:
            action: 実行するアクション

        Returns:
            tuple: (observation, reward, done, info)
                - done は terminated または truncated が True の場合に True
                - info には discount, terminated, truncated が追加される
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        info = dict(info) if info is not None else {}
        info.setdefault("terminated", terminated)
        info.setdefault("truncated", truncated)
        info.setdefault("discount", np.array(0.0 if terminated else 1.0, dtype=np.float32))

        # ChopTree: episode total reward = chops
        if done and isinstance(info.get("episode"), dict):
            ep_r = info["episode"].get("r", None)
            if ep_r is not None:
                info["episode"]["chops"] = as_float_scalar(ep_r)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """環境をリセットし、初期観測を返す。

        Gymnasium の (obs, info) タプルを obs のみに変換する。

        Args:
            **kwargs: 環境のreset()に渡すキーワード引数

        Returns:
            観測（Gymnasiumのタプル形式を自動的にobsのみに変換）
        """
        # Gymnasium reset() -> (obs, info) を obs のみに潰す
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs
        return result




class CraftiumObsDictWrapper(gym.Wrapper):
    """画像観測をDreamerが必要とするdict形式に変換するラッパー。

    Craftiumからの画像観測を、DreamerV3が期待するフォーマット
    (image, is_first, is_terminal を含むdict) に変換する。

    Attributes:
        _obs_key (str): 画像を格納する辞書キー
        _channel_last (bool): チャンネル次元を最後に配置するかどうか
    """

    def __init__(self, env, obs_key: str = "image", channel_last: bool = True):
        """CraftiumObsDictWrapperを初期化する。

        Args:
            env: ラップするgym環境
            obs_key: 画像観測を格納する辞書キー（デフォルト: "image"）
            channel_last: Trueの場合、出力形状を(H, W, C)に変換（デフォルト: True）

        Raises:
            ValueError: 環境のobservation_spaceがBoxでない場合
        """
        super().__init__(env)
        self._obs_key = obs_key
        self._channel_last = channel_last

        raw_space = self.env.observation_space
        if isinstance(raw_space, gym.spaces.Box):
            img_shape = raw_space.shape
        else:
            img_shape = None

        if img_shape is None:
            raise ValueError("CraftiumObsDictWrapper expects Box observation space")

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
        """生の観測を画像形式に変換する。

        Args:
            obs: 変換する観測データ

        Returns:
            np.ndarray: uint8形式の画像配列（channel_lastの設定に応じて(H,W,C)または(C,H,W)）
        """
        arr = np.array(obs)
        if self._channel_last:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)

    def reset(self, **kwargs):
        """環境をリセットし、dict形式の初期観測を返す。

        Args:
            **kwargs: 環境のreset()に渡すキーワード引数

        Returns:
            dict: 'image', 'is_first', 'is_terminal'を含む観測辞書
        """
        obs = self.env.reset(**kwargs)
        obs_dict = {
            self._obs_key: self._to_image(obs),
            "is_first": True,
            "is_terminal": False,
        }
        return obs_dict

    def step(self, action):
        """環境を1ステップ進め、dict形式の観測を返す。

        Args:
            action: 実行するアクション

        Returns:
            tuple: (obs_dict, reward, done, info)
                - obs_dict: 'image', 'is_first', 'is_terminal'を含む観測辞書
        """
        obs, reward, done, info = self.env.step(action)
        terminated = bool(info.get("terminated", done))
        obs_dict = {
            self._obs_key: self._to_image(obs),
            "is_first": False,
            "is_terminal": terminated,
        }
        # pass through discount if present; simulate() picks it from info
        return obs_dict, reward, done, info


# ---------------------------------------------------------------------------
# Checkpoint helpers (agent + optimizers + RNG + run metadata)
# ---------------------------------------------------------------------------

def _find_latest_checkpoint_in_dir(ckpt_dir: str) -> Optional[str]:
    """指定ディレクトリ内の最新チェックポイントファイルを検索する。

    Args:
        ckpt_dir: チェックポイントを検索するディレクトリパス

    Returns:
        Optional[str]: 最新の.ptファイルの完全パス、見つからない場合はNone
    """
    if not ckpt_dir or (not os.path.isdir(ckpt_dir)):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None
    files = sorted(files)
    return os.path.join(ckpt_dir, files[-1])


def resolve_resume_path(resume_path: str) -> Optional[str]:
    """再開用チェックポイントのパスを解決する。

    ファイルパスまたはディレクトリパスを受け取り、実際のチェックポイントファイルパスを返す。

    Args:
        resume_path: チェックポイントファイルまたはディレクトリのパス

    Returns:
        Optional[str]: 解決されたチェックポイントファイルパス、見つからない場合はNone
    """
    if resume_path is None:
        return None
    p = str(resume_path).strip()
    if p == "":
        return None
    if os.path.isdir(p):
        return _find_latest_checkpoint_in_dir(p)
    if os.path.isfile(p):
        return p
    return None


def peek_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """チェックポイントファイルからメタデータのみを読み取る。

    完全なモデル復元を行わず、run_name, wandb_run_id, global_step等のメタ情報のみを取得する。

    Args:
        path: チェックポイントファイルのパス

    Returns:
        Dict[str, Any]: メタデータを含む辞書、読み取りに失敗した場合は空の辞書
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
        meta = {}
        for k in ["run_name", "wandb_run_id", "global_step", "iteration", "agent_step"]:
            if k in ckpt:
                meta[k] = ckpt[k]
        return meta
    except Exception:
        return {}


def save_checkpoint(
    path: str,
    agent: torch.nn.Module,
    global_step: int,
    iteration: int,
    cfg: DictConfig,
    run_name: str,
    wandb_run_id: Optional[str],
):
    """学習状態をチェックポイントファイルに保存する。

    エージェントの状態、オプティマイザの状態、乱数状態、実行メタデータを保存する。
    同時に latest.pt も更新する。

    Args:
        path: 保存先のファイルパス
        agent: 保存するDreamerエージェント
        global_step: 現在のグローバルステップ数
        iteration: 現在のイテレーション数
        cfg: Hydra設定オブジェクト
        run_name: 実行名
        wandb_run_id: WandBの実行ID（再開時に使用）
    """
    ckpt = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "global_step": int(global_step),
        "agent_step": int(getattr(agent, "_step", 0)),
        "iteration": int(iteration),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "run_name": str(run_name),
        "wandb_run_id": wandb_run_id,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, path)
    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    try:
        torch.save(ckpt, latest_path)
    except Exception:
        pass


def load_checkpoint(path: str, agent: torch.nn.Module, device: torch.device):
    """チェックポイントから学習状態を復元する。

    Args:
        path: チェックポイントファイルのパス
        agent: 状態を復元するDreamerエージェント
        device: 使用するtorchデバイス

    Returns:
        tuple: (global_step, iteration, agent_step, run_name, wandb_run_id)
            復元されたメタデータ
    """
    ckpt = torch.load(path, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, ckpt.get("optims_state_dict", {}))
    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
    if "rng_numpy" in ckpt:
        np.random.set_state(ckpt["rng_numpy"])
    if "rng_torch" in ckpt:
        torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    global_step = int(ckpt.get("global_step", 0))
    iteration = int(ckpt.get("iteration", 0))
    run_name = str(ckpt.get("run_name", ""))
    wandb_run_id = ckpt.get("wandb_run_id", None)
    agent_step = int(ckpt.get("agent_step", 0))
    return global_step, iteration, agent_step, run_name, wandb_run_id


# ---------------------------------------------------------------------------
# WandB-aware Logger compatible with tools.Logger
# ---------------------------------------------------------------------------
class WandbLogger(tools.Logger):
    """WandB連携機能を持つDreamerV3互換ロガー。

    tools.Loggerを継承し、WandBへのメトリクス自動送信機能を追加する。

    Attributes:
        _wandb_run: WandBの実行オブジェクト（Noneの場合はWandBへのログを無効化）
    """

    def __init__(self, logdir, step, wandb_run=None):
        """WandbLoggerを初期化する。

        Args:
            logdir: ログを保存するディレクトリ
            step: 初期ステップ数
            wandb_run: WandBの実行オブジェクト（オプション）
        """
        super().__init__(logdir, step)
        self._wandb_run = wandb_run

    def write(self, fps=False, step=False):
        """蓄積されたメトリクスを書き出す。

        コンソール、JSONLファイル、TensorBoard、WandBにメトリクスを出力する。

        Args:
            fps: Trueの場合、FPS（フレーム/秒）も計算してログに含める
            step: 記録するステップ数。Falseの場合は現在のstepを使用
        """
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)
        if self._wandb_run is not None and scalars:
            wandb_log = {k: v for k, v in scalars}
            self._wandb_run.log(wandb_log, step=step)
        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------

def make_env(cfg, idx: int, run_name: str, mode: str = "train"):
    """Craftium環境を作成するファクトリ関数を返す。

    環境にはグレースケール変換、リサイズ、フレームスタック、各種ラッパーが適用される。

    Args:
        cfg: Hydra設定オブジェクト
        idx: 環境インデックス（並列環境の識別に使用）
        run_name: 実行名（ビデオ保存のプレフィックスに使用）
        mode: "train" または "eval"（現在は未使用だが将来の拡張用）

    Returns:
        Callable: 呼び出すと環境インスタンスを返すthunk関数
    """
    def thunk():
        craftium_kwargs = dict(
            run_dir_prefix=cfg.mt_wd,
            mt_port=cfg.mt_port + idx,
            frameskip=cfg.frameskip,
            rgb_observations=True,
            mt_listen_timeout=300_000,
            seed=int(cfg.seed) + idx,
        )

        env = gym.make(cfg.env_id, render_mode="rgb_array" if (cfg.capture_video and idx == 0) else None, **craftium_kwargs)

        if cfg.capture_video and idx == 0:
            video_folder = "videos"
            os.makedirs(video_folder, exist_ok=True)
            safe_prefix = run_name.replace("/", "__")
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=safe_prefix)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, int(cfg.obs_size))
        env = gym.wrappers.FrameStack(env, 1)

        # convert to gym v21 API after gymnasium wrappers
        env = GymV26ToV21(env)
        env = CraftiumObsDictWrapper(env, obs_key="image", channel_last=True)
        if int(getattr(cfg, "time_limit", 0)) > 0:
            env = wrappers.TimeLimit(env, int(cfg.time_limit))
        env = OneHotActionCompat(env)
        print("action_space:", env.action_space, type(env.action_space))
        env = wrappers.UUID(env)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Utility: convert DictConfig to Dreamer-compatible SimpleNamespace
# ---------------------------------------------------------------------------

def to_dreamer_config(cfg: DictConfig, device_str: str) -> SimpleNamespace:
    """Hydra DictConfigをDreamerV3互換のSimpleNamespaceに変換する。

    DreamerV3が期待するデフォルト値を設定し、SimpleNamespaceとして返す。

    Args:
        cfg: Hydraから読み込まれた設定オブジェクト
        device_str: 使用するデバイス文字列（"cuda" または "cpu"）

    Returns:
        SimpleNamespace: DreamerV3で使用可能な設定オブジェクト
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    container = dict(container)
    container.setdefault("device", device_str)
    container.setdefault("size", [int(cfg.obs_size), int(cfg.obs_size)])
    container.setdefault("logdir", "logs")
    container.setdefault("traindir", None)
    container.setdefault("evaldir", None)
    container.setdefault("offline_traindir", "")
    container.setdefault("offline_evaldir", "")
    container.setdefault("envs", int(cfg.num_envs))
    container.setdefault("action_repeat", int(cfg.frameskip))
    container.setdefault("time_limit", int(getattr(cfg, "time_limit", 0)))
    return SimpleNamespace(**container)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """DreamerV3エージェントの学習を実行するメイン関数。

    Hydraによって設定が注入され、以下の処理を行う：
    1. ロギングとディレクトリのセットアップ
    2. シード設定と再現性の確保
    3. チェックポイントからの再開（オプション）
    4. WandB実験トラッキングの初期化（オプション）
    5. 環境とエージェントの作成
    6. メイン学習ループの実行
    7. 定期的なチェックポイント保存と評価

    Args:
        cfg: Hydraによって注入される設定オブジェクト
    """
    setup_logging()
    log = logging.getLogger(__name__)

    t = int(time.time())
    if cfg.seed is None:
        cfg.seed = t

    cfg.num_envs = int(cfg.num_envs)
    cfg.frameskip = int(cfg.frameskip)
    cfg.obs_size = int(cfg.obs_size)
    cfg.mt_port = int(cfg.mt_port)

    batch_time = t
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{batch_time}"

    os.makedirs("agents", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # resume handling
    resume_ckpt_path = None
    if bool(getattr(cfg, "resume", False)):
        rp = str(getattr(cfg, "resume_path", "")).strip()
        if rp == "":
            local_latest = resolve_resume_path(os.path.join("checkpoints", "latest.pt"))
            if local_latest is None:
                local_latest = resolve_resume_path("checkpoints")
            resume_ckpt_path = local_latest
        else:
            resume_ckpt_path = resolve_resume_path(rp)
        if resume_ckpt_path is None:
            log.warning("resume=true but no checkpoint found. start from scratch.")
        else:
            meta = peek_checkpoint_metadata(resume_ckpt_path)
            if isinstance(meta.get("run_name"), str) and meta["run_name"]:
                run_name = meta["run_name"]

    log.info(f"run_name={run_name}")
    log.info(f"output_dir={os.getcwd()}")

    # seed
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch.backends.cudnn.deterministic = bool(cfg.torch_deterministic)

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"
    log.info(f"device={device}")

    dreamer_config = to_dreamer_config(cfg, device_str)

    # W&B
    wandb_run = None
    wandb_run_id_for_resume = None
    if cfg.track:
        import wandb

        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        if bool(getattr(cfg, "resume", False)) and bool(getattr(cfg, "wandb_resume", True)) and resume_ckpt_path:
            meta = peek_checkpoint_metadata(resume_ckpt_path)
            wandb_run_id_for_resume = meta.get("wandb_run_id", None)
        if wandb_run_id_for_resume is not None:
            wandb_run = wandb.init(
                project=cfg.wandb_project_name,
                entity=cfg.wandb_entity,
                id=str(wandb_run_id_for_resume),
                resume="allow",
                name=run_name,
                config=wandb_cfg,
                save_code=True,
            )
        else:
            wandb_run = wandb.init(
                project=cfg.wandb_project_name,
                entity=cfg.wandb_entity,
                name=run_name,
                config=wandb_cfg,
                save_code=True,
            )
    
    # envs
    make_train = lambda i: make_env(cfg, i, run_name, mode="train")
    make_eval = lambda i: make_env(cfg, i, run_name, mode="eval")
    
    # train envs は常に作る
    train_envs = [make_train(i)() for i in range(cfg.num_envs)]
    
    # eval envs は eval_episode_num > 0 のときだけ作る
    eval_envs = []
    cfg.eval_episode_num = int(getattr(cfg, "eval_episode_num", 0))
    if cfg.eval_episode_num > 0:
        eval_envs = [make_eval(i)() for i in range(cfg.num_envs)]


    if getattr(cfg, "parallel", False):
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    acts = train_envs[0].action_space
    dreamer_config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # directories for replay/logs
    logdir = pathlib.Path(dreamer_config.logdir).expanduser()
    dreamer_config.traindir = dreamer_config.traindir or logdir / "train_eps"
    
    # eval は cfg.eval_episode_num > 0 のときだけ作る
    dreamer_config.eval_episode_num = int(getattr(cfg, "eval_episode_num", 0))
    dreamer_config.evaldir = None
    if dreamer_config.eval_episode_num > 0:
        dreamer_config.evaldir = (dreamer_config.evaldir or (logdir / "eval_eps"))
    
    logdir.mkdir(parents=True, exist_ok=True)
    dreamer_config.traindir.mkdir(parents=True, exist_ok=True)
    if dreamer_config.eval_episode_num > 0:
        pathlib.Path(dreamer_config.evaldir).mkdir(parents=True, exist_ok=True)
    
    step_in_episodes = dv3.count_steps(dreamer_config.traindir)
    logger = WandbLogger(logdir, dreamer_config.action_repeat * step_in_episodes, wandb_run=wandb_run)
    
    # offline / replay load
    if dreamer_config.offline_traindir:
        directory = dreamer_config.offline_traindir.format(**vars(dreamer_config))
    else:
        directory = dreamer_config.traindir
    train_eps = tools.load_episodes(directory, limit=dreamer_config.dataset_size)
    
    eval_eps = None
    if dreamer_config.eval_episode_num > 0:
        if dreamer_config.offline_evaldir:
            directory = dreamer_config.offline_evaldir.format(**vars(dreamer_config))
        else:
            directory = dreamer_config.evaldir
        eval_eps = tools.load_episodes(directory, limit=1)
    
    # datasets
    train_dataset = dv3.make_dataset(train_eps, dreamer_config)
    eval_dataset = None
    if dreamer_config.eval_episode_num > 0 and eval_eps is not None:
        eval_dataset = dv3.make_dataset(eval_eps, dreamer_config)


    agent = dv3.Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        dreamer_config,
        logger,
        train_dataset,
    ).to(device)
    agent.requires_grad_(requires_grad=False)

    global_step = int(logger.step)
    resume_iteration = 0
    if bool(getattr(cfg, "resume", False)) and resume_ckpt_path is not None:
        log.info(f"loading checkpoint: {resume_ckpt_path}")
        try:
            global_step, resume_iteration, agent_step, ckpt_run_name, ckpt_wandb_id = load_checkpoint(
                resume_ckpt_path, agent, device
            )
            if ckpt_run_name:
                run_name = ckpt_run_name
            if ckpt_wandb_id is not None and wandb_run is None and cfg.track:
                import wandb

                wandb_run = wandb.init(
                    project=cfg.wandb_project_name,
                    entity=cfg.wandb_entity,
                    id=str(ckpt_wandb_id),
                    resume="allow",
                    name=run_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    save_code=True,
                )
                logger._wandb_run = wandb_run
            if agent_step > 0:
                agent._step = agent_step
            agent._should_pretrain._once = False
            logger.step = max(logger.step, global_step)
            log.info(f"resumed: iteration={resume_iteration} global_step={global_step}")
        except Exception as e:
            log.warning(f"failed to load checkpoint: {e}. start from scratch.")
            global_step = int(logger.step)
            resume_iteration = 0

    start_time = time.time()
    iteration = int(resume_iteration)
    state = None

    # main training loop
    while agent._step < dreamer_config.steps + dreamer_config.eval_every:
        iteration += 1
        logger.write()

        if dreamer_config.eval_episode_num > 0:
            prev_eval_keys = set(eval_eps.keys()) if eval_eps is not None else set()
            
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                dreamer_config.evaldir,
                logger,
                is_eval=True,
                episodes=dreamer_config.eval_episode_num,
            )
                        
            
            if eval_eps is not None:
                log_new_episode_chops(
                    log=log,
                    logger=logger,
                    eps=eval_eps,
                    prev_keys=prev_eval_keys,
                    prefix="eval",
                    flush_now=True,
                )
                
            if dreamer_config.video_pred_log and (eval_dataset is not None):
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        prev_keys = set(train_eps.keys())
        
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            dreamer_config.traindir,
            logger,
            limit=dreamer_config.dataset_size,
            steps=dreamer_config.eval_every,
            state=state,
        )

        log_new_episode_chops(
            log=log,
            logger=logger,
            eps=train_eps,
            prev_keys=prev_keys,
            prefix="train",
            flush_now=True,
        )


        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, pathlib.Path(logdir) / "latest.pt")

        global_step = int(agent._step * dreamer_config.action_repeat)
        elapsed_seconds = time.time() - start_time
        sps = int(global_step / max(elapsed_seconds, 1e-6))
        log.info(f"iteration={iteration} global_step={global_step} SPS={sps}")
        if wandb_run is not None:
            wandb_run.log({"charts/SPS": sps, "charts/elapsed_time": elapsed_seconds}, step=global_step)

        # save agent weights
        if cfg.save_agent:
            save_every = max(1, int((dreamer_config.steps or 1) // max(int(cfg.save_num), 1)))
            if (iteration % save_every == 0) or (agent._step >= dreamer_config.steps):
                path = f"agents/agent_step_{global_step}.pt"
                torch.save(agent.state_dict(), path)
                log.info(f"saved={path}")
                if wandb_run is not None:
                    wandb_run.save(path)

        if bool(getattr(cfg, "save_checkpoint", True)):
            ckpt_every = int(getattr(cfg, "checkpoint_every", 1))
            if ckpt_every <= 0:
                ckpt_every = 1
            if (iteration % ckpt_every == 0) or (agent._step >= dreamer_config.steps):
                ckpt_path = f"checkpoints/ckpt_iter_{iteration:06d}_step_{global_step:012d}.pt"
                save_checkpoint(
                    ckpt_path,
                    agent=agent,
                    global_step=global_step,
                    iteration=iteration,
                    cfg=cfg,
                    run_name=run_name,
                    wandb_run_id=wandb_run.id if wandb_run is not None else None,
                )
                log.info(f"checkpoint_saved={ckpt_path}")
                if wandb_run is not None:
                    wandb_run.save(ckpt_path)

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
