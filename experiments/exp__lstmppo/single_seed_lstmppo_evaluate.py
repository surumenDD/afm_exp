# -*- coding: utf-8 -*-
"""
Craftium PPO(LSTM) 学習済みエージェントの評価スクリプト（単発seed）
- lstm_ppo_train.py と同じ前処理・同じネットワーク構造（CNN + LSTM）
- 複数エピソード実行して平均報酬・平均長などを表示
- 動画録画（RecordVideo）対応
- agent_state_dict形式チェックポイント / state_dict単体の両方を自動判別してロード
"""

import os
import json
import time
import argparse
import datetime

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import craftium


# --------------------------
# Env builder (match training)
# --------------------------
def make_eval_env(
    env_id: str,
    mt_wd: str,
    mt_port: int,
    frameskip: int,
    seed: int,
    record_video: bool = False,
    video_dir: str | None = None,
    name_prefix: str = "eval",
):
    craftium_kwargs = dict(
        run_dir_prefix=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        rgb_observations=True,  # trainingと同じ
        mt_listen_timeout=300_000,
        seed=int(seed),
    )

    if record_video:
        if video_dir is None:
            raise ValueError("record_video=True requires video_dir")
        os.makedirs(video_dir, exist_ok=True)

        env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)

        safe_prefix = str(name_prefix).replace("/", "__")

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix=safe_prefix,
            episode_trigger=lambda episode_id: True,  # ★全エピソード保存
        )
    else:
        env = gym.make(env_id, **craftium_kwargs)


    # trainingと完全一致の前処理
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, 84)
    env = gym.wrappers.FrameStack(env, 4)

    return env


# --------------------------
# Agent (must match training)
# --------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, action_n: int):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(128, action_n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size)).float()

        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def act(self, x, lstm_state, done, deterministic: bool):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if deterministic:
            action = probs.probs.argmax(dim=-1)
        else:
            action = probs.sample()
        return action, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)


# --------------------------
# Checkpoint loader (auto-detect)
# --------------------------
def load_agent_state_dict(path: str, device: torch.device) -> dict:
    obj = torch.load(path, map_location=device)

    # trainingのcheckpoint形式: {"agent_state_dict": ..., ...}
    if isinstance(obj, dict) and "agent_state_dict" in obj:
        return obj["agent_state_dict"]

    # trainingのsave_agent形式: state_dictそのもの
    if isinstance(obj, dict):
        return obj

    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def evaluate(
    agent_path: str,
    env_id: str,
    num_episodes: int,
    deterministic: bool,
    record_video: bool,
    video_dir: str | None,
    mt_wd: str,
    mt_port: int,
    frameskip: int,
    seed: int,
    device_str: str,
):
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")
    print(f"[Eval] device={device}")
    print(f"[Eval] agent_path={agent_path}")
    print(f"[Eval] env_id={env_id}")
    print(f"[Eval] episodes={num_episodes}")
    print(f"[Eval] deterministic={deterministic}")
    print(f"[Eval] seed={seed}")
    print(f"[Eval] frameskip={frameskip}")
    if record_video:
        print(f"[Eval] video_dir={video_dir}")

    # env
    name_prefix = f"eval_seed{seed}"
    env = make_eval_env(
        env_id=env_id,
        mt_wd=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        seed=seed,
        record_video=record_video,
        video_dir=video_dir,
        name_prefix=name_prefix,
    )

    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    action_n = env.action_space.n
    print(f"[Eval] action_n={action_n}")
    print(f"[Eval] obs_shape={env.observation_space.shape}")

    # agent
    agent = Agent(action_n=action_n).to(device)
    state_dict = load_agent_state_dict(agent_path, device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print("[Eval] agent loaded")

    # stats
    episode_returns = []
    episode_lengths = []
    episode_chops = []

    # LSTM init (num_layers=1)
    def init_lstm_state():
        h = torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size, device=device)
        c = torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size, device=device)
        return (h, c)

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        obs = torch.tensor(obs, device=device).unsqueeze(0)  # (1,4,84,84)

        lstm_state = init_lstm_state()
        done_t = torch.zeros(1, device=device, dtype=torch.float32)

        done = False
        ep_return = 0.0
        ep_len = 0
        ep_chop = 0

        while not done:
            with torch.no_grad():
                action, lstm_state = agent.act(obs, lstm_state, done_t, deterministic=deterministic)

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1
            if float(reward) >= 1.0:
                ep_chop += 1

            obs = torch.tensor(next_obs, device=device).unsqueeze(0)
            done_t = torch.tensor([1.0 if done else 0.0], device=device, dtype=torch.float32)

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len * frameskip)  # trainingと同じ「実ステップ換算」
        episode_chops.append(ep_chop)

        print(f"  Episode {ep+1}/{num_episodes}: return={ep_return:.1f}, length={ep_len*frameskip}, chops={ep_chop}")

    env.close()

    results = {
        "agent_path": agent_path,
        "env_id": env_id,
        "num_episodes": int(num_episodes),
        "deterministic": bool(deterministic),
        "seed": int(seed),
        "frameskip": int(frameskip),
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
    }

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Mean Return:  {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Min/Max:      {results['min_return']:.1f} / {results['max_return']:.1f}")
    print(f"  Mean Length:  {results['mean_length']:.0f} ± {results['std_length']:.0f}")
    print(f"  Mean Chops:   {results['mean_chops']:.2f} ± {results['std_chops']:.2f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Craftium PPO(LSTM) agent (single seed)")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to agent .pt (state_dict or checkpoint)")
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy (argmax) action")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--video-dir", type=str, default=None, help="Video output directory")
    parser.add_argument("--mt-wd", type=str, default="./eval_runs", help="Minetest run directory prefix")
    parser.add_argument("--mt-port", type=int, default=49200, help="Minetest port")
    parser.add_argument("--frameskip", type=int, default=4, help="Frame skip (must match training)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
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
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        record_video=(not args.no_video),
        video_dir=(args.video_dir if not args.no_video else None),
        mt_wd=args.mt_wd,
        mt_port=args.mt_port,
        frameskip=args.frameskip,
        seed=args.seed,
        device_str=args.device,
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
