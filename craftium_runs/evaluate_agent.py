# -*- coding: utf-8 -*-
"""
学習済みエージェントの評価スクリプト
- 複数エピソードを実行して性能を評価
- 動画を録画して視覚的に確認
- 統計情報（平均報酬、エピソード長など）を出力
"""

import os
import json
import time
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

import craftium

# --------------------------
# Agent definition (must match training)
# --------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_shape, action_n, z_dim: int = 256, z_project_dim: int = 128):
        super().__init__()
        c = observation_shape[0]
        self.visual = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape)
            v = self.visual(dummy)
            flat = v.shape[1]

        self.z_project = nn.Sequential(
            layer_init(nn.Linear(z_dim, z_project_dim)),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(flat + z_project_dim, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)
        self.z_dim = z_dim

    def forward_body(self, x, z):
        x = x / 255.0
        v = self.visual(x)
        zp = self.z_project(z)
        h = torch.cat([v, zp], dim=-1)
        h = self.mlp(h)
        return h

    def get_action(self, x, z, deterministic=False):
        h = self.forward_body(x, z)
        logits = self.actor(h)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
        return action

    def get_value(self, x, z):
        h = self.forward_body(x, z)
        return self.critic(h)


class GrayscaleToRGBRenderWrapper(gym.Wrapper):
    """Wrapper that converts grayscale render output to RGB for video recording."""
    def render(self):
        frame = self.env.render()
        if frame is not None and frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        return frame


def make_eval_env(env_id, mt_wd, mt_port, frameskip, seed, record_video=False, video_dir=None, fps_max=200):
    """評価用環境を作成"""
    craftium_kwargs = dict(
        run_dir_prefix=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        rgb_observations=False,
        seed=seed,
        fps_max=fps_max,
        pmul=1.0,
    )
    
    if record_video:
        env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)
        env = GrayscaleToRGBRenderWrapper(env)
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    else:
        env = gym.make(env_id, **craftium_kwargs)
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def evaluate(
    agent_path: str,
    env_id: str = "Craftium/ChopTree-v0",
    num_episodes: int = 10,
    deterministic: bool = True,
    record_video: bool = True,
    video_dir: str = None,
    mt_wd: str = "./eval_runs",
    mt_port: int = 49200,
    frameskip: int = 4,
    seed: int = 42,
    z_dim: int = 256,
    z_project_dim: int = 128,
    device: str = "cuda",
):
    """学習済みエージェントを評価"""
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Loading agent from: {agent_path}")
    print(f"[Eval] Environment: {env_id}")
    print(f"[Eval] Episodes: {num_episodes}")
    print(f"[Eval] Deterministic: {deterministic}")
    
    # 環境作成
    env = make_eval_env(
        env_id=env_id,
        mt_wd=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        seed=seed,
        record_video=record_video,
        video_dir=video_dir,
    )
    
    observation_shape = env.observation_space.shape
    action_n = env.action_space.n
    print(f"[Eval] Observation shape: {observation_shape}")
    print(f"[Eval] Action space: {action_n}")
    
    # エージェント作成・ロード
    agent = Agent(
        observation_shape=observation_shape,
        action_n=action_n,
        z_dim=z_dim,
        z_project_dim=z_project_dim,
    ).to(device)
    
    state_dict = torch.load(agent_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print("[Eval] Agent loaded successfully")
    
    # 評価ループ
    episode_rewards = []
    episode_lengths = []
    episode_chops = []  # 木を切った回数（報酬=1のカウント）
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        obs = torch.tensor(obs, device=device).unsqueeze(0)
        z = torch.zeros((1, z_dim), device=device)  # ARM-FMなしで評価
        
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_chops = 0
        
        while not done:
            with torch.no_grad():
                action = agent.get_action(obs, z, deterministic=deterministic)
            
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            
            ep_reward += reward
            ep_length += 1
            if reward >= 1.0:
                ep_chops += 1
            
            obs = torch.tensor(next_obs, device=device).unsqueeze(0)
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length * frameskip)  # 実際のステップ数
        episode_chops.append(ep_chops)
        
        print(f"  Episode {ep + 1}/{num_episodes}: reward={ep_reward:.1f}, length={ep_length * frameskip}, chops={ep_chops}")
    
    env.close()
    
    # 統計情報
    results = {
        "agent_path": agent_path,
        "env_id": env_id,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "frameskip": frameskip,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_chops": episode_chops,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_chops": float(np.mean(episode_chops)),
        "std_chops": float(np.std(episode_chops)),
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min/Max:      {results['min_reward']:.1f} / {results['max_reward']:.1f}")
    print(f"  Mean Length:  {results['mean_length']:.0f} ± {results['std_length']:.0f}")
    print(f"  Mean Chops:   {results['mean_chops']:.1f} ± {results['std_chops']:.1f}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to agent checkpoint (.pt)")
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory for video output")
    parser.add_argument("--mt-wd", type=str, default="./eval_runs", help="Minetest working directory")
    parser.add_argument("--mt-port", type=int, default=49200, help="Minetest port")
    parser.add_argument("--frameskip", type=int, default=4, help="Frame skip")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--z-dim", type=int, default=256, help="RM embedding dimension")
    parser.add_argument("--z-project-dim", type=int, default=128, help="RM projection dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()
    
    # デフォルトのビデオディレクトリ
    if args.video_dir is None:
        timestamp = int(time.time())
        args.video_dir = f"../output/eval_videos/eval_{timestamp}"
    
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.mt_wd, exist_ok=True)
    
    results = evaluate(
        agent_path=args.agent_path,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        record_video=not args.no_video,
        video_dir=args.video_dir,
        mt_wd=args.mt_wd,
        mt_port=args.mt_port,
        frameskip=args.frameskip,
        seed=args.seed,
        z_dim=args.z_dim,
        z_project_dim=args.z_project_dim,
        device=args.device,
    )
    
    # 結果をJSONに保存
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Eval] Results saved to: {args.output_json}")
    
    if not args.no_video:
        print(f"[Eval] Videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
