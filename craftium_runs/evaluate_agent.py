# 入力：obs と z=0ベクトルからz=RM状態embedding に修正。


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
import re

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

import craftium


# ==========================
# ARM-FM: load RM artifacts
# ==========================

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
    if text.endswith("```"):
        text = "\n".join(text.splitlines()[:-1])
    return text.strip()


class LARM:
    def __init__(self, states, initial_state, transitions, rewards, state_instructions, event_names, event_funcs, embeddings):
        self.states = states
        self.initial_state = initial_state
        self.transitions = transitions
        self.rewards = rewards
        self.state_instructions = state_instructions
        self.event_names = event_names
        self.event_funcs = event_funcs
        self.embeddings = embeddings
        self.current_state = initial_state

    def reset(self):
        self.current_state = self.initial_state

    def step(self, env_wrapper) -> float:
        sigma = None
        for ev in self.event_names:
            try:
                if self.event_funcs[ev](env_wrapper):
                    sigma = ev
                    break
            except Exception:
                continue

        next_state = self.transitions.get((self.current_state, sigma), None)
        if next_state is None:
            next_state = self.transitions.get(
                (self.current_state, "else"), self.current_state)
            r = self.rewards.get((self.current_state, "else", next_state), 0.0)
        else:
            r = self.rewards.get((self.current_state, sigma, next_state), 0.0)

        self.current_state = next_state
        return r

    def current_embedding(self):
        if self.current_state in self.embeddings:
            return self.embeddings[self.current_state]
        # fallback
        dim = len(next(iter(self.embeddings.values()))
                  ) if self.embeddings else 1
        return [0.0] * dim


class ARMFMEnvWrapper(gym.Wrapper):
    def __init__(self, env, larm, rm_reward_scale: float = 0.0, recent_window: int = 50):
        super().__init__(env)
        self.larm = larm
        self.rm_reward_scale = rm_reward_scale
        self.recent_window = recent_window

        self._armfm_last_reward = 0.0
        self._armfm_last_action = -1
        self._armfm_step = 0
        self._recent_rewards = []

        self._armfm_recent_chops = 0
        self._armfm_recent_steps = 0
        self._armfm_streak = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.larm.reset()
        self._armfm_last_reward = 0.0
        self._armfm_last_action = -1
        self._armfm_step = 0
        self._recent_rewards.clear()
        self._armfm_recent_chops = 0
        self._armfm_recent_steps = 0
        self._armfm_streak = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._armfm_step += 1
        self._armfm_last_action = int(action) if np.isscalar(action) else -1
        self._armfm_last_reward = float(reward)

        self._recent_rewards.append(self._armfm_last_reward)
        if len(self._recent_rewards) > self.recent_window:
            self._recent_rewards.pop(0)

        self._armfm_recent_chops = sum(
            1 for r in self._recent_rewards if r >= 1.0)
        self._armfm_recent_steps = min(self._armfm_step, self.recent_window)
        self._armfm_streak = (self._armfm_streak +
                              1) if self._armfm_last_reward >= 1.0 else 0

        rm_r = self.larm.step(self)
        total_r = reward + self.rm_reward_scale * rm_r

        info = info or {}
        info["env_reward"] = float(reward)
        info["armfm_rm_state"] = self.larm.current_state
        info["armfm_rm_reward"] = rm_r
        info["armfm_total_reward"] = total_r

        return obs, total_r, terminated, truncated, info

    def current_state_embedding(self) -> np.ndarray:
        return np.asarray(self.larm.current_embedding(), dtype=np.float32)


def make_eval_env(
    env_id, mt_wd, mt_port, frameskip, seed,
    record_video=False, video_dir=None, fps_max=200,
    enable_armfm=False, larm=None, rm_reward_scale=0.0,
):
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
        env = gym.wrappers.RecordVideo(
            env, video_dir, episode_trigger=lambda x: True)
    else:
        env = gym.make(env_id, **craftium_kwargs)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FrameStack(env, 4)

    # ARM-FM ON評価なら最後にwrapperを付ける
    if enable_armfm:
        if larm is None:
            raise ValueError("enable_armfm=True but larm is None")
        env = ARMFMEnvWrapper(env, larm=larm, rm_reward_scale=rm_reward_scale)

    return env


class RMParser:
    RM_PATTERN = re.compile(
        r"STATES:\s*(?P<states>.+?)\s*INITIAL_STATE:\s*(?P<init>.+?)\s*TRANSITION_FUNCTION:\s*(?P<trans>.+?)\s*REWARD_FUNCTION:\s*(?P<rewards>.+?)\s*STATE_INSTRUCTIONS:\s*(?P<instr>.+)",
        re.S | re.I
    )

    @staticmethod
    def parse_reward_machine(text: str):
        m = RMParser.RM_PATTERN.search(text)
        if not m:
            raise ValueError(
                "Failed to parse RM text. Ensure strict format and sections exist.")

        states_raw = m.group("states").strip()
        init_raw = m.group("init").strip()
        trans_raw = m.group("trans").strip()
        rewards_raw = m.group("rewards").strip()
        instr_raw = m.group("instr").strip()

        states = [s.strip() for s in states_raw.split(",") if s.strip()]
        initial_state = init_raw

        transitions = {}
        event_names = set()
        for line in trans_raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            m2 = re.match(r"\(([^,]+),\s*([^)]+)\)\s*->\s*(\S+)", line)
            if not m2:
                continue
            u_from = m2.group(1).strip()
            ev = m2.group(2).strip()
            u_to = m2.group(3).strip()
            transitions[(u_from, ev)] = u_to
            if ev != "else":
                event_names.add(ev)

        rewards = {}
        for line in rewards_raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            m3 = re.match(
                r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)\s*->\s*([+-]?\d+(\.\d+)?)", line)
            if not m3:
                continue
            u_from = m3.group(1).strip()
            ev = m3.group(2).strip()
            u_to = m3.group(3).strip()
            val = float(m3.group(4))
            rewards[(u_from, ev, u_to)] = val
            if ev != "else":
                event_names.add(ev)

        state_instructions = {}
        for line in instr_raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            state_instructions[k.strip()] = v.strip()

        return states, initial_state, transitions, rewards, state_instructions, sorted(event_names)


def hash_embed(text: str, dim: int = 256):
    # 学習側が「embed_model=None（hash fallback）」だった場合はこれで一致する
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v


def load_larm_from_artifacts(artifacts_dir: str, z_dim: int = 256):
    rm_path = os.path.join(artifacts_dir, "rm_spec.txt")
    code_path = os.path.join(artifacts_dir, "labeling_code.py")
    instr_path = os.path.join(artifacts_dir, "state_instructions.json")

    if not os.path.exists(rm_path):
        raise FileNotFoundError(f"rm_spec.txt not found: {rm_path}")
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"labeling_code.py not found: {code_path}")
    if not os.path.exists(instr_path):
        raise FileNotFoundError(
            f"state_instructions.json not found: {instr_path}")

    rm_text = open(rm_path, "r", encoding="utf-8").read()
    code_text = open(code_path, "r", encoding="utf-8").read()
    state_instructions = json.load(open(instr_path, "r", encoding="utf-8"))

    rm_text = _strip_fences(rm_text)
    code_text = _strip_fences(code_text)

    states, initial_state, transitions, rewards, _, event_names = RMParser.parse_reward_machine(
        rm_text)

    # compile labeling funcs
    glb = {}
    loc = {}
    exec(code_text, glb, loc)
    event_funcs = {}
    for ev in event_names:
        if ev not in loc or not callable(loc[ev]):
            raise ValueError(f"Missing labeling function for event: {ev}")
        event_funcs[ev] = loc[ev]

    # embeddings（学習側がhashなら一致。学習側がAPI embedなら本当はそれを保存して読むのが理想）
    embeddings = {k: hash_embed(v, dim=z_dim)
                  for k, v in state_instructions.items()}

    larm = LARM(
        states=states,
        initial_state=initial_state,
        transitions=transitions,
        rewards=rewards,
        state_instructions=state_instructions,
        event_names=event_names,
        event_funcs=event_funcs,
        embeddings=embeddings,
    )
    return larm


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
    enable_armfm: bool = False,
    armfm_artifacts_dir: str = None,
    rm_reward_scale: float = 0.0,
):
    """学習済みエージェントを評価"""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Loading agent from: {agent_path}")
    print(f"[Eval] Environment: {env_id}")
    print(f"[Eval] Episodes: {num_episodes}")
    print(f"[Eval] Deterministic: {deterministic}")
    print(f"[Eval] Seed: {seed}")

    larm = None
    if enable_armfm:
        if armfm_artifacts_dir is None:
            raise ValueError("enable_armfm=True requires armfm_artifacts_dir")
        larm = load_larm_from_artifacts(armfm_artifacts_dir, z_dim=z_dim)

    # 環境作成
    env = make_eval_env(
        env_id=env_id,
        mt_wd=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        seed=seed,
        record_video=record_video,
        video_dir=video_dir,
        enable_armfm=enable_armfm,
        larm=larm,
        rm_reward_scale=rm_reward_scale,
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

        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_chops = 0

        while not done:
            # RM状態embeddingを読む
            if enable_armfm:
                z_np = env.current_state_embedding()   # (z_dim,)
                z = torch.tensor(z_np, device=device).unsqueeze(0)
            else:
                z = torch.zeros((1, z_dim), device=device)

            with torch.no_grad():
                action = agent.get_action(obs, z, deterministic=deterministic)

            next_obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()[0])
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            env_reward = info.get("env_reward", reward)
            if env_reward >= 1.0:
                ep_chops += 1

            obs = torch.tensor(next_obs, device=device).unsqueeze(0)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length * frameskip)  # 実際のステップ数
        episode_chops.append(ep_chops)

        print(
            f"  Episode {ep + 1}/{num_episodes}: reward={ep_reward:.1f}, length={ep_length * frameskip}, chops={ep_chops}")

    env.close()

    # 統計情報
    results = {
        "agent_path": agent_path,
        "env_id": env_id,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "seed": seed,
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
    print(
        f"  Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(
        f"  Min/Max:      {results['min_reward']:.1f} / {results['max_reward']:.1f}")
    print(
        f"  Mean Length:  {results['mean_length']:.0f} ± {results['std_length']:.0f}")
    print(
        f"  Mean Chops:   {results['mean_chops']:.1f} ± {results['std_chops']:.1f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--agent-path", type=str, required=True,
                        help="Path to agent checkpoint (.pt)")
    parser.add_argument("--env-id", type=str,
                        default="Craftium/ChopTree-v0", help="Environment ID")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video recording")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory for video output")
    parser.add_argument("--mt-wd", type=str, default="./eval_runs",
                        help="Minetest working directory")
    parser.add_argument("--mt-port", type=int,
                        default=49200, help="Minetest port")
    parser.add_argument("--frameskip", type=int, default=4, help="Frame skip")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--z-dim", type=int, default=256,
                        help="RM embedding dimension")
    parser.add_argument("--z-project-dim", type=int,
                        default=128, help="RM projection dimension")
    parser.add_argument("--device", type=str,
                        default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-json", type=str,
                        default=None, help="Save results to JSON file")
    parser.add_argument("--enable-armfm", action="store_true",
                        help="Enable ARM-FM evaluation (load RM artifacts + use z embedding)")
    parser.add_argument("--armfm-artifacts-dir", type=str, default=None,
                        help="Path to armfm_artifacts directory (contains rm_spec.txt, labeling_code.py, state_instructions.json)")
    parser.add_argument("--rm-reward-scale", type=float, default=0.0,
                        help="RM reward scale used in ARMFMEnvWrapper during eval")
    parser.add_argument(
        "--seed_num",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of fixed seeds to evaluate (1...3). Uses [42, 123, 456] in order.",
    )

    args = parser.parse_args()

    # デフォルトのビデオディレクトリ
    if args.video_dir is None:
        timestamp = int(time.time())
        args.video_dir = f"../output/eval_videos/eval_{timestamp}"

    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.mt_wd, exist_ok=True)

    FIXED_SEEDS = [42, 123, 456]
    eval_seeds = FIXED_SEEDS[: args.seed_num]

    all_results = []
    for i, s in enumerate(eval_seeds, start=1):
        print("\n" + "#" * 70)
        print(f"[Multi-Seed Eval] {i}/{len(eval_seeds)}  seed={s}")
        print("#" * 70)

        # seedごとに動画ディレクトリを分ける（衝突回避）
        per_seed_video_dir = args.video_dir
        if not args.no_video:
            per_seed_video_dir = os.path.join(args.video_dir, f"seed{s}")
            os.makedirs(per_seed_video_dir, exist_ok=True)

        res = evaluate(
            agent_path=args.agent_path,
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            deterministic=args.deterministic,
            record_video=not args.no_video,
            video_dir=per_seed_video_dir,
            mt_wd=args.mt_wd,
            mt_port=args.mt_port,
            frameskip=args.frameskip,
            seed=s,
            z_dim=args.z_dim,
            z_project_dim=args.z_project_dim,
            device=args.device,
            enable_armfm=args.enable_armfm,
            armfm_artifacts_dir=args.armfm_artifacts_dir,
            rm_reward_scale=args.rm_reward_scale,
        )
        all_results.append(res)

    # 集計（seed間の平均）
    mean_rewards = [r["mean_reward"] for r in all_results]
    mean_lengths = [r["mean_length"] for r in all_results]
    mean_chops = [r["mean_chops"] for r in all_results]

    agg = {
        "seed_num": args.seed_num,
        "seeds": eval_seeds,
        "mean_reward_over_seeds": float(np.mean(mean_rewards)),
        "std_reward_over_seeds": float(np.std(mean_rewards)),
        "mean_length_over_seeds": float(np.mean(mean_lengths)),
        "std_length_over_seeds": float(np.std(mean_lengths)),
        "mean_chops_over_seeds": float(np.mean(mean_chops)),
        "std_chops_over_seeds": float(np.std(mean_chops)),
    }

    print("\n" + "=" * 60)
    print("Multi-Seed Summary")
    print("=" * 60)
    for r in all_results:
        print(
            f"  seed={r['seed']}: mean_reward={r['mean_reward']:.2f}, mean_length={r['mean_length']:.0f}, mean_chops={r['mean_chops']:.2f}"
        )
    print("-" * 60)
    print(
        f"  over_seeds: mean_reward={agg['mean_reward_over_seeds']:.2f} ± {agg['std_reward_over_seeds']:.2f}"
    )
    print("=" * 60)


    # 結果をJSONに保存
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Eval] Results saved to: {args.output_json}")

    if not args.no_video:
        print(f"[Eval] Videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
