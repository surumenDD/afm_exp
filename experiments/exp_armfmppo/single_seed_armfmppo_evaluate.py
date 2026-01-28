
## experiments/exp_armfmppo/single_seed_armfmppo_evaluate.py
# -*- coding: utf-8 -*-
"""
ARMFM+PPO 学習済みエージェント評価（単発 seed / 複数固定 seed）
- training と同じ obs 前処理: RecordEpisodeStatistics + FrameStack(4)
- ARM-FM 評価時: armfm_artifacts を読み、RM 状態 embedding を z として与える
- agent の checkpoint 形式は state_dict を想定（必要なら dict 判別も追加可能）
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

# train.py と同じ補助クラスを最小再掲（依存を増やさない）
import re
import hashlib


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = "\n".join(t.splitlines()[1:])
    if t.endswith("```"):
        t = "\n".join(t.splitlines()[:-1])
    return t.strip()


class RMParser:
    RM_PATTERN = re.compile(
        r"STATES:\s*(?P<states>.+?)\s*INITIAL_STATE:\s*(?P<init>.+?)\s*TRANSITION_FUNCTION:\s*(?P<trans>.+?)\s*REWARD_FUNCTION:\s*(?P<rewards>.+?)\s*STATE_INSTRUCTIONS:\s*(?P<instr>.+)",
        re.S | re.I,
    )

    @staticmethod
    def parse_reward_machine(text: str):
        m = RMParser.RM_PATTERN.search(text)
        if not m:
            raise ValueError("Failed to parse RM text. Ensure strict format and sections exist.")
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
            m3 = re.match(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)\s*->\s*([+-]?\d+(\.\d+)?)", line)
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


def hash_embed(text: str, dim: int) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v


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

        nxt = self.transitions.get((self.current_state, sigma), None)
        if nxt is None:
            nxt = self.transitions.get((self.current_state, "else"), self.current_state)
            r = self.rewards.get((self.current_state, "else", nxt), 0.0)
        else:
            r = self.rewards.get((self.current_state, sigma, nxt), 0.0)
        self.current_state = nxt
        return float(r)

    def current_embedding(self) -> np.ndarray:
        z = self.embeddings.get(self.current_state, None)
        if z is None:
            any_z = next(iter(self.embeddings.values()), None)
            dim = int(any_z.shape[0]) if any_z is not None else 1
            return np.zeros((dim,), dtype=np.float32)
        return z.astype(np.float32)


class ARMFMEnvWrapper(gym.Wrapper):
    def __init__(self, env, larm, rm_reward_scale: float = 0.0, recent_window: int = 50):
        super().__init__(env)
        self.larm = larm
        self.rm_reward_scale = float(rm_reward_scale)
        self.recent_window = int(recent_window)

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

        self._armfm_recent_chops = sum(1 for r in self._recent_rewards if r >= 1.0)
        self._armfm_recent_steps = min(self._armfm_step, self.recent_window)
        self._armfm_streak = (self._armfm_streak + 1) if self._armfm_last_reward >= 1.0 else 0

        rm_r = self.larm.step(self)
        total_r = float(reward) + self.rm_reward_scale * float(rm_r)

        info = info or {}
        info["env_reward"] = float(reward)
        info["armfm_rm_state"] = str(self.larm.current_state)
        info["armfm_rm_reward"] = float(rm_r)
        info["armfm_total_reward"] = float(total_r)

        return obs, total_r, terminated, truncated, info

    def current_state_embedding(self) -> np.ndarray:
        return self.larm.current_embedding()


class GrayscaleToRGBRenderWrapper(gym.Wrapper):
    def render(self):
        frame = self.env.render()
        if frame is not None and frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        return frame


def load_larm_from_artifacts(artifacts_dir: str, z_dim: int):
    rm_path = os.path.join(artifacts_dir, "rm_spec.txt")
    code_path = os.path.join(artifacts_dir, "labeling_code.py")
    instr_path = os.path.join(artifacts_dir, "state_instructions.json")

    rm_text = _strip_fences(open(rm_path, "r", encoding="utf-8").read())
    code_text = _strip_fences(open(code_path, "r", encoding="utf-8").read())
    state_instructions = json.load(open(instr_path, "r", encoding="utf-8"))

    states, initial_state, transitions, rewards, _, event_names = RMParser.parse_reward_machine(rm_text)

    glb = {}
    loc = {}
    exec(code_text, glb, loc)

    event_funcs = {}
    for ev in event_names:
        if ev not in loc or not callable(loc[ev]):
            raise ValueError(f"Missing labeling function for event: {ev}")
        event_funcs[ev] = loc[ev]

    # embeddings.npy があればそれを使い、なければ hash で再計算
    emb_path = os.path.join(artifacts_dir, "embeddings.npy")
    keys_path = os.path.join(artifacts_dir, "embeddings_keys.json")
    embeddings = {}

    if os.path.exists(emb_path) and os.path.exists(keys_path):
        keys = json.load(open(keys_path, "r", encoding="utf-8"))
        mat = np.load(emb_path).astype(np.float32)
        for i, k in enumerate(keys):
            embeddings[k] = mat[i][: int(z_dim)]
    else:
        for k, v in state_instructions.items():
            embeddings[k] = hash_embed(v, dim=int(z_dim))

    return LARM(states, initial_state, transitions, rewards, state_instructions, event_names, event_funcs, embeddings)


def make_eval_env(
    env_id: str,
    mt_wd: str,
    mt_port: int,
    frameskip: int,
    seed: int,
    record_video: bool,
    video_dir: str,
    fps_max: int,
    pmul: float,
    enable_armfm: bool,
    larm: LARM,
    rm_reward_scale: float,
):
    craftium_kwargs = dict(
        run_dir_prefix=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        rgb_observations=False,
        seed=seed,
        fps_max=fps_max,
        pmul=pmul,
    )

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)
        env = GrayscaleToRGBRenderWrapper(env)
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda _: True)
    else:
        env = gym.make(env_id, **craftium_kwargs)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FrameStack(env, 4)

    if enable_armfm:
        env = ARMFMEnvWrapper(env, larm=larm, rm_reward_scale=rm_reward_scale)

    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_shape, action_n, z_dim: int, z_project_dim: int):
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
            flat = int(v.shape[1])

        self.z_project = nn.Sequential(
            layer_init(nn.Linear(int(z_dim), int(z_project_dim))),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(flat + int(z_project_dim), 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)
        self.z_dim = int(z_dim)

    def forward_body(self, x, z):
        x = x / 255.0
        v = self.visual(x)
        zp = self.z_project(z)
        h = torch.cat([v, zp], dim=-1)
        h = self.mlp(h)
        return h

    def get_action(self, x, z, deterministic: bool):
        h = self.forward_body(x, z)
        logits = self.actor(h)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()


def evaluate_one(
    agent_path: str,
    env_id: str,
    num_episodes: int,
    deterministic: bool,
    record_video: bool,
    video_dir: str,
    mt_wd: str,
    mt_port: int,
    frameskip: int,
    seed: int,
    device_str: str,
    fps_max: int,
    pmul: float,
    enable_armfm: bool,
    armfm_artifacts_dir: str,
    rm_reward_scale: float,
    z_dim: int,
    z_project_dim: int,
):
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")

    larm = None
    if enable_armfm:
        larm = load_larm_from_artifacts(armfm_artifacts_dir, z_dim=int(z_dim))

    env = make_eval_env(
        env_id=env_id,
        mt_wd=mt_wd,
        mt_port=mt_port,
        frameskip=frameskip,
        seed=seed,
        record_video=record_video,
        video_dir=video_dir,
        fps_max=fps_max,
        pmul=pmul,
        enable_armfm=enable_armfm,
        larm=larm,
        rm_reward_scale=rm_reward_scale,
    )

    obs_shape = env.observation_space.shape
    action_n = env.action_space.n

    agent = Agent(obs_shape, action_n, z_dim=int(z_dim), z_project_dim=int(z_project_dim)).to(device)
    state_dict = torch.load(agent_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    episode_rewards = []
    episode_lengths = []
    episode_chops = []

    for ep in range(int(num_episodes)):
        obs, _ = env.reset(seed=int(seed) + ep)
        obs_t = torch.tensor(obs, device=device).unsqueeze(0)

        done = False
        ep_r = 0.0
        ep_l = 0
        ep_chop = 0

        while not done:
            if enable_armfm:
                z_np = env.current_state_embedding()
                z = torch.tensor(z_np, device=device).unsqueeze(0)
            else:
                z = torch.zeros((1, int(z_dim)), device=device)

            with torch.no_grad():
                action = agent.get_action(obs_t, z, deterministic=deterministic)

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = bool(terminated or truncated)

            ep_r += float(reward)
            ep_l += 1

            env_reward = float(info.get("env_reward", reward))
            if env_reward >= 1.0:
                ep_chop += 1

            obs_t = torch.tensor(next_obs, device=device).unsqueeze(0)

        episode_rewards.append(ep_r)
        episode_lengths.append(ep_l * int(frameskip))
        episode_chops.append(ep_chop)

        print(f"  Episode {ep+1}/{num_episodes}: reward={ep_r:.1f}, length={ep_l*frameskip}, chops={ep_chop}")

    env.close()

    res = {
        "agent_path": agent_path,
        "env_id": env_id,
        "num_episodes": int(num_episodes),
        "deterministic": bool(deterministic),
        "seed": int(seed),
        "frameskip": int(frameskip),
        "enable_armfm": bool(enable_armfm),
        "armfm_artifacts_dir": armfm_artifacts_dir if enable_armfm else None,
        "rm_reward_scale": float(rm_reward_scale),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_chops": episode_chops,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_chops": float(np.mean(episode_chops)),
        "std_chops": float(np.std(episode_chops)),
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-path", type=str, required=True)
    p.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--video-dir", type=str, default=None)
    p.add_argument("--mt-wd", type=str, default="./eval_runs")
    p.add_argument("--mt-port", type=int, default=49200)
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fps-max", type=int, default=200)
    p.add_argument("--pmul", type=float, default=1.0)

    p.add_argument("--enable-armfm", action="store_true")
    p.add_argument("--armfm-artifacts-dir", type=str, default=None)
    p.add_argument("--rm-reward-scale", type=float, default=0.0)

    p.add_argument("--z-dim", type=int, default=256)
    p.add_argument("--z-project-dim", type=int, default=128)

    p.add_argument("--seed-num", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--output-json", type=str, default=None)

    args = p.parse_args()

    if args.video_dir is None:
        ts = int(time.time())
        args.video_dir = f"./eval_videos/eval_{ts}"

    os.makedirs(args.mt_wd, exist_ok=True)
    if not args.no_video:
        os.makedirs(args.video_dir, exist_ok=True)

    if args.enable_armfm and not args.armfm_artifacts_dir:
        raise ValueError("--enable-armfm requires --armfm-artifacts-dir")

    FIXED_SEEDS = [42, 123, 456]
    eval_seeds = FIXED_SEEDS[: int(args.seed_num)]

    all_res = []
    for s in eval_seeds:
        print("\n" + "#" * 70)
        print(f"[Eval] seed={s}")
        print("#" * 70)

        per_seed_video_dir = args.video_dir
        if not args.no_video:
            per_seed_video_dir = os.path.join(args.video_dir, f"seed{s}")
            os.makedirs(per_seed_video_dir, exist_ok=True)

        res = evaluate_one(
            agent_path=args.agent_path,
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            deterministic=args.deterministic,
            record_video=(not args.no_video),
            video_dir=per_seed_video_dir,
            mt_wd=args.mt_wd,
            mt_port=args.mt_port,
            frameskip=args.frameskip,
            seed=s,
            device_str=args.device,
            fps_max=args.fps_max,
            pmul=args.pmul,
            enable_armfm=args.enable_armfm,
            armfm_artifacts_dir=args.armfm_artifacts_dir,
            rm_reward_scale=args.rm_reward_scale,
            z_dim=args.z_dim,
            z_project_dim=args.z_project_dim,
        )
        all_res.append(res)

    mean_rewards = [r["mean_reward"] for r in all_res]
    mean_chops = [r["mean_chops"] for r in all_res]

    print("\n" + "=" * 60)
    print("Multi-Seed Summary")
    print("=" * 60)
    for r in all_res:
        print(f"  seed={r['seed']}: mean_reward={r['mean_reward']:.2f}, mean_chops={r['mean_chops']:.2f}")
    print("-" * 60)
    print(f"  over_seeds: mean_reward={float(np.mean(mean_rewards)):.2f} ± {float(np.std(mean_rewards)):.2f}")
    print(f"  over_seeds: mean_chops={float(np.mean(mean_chops)):.2f} ± {float(np.std(mean_chops)):.2f}")
    print("=" * 60)

    if args.output_json:
        out = {
            "runs": all_res,
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