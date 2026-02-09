# -*- coding: utf-8 -*-
# ARM-FM + PPO for Craftium ChopTree (Hydra-based)
#
# 目的:
# - `experiments/exp__lstmppo` と同様に Hydra の `run.dir` 配下へ成果物を集約する。
# - LLM (FM) を毎回呼び出さないよう、`fm_output_dir` を導入して既存成果物を再利用可能にする。
#
# Note: Hydra の `chdir=true` 設定により、実行時のカレントディレクトリは `hydra.run.dir` に移動します。

import os
import re
import json
import time
import random
import hashlib
import textwrap
import logging
from typing import Optional, List, Dict, Any, Callable, Tuple

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import requests
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig, OmegaConf

import craftium


# --------------------------
# Logging
# --------------------------
def setup_logging():
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


# --------------------------
# FM Client (OpenAI-compatible)
# --------------------------
class FMClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        chat_model: str,
        embed_model: Optional[str],
        timeout_s: float = 60.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout_s = timeout_s

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _post_json(self, url: str, payload: dict, max_retries: int = 5) -> dict:
        for attempt in range(max_retries):
            resp = self.session.post(url, json=payload, timeout=self.timeout_s)
            if resp.status_code == 429:
                wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
                time.sleep(wait_time)
                continue
            if not resp.ok:
                # 失敗時に原因を残す
                raise RuntimeError(f"[FM API] {resp.status_code}: {resp.text}")
            return resp.json()
        resp = self.session.post(url, json=payload, timeout=self.timeout_s)
        if not resp.ok:
            raise RuntimeError(f"[FM API] {resp.status_code}: {resp.text}")
        return resp.json()

    def chat(self, messages: List[Dict[str, str]], temperature: float = 1.0, max_completion_tokens: int = 2000) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        data = self._post_json(url, payload)
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embed_model:
            return [self._hash_embed(t, dim=256) for t in texts]
        url = f"{self.api_base}/embeddings"
        payload = {"model": self.embed_model, "input": texts}
        data = self._post_json(url, payload)
        return [item["embedding"] for item in data["data"]]

    @staticmethod
    def _hash_embed(text: str, dim: int = 256) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v.tolist()


# --------------------------
# Prompts
# --------------------------
RM_GENERATOR_SYSTEM = """You are a Reward Machine Generator for Reinforcement Learning.
You must generate a compact, correct finite-state automaton (Reward Machine) that densifies sparse rewards, uses boolean predicate events, and follows the strict output format.
Output must be plain text only, wrapped in ```plaintext code fences. No comments or extra text.
"""

RM_GENERATOR_USER_TEMPLATE = """Environment:
- Domain: Craftium (Minecraft-like RL)
- Task: ChopTree
- Observations: Agent-centered view; reward +1.0 each time a tree is chopped; episode ends after 2000 steps by default.
- Notation: Events are boolean predicates derived from the environment state and/or last transition signals exposed via a wrapper.
- You may not use raw actions as events; only boolean predicates over state (and wrapper exposed attributes) are allowed.

Mission (Japanese):
"{mission_text}"

Your Role: Reward Machine Generator
Generate a concise, correct, compact Reward Machine as plain text and wrap it with ```plaintext.

Your machine must:
1) Densify learning signal towards the mission goal (chopping trees frequently).
2) Use boolean predicate events (functions taking the env wrapper as input) and avoid raw actions as events.
3) Maximize compactness; group irrelevant events as (state, else) -> state.
4) Follow STRICT FORMAT below; no extra commentary.
5) Use clear event names that are valid Python function names.

STRICT REWARD_MACHINE FORMAT:
```plaintext
STATES: u0, u1, ...
INITIAL_STATE: u0
TRANSITION_FUNCTION:
(u0, event_name) -> u1
(u0, else) -> u0
...
REWARD_FUNCTION:
(u0, event_name, u1) -> X
...
STATE_INSTRUCTIONS:
u0: <natural language instruction of subgoal>
u1: <...>
...
```

* Only list non-zero rewards in REWARD_FUNCTION; others are assumed to be 0.
* STATE_INSTRUCTIONS are short natural language strings (Japanese) that describe the subgoal at each state.

Hints for ChopTree:

* The wrapper exposes env._armfm_last_reward (last scalar env reward), env._armfm_last_action (last action id), and may store simple counters, e.g., env._armfm_recent_chops.
* Keep it compact (2~4 states).
"""

LABELING_GENERATOR_SYSTEM = """You are a Python labeling function generator for an RL Reward Machine.
You will implement one boolean function per event name in pure Python, using ONLY allowed attributes of the provided env wrapper.
Output ONLY valid Python code wrapped in ```python fences. No extra text or comments.
"""

ENV_SPEC_FOR_LABELING = """Allowed env wrapper attributes and meaning:

* env._armfm_last_reward: float, last environment reward received on the previous step (0.0 or 1.0 in ChopTree).
* env._armfm_last_action: int, last action index taken (if available).
* env._armfm_step: int, current step counter in episode.
* env._armfm_recent_chops: int, chops counted in a short window (sliding window maintained by wrapper).
* env._armfm_recent_steps: int, steps counted in a short window.
* env._armfm_streak: int, current chopping streak length (wrapper maintained).
No imports allowed. Functions receive a single argument env and must return True/False.
Do not define any "else" event function.
"""

LABELING_GENERATOR_USER_TEMPLATE = """Reward Machine to implement events for:
{reward_machine_text}

Guidelines:

* Define exactly one function per event listed in the RM.
* Function name must match the event name exactly.
* Signature must be: def {{event}}(env): -> bool
* Use ONLY the allowed env wrapper attributes:
{env_spec}

Output rules:

* Only output Python code.
* No comments or extra text.
* Wrap code in ```python fences.
"""

CRITIC_SYSTEM = """You are a strict critic and repair assistant for generated Reward Machines and labeling code.
Given errors, you must produce a corrected version following the same strict format.
No commentary, only the corrected artifact.
"""

CRITIC_USER_TEMPLATE_RM = """The previous Reward Machine had errors:
{error_text}

Please regenerate a corrected Reward Machine (same strict plaintext format).
"""

CRITIC_USER_TEMPLATE_CODE = """The previous labeling code had errors:
{error_text}

Please regenerate corrected Python labeling code with the same function names (strict code fence).
"""


# --------------------------
# RM runtime
# --------------------------
class LARM:
    def __init__(
        self,
        states: List[str],
        initial_state: str,
        transitions: Dict[Tuple[str, str], str],
        rewards: Dict[Tuple[str, str, str], float],
        state_instructions: Dict[str, str],
        event_names: List[str],
        event_funcs: Dict[str, Callable[[Any], bool]],
        embeddings: Dict[str, np.ndarray],
    ):
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
            # fallback: zero vector
            any_z = next(iter(self.embeddings.values()), None)
            dim = int(any_z.shape[0]) if any_z is not None else 1
            return np.zeros((dim,), dtype=np.float32)
        return z


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

        transitions: Dict[Tuple[str, str], str] = {}
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

        rewards: Dict[Tuple[str, str, str], float] = {}
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

        state_instructions: Dict[str, str] = {}
        for line in instr_raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            state_instructions[k.strip()] = v.strip()

        return states, initial_state, transitions, rewards, state_instructions, sorted(event_names)


def _extract_fenced(text: str, lang: str) -> Optional[str]:
    if lang:
        pat = re.compile(rf"```{lang}\s*\n?(.+?)\s*```", re.S | re.I)
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    pat = re.compile(r"```\w*\s*\n?(.+?)\s*```", re.S)
    m = pat.search(text)
    return m.group(1).strip() if m else None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = "\n".join(t.splitlines()[1:])
    if t.endswith("```"):
        t = "\n".join(t.splitlines()[:-1])
    return t.strip()


def hash_embed(text: str, dim: int) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v


class LARMBuilder:
    def __init__(self, fm: Optional[FMClient], artifacts_dir: str, z_dim: int, max_retries: int = 3):
        self.fm = fm
        self.artifacts_dir = artifacts_dir
        self.z_dim = int(z_dim)
        self.max_retries = int(max_retries)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def build(self, mission_text: str, gen_rounds: int) -> LARM:
        rm_text = self._generate_rm_text(mission_text, gen_rounds=int(gen_rounds))
        states, initial_state, transitions, rewards, state_instructions, event_names = RMParser.parse_reward_machine(rm_text)

        label_code = self._generate_labeling_code(rm_text, event_names)
        event_funcs = self._compile_labeling_functions(label_code, event_names)

        embeddings = self._embed_instructions(state_instructions)

        # 保存（再利用用）
        with open(os.path.join(self.artifacts_dir, "rm_spec.txt"), "w", encoding="utf-8") as f:
            f.write(rm_text)
        with open(os.path.join(self.artifacts_dir, "labeling_code.py"), "w", encoding="utf-8") as f:
            f.write(label_code)
        with open(os.path.join(self.artifacts_dir, "state_instructions.json"), "w", encoding="utf-8") as f:
            json.dump(state_instructions, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.artifacts_dir, "event_names.json"), "w", encoding="utf-8") as f:
            json.dump(event_names, f, ensure_ascii=False, indent=2)

        # embeddings を保存（fm_output_dir 再利用時に API を呼ばないため）
        emb_path = os.path.join(self.artifacts_dir, "embeddings.npy")
        # state順で保存（読み出し側もこの順で復元）
        keys = list(state_instructions.keys())
        mat = np.stack([embeddings[k] for k in keys], axis=0).astype(np.float32)
        np.save(emb_path, mat)
        with open(os.path.join(self.artifacts_dir, "embeddings_keys.json"), "w", encoding="utf-8") as f:
            json.dump(keys, f, ensure_ascii=False, indent=2)

        return LARM(
            states=states,
            initial_state=initial_state,
            transitions=transitions,
            rewards=rewards,
            state_instructions=state_instructions,
            event_names=event_names,
            event_funcs=event_funcs,
            embeddings=embeddings,
        )

    def _generate_rm_text(self, mission_text: str, gen_rounds: int) -> str:
        if self.fm is None:
            # 決定的フォールバック
            rm_text = textwrap.dedent(
                """\
                STATES: u0
                INITIAL_STATE: u0
                TRANSITION_FUNCTION:
                (u0, chopped_tree_recently) -> u0
                (u0, else) -> u0
                REWARD_FUNCTION:
                (u0, chopped_tree_recently, u0) -> 0.0
                STATE_INSTRUCTIONS:
                u0: 近くの木を見つけて斧で切り続ける
                """
            ).strip()
            return rm_text

        sys = {"role": "system", "content": RM_GENERATOR_SYSTEM}
        user = {"role": "user", "content": RM_GENERATOR_USER_TEMPLATE.format(mission_text=mission_text)}
        content = self.fm.chat([sys, user])
        rm_text = _extract_fenced(content, "plaintext") or _extract_fenced(content, "") or content
        rm_text = _strip_fences(rm_text)

        # parseできるまで critic 修正
        for _ in range(max(1, gen_rounds)):
            try:
                _ = RMParser.parse_reward_machine(rm_text)
                break
            except Exception as e:
                critic_sys = {"role": "system", "content": CRITIC_SYSTEM}
                critic_user = {"role": "user", "content": CRITIC_USER_TEMPLATE_RM.format(error_text=str(e))}
                fixed = self.fm.chat([critic_sys, critic_user])
                fixed_txt = _extract_fenced(fixed, "plaintext") or fixed
                rm_text = _strip_fences(fixed_txt)

        return rm_text

    def _generate_labeling_code(self, rm_text: str, event_names: List[str]) -> str:
        if self.fm is None:
            # フォールバック
            code = textwrap.dedent(
                """\
                def chopped_tree_recently(env):
                    return float(getattr(env, "_armfm_last_reward", 0.0)) >= 1.0
                """
            ).strip()
            return code

        sys = {"role": "system", "content": LABELING_GENERATOR_SYSTEM}
        u = {
            "role": "user",
            "content": LABELING_GENERATOR_USER_TEMPLATE.format(reward_machine_text=rm_text, env_spec=ENV_SPEC_FOR_LABELING),
        }
        content = self.fm.chat([sys, u])
        code = _extract_fenced(content, "python") or content
        code = _strip_fences(code)

        for _ in range(self.max_retries):
            try:
                _ = self._compile_labeling_functions(code, event_names)
                break
            except Exception as e:
                critic_sys = {"role": "system", "content": CRITIC_SYSTEM}
                critic_user = {"role": "user", "content": CRITIC_USER_TEMPLATE_CODE.format(error_text=str(e))}
                fixed = self.fm.chat([critic_sys, critic_user])
                code = _extract_fenced(fixed, "python") or fixed
                code = _strip_fences(code)

        return code

    def _compile_labeling_functions(self, code: str, event_names: List[str]) -> Dict[str, Callable]:
        glb: Dict[str, Any] = {}
        loc: Dict[str, Any] = {}
        exec(code, glb, loc)
        funcs: Dict[str, Callable] = {}
        for ev in event_names:
            if ev == "else":
                continue
            if ev not in loc or not callable(loc[ev]):
                raise ValueError(f"Missing labeling function for event: {ev}")
            funcs[ev] = loc[ev]
        return funcs

    def _embed_instructions(self, state_instructions: Dict[str, str]) -> Dict[str, np.ndarray]:
        keys = list(state_instructions.keys())
        texts = [state_instructions[k] for k in keys]

        if self.fm is None:
            vecs = [hash_embed(t, dim=self.z_dim) for t in texts]
        else:
            raw = self.fm.embed(texts)
            vecs = []
            for v in raw:
                vv = np.asarray(v, dtype=np.float32)
                if vv.shape[0] < self.z_dim:
                    pad = np.zeros((self.z_dim - vv.shape[0],), dtype=np.float32)
                    vv = np.concatenate([vv, pad], axis=0)
                vv = vv[: self.z_dim]
                vecs.append(vv)

        return {k: vecs[i] for i, k in enumerate(keys)}


def load_larm_from_artifacts(artifacts_dir: str, z_dim: int) -> LARM:
    rm_path = os.path.join(artifacts_dir, "rm_spec.txt")
    code_path = os.path.join(artifacts_dir, "labeling_code.py")
    instr_path = os.path.join(artifacts_dir, "state_instructions.json")

    if not os.path.exists(rm_path):
        raise FileNotFoundError(f"rm_spec.txt not found: {rm_path}")
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"labeling_code.py not found: {code_path}")
    if not os.path.exists(instr_path):
        raise FileNotFoundError(f"state_instructions.json not found: {instr_path}")

    rm_text = _strip_fences(open(rm_path, "r", encoding="utf-8").read())
    code_text = _strip_fences(open(code_path, "r", encoding="utf-8").read())
    state_instructions = json.load(open(instr_path, "r", encoding="utf-8"))

    states, initial_state, transitions, rewards, _, event_names = RMParser.parse_reward_machine(rm_text)

    glb: Dict[str, Any] = {}
    loc: Dict[str, Any] = {}
    exec(code_text, glb, loc)
    event_funcs: Dict[str, Callable] = {}
    for ev in event_names:
        if ev not in loc or not callable(loc[ev]):
            raise ValueError(f"Missing labeling function for event: {ev}")
        event_funcs[ev] = loc[ev]

    # APIを呼ばない。embeddings.npy があればそれを優先し、なければ hash_embed にフォールバック。
    emb_path = os.path.join(artifacts_dir, "embeddings.npy")
    keys_path = os.path.join(artifacts_dir, "embeddings_keys.json")
    embeddings: Dict[str, np.ndarray] = {}

    if os.path.exists(emb_path) and os.path.exists(keys_path):
        keys = json.load(open(keys_path, "r", encoding="utf-8"))
        mat = np.load(emb_path).astype(np.float32)
        for i, k in enumerate(keys):
            embeddings[k] = mat[i][: int(z_dim)]
    else:
        for k, v in state_instructions.items():
            embeddings[k] = hash_embed(v, dim=int(z_dim))

    return LARM(
        states=states,
        initial_state=initial_state,
        transitions=transitions,
        rewards=rewards,
        state_instructions=state_instructions,
        event_names=event_names,
        event_funcs=event_funcs,
        embeddings=embeddings,
    )


# --------------------------
# Env wrapper (ARM-FM)
# --------------------------
class ARMFMEnvWrapper(gym.Wrapper):
    def __init__(self, env, larm: LARM, rm_reward_scale: float = 0.0, recent_window: int = 50):
        super().__init__(env)
        self.larm = larm
        self.rm_reward_scale = float(rm_reward_scale)
        self.recent_window = int(recent_window)

        self._armfm_last_reward = 0.0
        self._armfm_last_action = -1
        self._armfm_step = 0
        self._recent_rewards: List[float] = []
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
        self._armfm_last_action = int(action) if np.isscalar(action) or isinstance(action, (int, np.integer)) else -1
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
        return self.larm.current_embedding().astype(np.float32)


class GrayscaleToRGBRenderWrapper(gym.Wrapper):
    def render(self):
        frame = self.env.render()
        if frame is not None and frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        return frame


# --------------------------
# PPO Agent (obs + z conditioning)
# --------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, z_dim: int, z_project_dim: int):
        super().__init__()
        c = envs.single_observation_space.shape[0]
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
            dummy = torch.zeros(1, *envs.single_observation_space.shape)
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
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward_body(self, x, z):
        x = x / 255.0
        v = self.visual(x)
        zp = self.z_project(z)
        h = torch.cat([v, zp], dim=-1)
        h = self.mlp(h)
        return h

    def get_value(self, x, z):
        h = self.forward_body(x, z)
        return self.critic(h)

    def get_action_and_value(self, x, z, action=None):
        h = self.forward_body(x, z)
        logits = self.actor(h)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(h)


# --------------------------
# Env factory
# --------------------------
def make_env(cfg: DictConfig, idx: int, run_name: str, larm: Optional[LARM]):
    def thunk():
        craftium_kwargs = dict(
            run_dir_prefix=str(cfg.env.mt_wd),
            mt_port=int(cfg.env.mt_port) + int(idx),
            frameskip=int(cfg.env.frameskip),
            rgb_observations=False,
            seed=int(cfg.seed) + int(idx),
            fps_max=int(cfg.env.fps_max),
            pmul=float(cfg.env.pmul),
            mt_listen_timeout=int(cfg.env.mt_listen_timeout),
        )

        if bool(cfg.env.capture_video) and int(idx) == 0:
            env = gym.make(str(cfg.env.env_id), render_mode="rgb_array", **craftium_kwargs)
            env = GrayscaleToRGBRenderWrapper(env)
            os.makedirs("videos", exist_ok=True)
            safe_prefix = str(run_name).replace("/", "__")
            env = gym.wrappers.RecordVideo(
                env,
                # "/" が含まれるとディレクトリ階層として解釈されてエラーになるため、置換する
                name_prefix=safe_prefix,
            )
        else:
            env = gym.make(str(cfg.env.env_id), **craftium_kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FrameStack(env, 4)

        if bool(cfg.armfm.enable) and larm is not None:
            env = ARMFMEnvWrapper(env, larm=larm, rm_reward_scale=float(cfg.armfm.rm_reward_scale))

        return env

    return thunk


def collect_current_z(envs, z_dim: int, enable_armfm: bool) -> np.ndarray:
    if not enable_armfm:
        return np.zeros((envs.num_envs, int(z_dim)), dtype=np.float32)
    zs_arr = envs.call("current_state_embedding")
    return np.stack(zs_arr, axis=0).astype(np.float32)


def save_text_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------
# Main
# --------------------------
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    load_dotenv()
    setup_logging()
    log = logging.getLogger(__name__)

    # seed
    now = int(time.time())
    if cfg.seed is None:
        cfg.seed = now

    # derive sizes
    cfg.ppo.num_envs = int(cfg.ppo.num_envs)
    cfg.ppo.num_steps = int(cfg.ppo.num_steps)
    cfg.env.frameskip = int(cfg.env.frameskip)
    cfg.ppo.num_minibatches = int(cfg.ppo.num_minibatches)

    if cfg.ppo.num_envs < 1:
        cfg.ppo.num_envs = 1
    if cfg.ppo.num_minibatches < 1:
        cfg.ppo.num_minibatches = 1
    if cfg.ppo.num_minibatches > cfg.ppo.num_envs:
        cfg.ppo.num_minibatches = cfg.ppo.num_envs
    while cfg.ppo.num_envs % cfg.ppo.num_minibatches != 0 and cfg.ppo.num_minibatches > 1:
        cfg.ppo.num_minibatches -= 1

    batch_size = int(cfg.ppo.num_envs * cfg.ppo.num_steps)
    minibatch_size = int(batch_size // cfg.ppo.num_minibatches)
    num_iterations = (int(cfg.ppo.total_timesteps) // int(cfg.env.frameskip)) // int(batch_size)

    run_name = f"{cfg.env.env_id}__{cfg.exp_name}__{cfg.seed}__{now}"
    log.info(f"run_name={run_name}")
    log.info(f"output_dir={os.getcwd()}")

    # dirs
    os.makedirs("agents", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("armfm_artifacts", exist_ok=True)

    # W&B (optional)
    wandb_run = None
    if bool(cfg.track):
        import wandb
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_run = wandb.init(
            project=str(cfg.wandb_project_name),
            entity=cfg.wandb_entity,
            name=run_name,
            config=wandb_cfg,
            save_code=True,
        )

    # RNG
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch.backends.cudnn.deterministic = bool(cfg.torch_deterministic)

    device = torch.device("cuda" if (torch.cuda.is_available() and bool(cfg.cuda)) else "cpu")
    log.info(f"device={device}")

    # TensorBoard / CSV
    writer = None
    if bool(cfg.log.tensorboard):
        os.makedirs("tb", exist_ok=True)
        writer = SummaryWriter("tb")
        writer.add_text("hydra_config", f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```", 0)

    csv_path = os.path.join("tb", "train_metrics.csv")
    if bool(cfg.log.csv):
        os.makedirs("tb", exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("global_step,iteration,lr,value_loss,policy_loss,entropy,approx_kl,clipfrac,explained_var,sps,rollout_mean_reward\n")

    # --------------------------
    # ARM-FM: build or reuse
    # --------------------------
    larm = None
    enable_armfm = bool(cfg.armfm.enable)

    fm_output_dir = str(getattr(cfg.armfm, "fm_output_dir", "")).strip()
    if enable_armfm and fm_output_dir != "":
        # 既存成果物参照（LLM を呼ばない）
        log.info(f"ARM-FM reuse: fm_output_dir={fm_output_dir}")
        larm = load_larm_from_artifacts(fm_output_dir, z_dim=int(cfg.armfm.z_dim))
        # 参照元をログへ残す
        save_text_json(os.path.join("armfm_artifacts", "reused_from.json"), {"fm_output_dir": fm_output_dir})
    elif enable_armfm and bool(cfg.armfm.auto_build_larm):
        # FM client（env からも拾う）
        api_base = cfg.armfm.fm_api_base or os.getenv("FM_API_BASE")
        api_key = cfg.armfm.fm_api_key or os.getenv("FM_API_KEY")
        chat_model = cfg.armfm.fm_chat_model or os.getenv("FM_CHAT_MODEL")
        embed_model = cfg.armfm.fm_embed_model or os.getenv("FM_EMBED_MODEL")

        fm_client = None
        fm_cfg = {"api_base": None, "chat_model": None, "embed_model": None}
        if api_base and api_key and chat_model:
            fm_client = FMClient(
                api_base=str(api_base),
                api_key=str(api_key),
                chat_model=str(chat_model),
                embed_model=(str(embed_model) if embed_model else None),
                timeout_s=float(cfg.armfm.fm_timeout_s),
            )
            fm_cfg = {"api_base": str(api_base), "chat_model": str(chat_model), "embed_model": (str(embed_model) if embed_model else None)}
        else:
            # フォールバック
            fm_client = None

        builder = LARMBuilder(
            fm=fm_client,
            artifacts_dir="armfm_artifacts",
            z_dim=int(cfg.armfm.z_dim),
            max_retries=int(cfg.armfm.larm_max_retries),
        )
        larm = builder.build(mission_text=str(cfg.armfm.mission_text), gen_rounds=int(cfg.armfm.larm_gen_rounds))

        # prompts / config 保存
        save_text_json(
            os.path.join("armfm_artifacts", "prompts_and_config.json"),
            {
                "rm_generator_system": RM_GENERATOR_SYSTEM,
                "rm_generator_user_template": RM_GENERATOR_USER_TEMPLATE,
                "labeling_generator_system": LABELING_GENERATOR_SYSTEM,
                "labeling_env_spec": ENV_SPEC_FOR_LABELING,
                "labeling_generator_user_template": LABELING_GENERATOR_USER_TEMPLATE,
                "critic_system": CRITIC_SYSTEM,
                "critic_user_template_rm": CRITIC_USER_TEMPLATE_RM,
                "critic_user_template_code": CRITIC_USER_TEMPLATE_CODE,
                "mission_text": str(cfg.armfm.mission_text),
                "fm_config": fm_cfg,
            },
        )
    else:
        larm = None

    # --------------------------
    # Vector env
    # --------------------------
    vector_env = gym.vector.SyncVectorEnv if not bool(cfg.env.async_envs) else gym.vector.AsyncVectorEnv
    envs = vector_env([make_env(cfg, i, run_name, larm) for i in range(int(cfg.ppo.num_envs))])

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Agent
    agent = Agent(envs, z_dim=int(cfg.armfm.z_dim), z_project_dim=int(cfg.armfm.z_project_dim)).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(cfg.ppo.learning_rate), eps=1e-5)

    # Storage
    obs = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs)) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs)) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs)), device=device)
    rewards = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs)), device=device)
    dones = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs)), device=device)
    values = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs)), device=device)
    zs = torch.zeros((int(cfg.ppo.num_steps), int(cfg.ppo.num_envs), int(cfg.armfm.z_dim)), device=device)

    # Reset
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=int(cfg.seed))
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(int(cfg.ppo.num_envs), device=device, dtype=torch.float32)

    # PPO loop
    for iteration in range(1, int(num_iterations) + 1):
        # LR anneal
        if bool(cfg.ppo.anneal_lr):
            frac = 1.0 - (iteration - 1.0) / float(num_iterations)
            lrnow = frac * float(cfg.ppo.learning_rate)
            optimizer.param_groups[0]["lr"] = lrnow

        # rollout
        for step in range(int(cfg.ppo.num_steps)):
            global_step += int(cfg.ppo.num_envs) * int(cfg.env.frameskip)
            obs[step] = next_obs
            dones[step] = next_done

            z_np = collect_current_z(envs, int(cfg.armfm.z_dim), enable_armfm=enable_armfm)
            z_t = torch.tensor(z_np, device=device)
            zs[step] = z_t

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, z_t)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.tensor(np.logical_or(terminations, truncations), device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_r = info["episode"]["r"]
                        ep_l = info["episode"]["l"] * int(cfg.env.frameskip)
                        log.info(f"global_step={global_step} episodic_return={float(ep_r)} episodic_length={int(ep_l)}")
                        if writer is not None:
                            writer.add_scalar("charts/episodic_return", float(ep_r), global_step)
                            writer.add_scalar("charts/episodic_length", float(ep_l), global_step)
                            if "armfm_rm_state" in info:
                                writer.add_text("armfm/last_rm_state", str(info["armfm_rm_state"]), global_step)

        # GAE
        with torch.no_grad():
            z_np = collect_current_z(envs, int(cfg.armfm.z_dim), enable_armfm=enable_armfm)
            next_z = torch.tensor(z_np, device=device)
            next_value = agent.get_value(next_obs, next_z).reshape(1, -1)

            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(int(cfg.ppo.num_steps))):
                if t == int(cfg.ppo.num_steps) - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + float(cfg.ppo.gamma) * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + float(cfg.ppo.gamma) * float(cfg.ppo.gae_lambda) * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_zs = zs.reshape((-1, int(cfg.armfm.z_dim)))

        # Optimize
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(int(cfg.ppo.update_epochs)):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > float(cfg.ppo.clip_coef)).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if bool(cfg.ppo.norm_adv):
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - float(cfg.ppo.clip_coef), 1 + float(cfg.ppo.clip_coef))
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if bool(cfg.ppo.clip_vloss):
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -float(cfg.ppo.clip_coef),
                        float(cfg.ppo.clip_coef),
                    )
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - float(cfg.ppo.ent_coef) * entropy_loss + v_loss * float(cfg.ppo.vf_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), float(cfg.ppo.max_grad_norm))
                optimizer.step()

            if cfg.ppo.target_kl is not None and float(approx_kl.item()) > float(cfg.ppo.target_kl):
                break

        # Metrics
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        rollout_mean_reward = float(rewards.mean().item())

        if writer is not None:
            writer.add_scalar("charts/learning_rate", float(optimizer.param_groups[0]["lr"]), global_step)
            writer.add_scalar("losses/value_loss", float(v_loss.item()), global_step)
            writer.add_scalar("losses/policy_loss", float(pg_loss.item()), global_step)
            writer.add_scalar("losses/entropy", float(entropy_loss.item()), global_step)
            writer.add_scalar("losses/old_approx_kl", float(old_approx_kl.item()), global_step)
            writer.add_scalar("losses/approx_kl", float(approx_kl.item()), global_step)
            writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
            writer.add_scalar("charts/SPS", float(sps), global_step)
            if bool(cfg.log.flush_tb):
                writer.flush()

        if bool(cfg.log.csv):
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step},{iteration},{optimizer.param_groups[0]['lr']:.8g},"
                    f"{v_loss.item():.8g},{pg_loss.item():.8g},{entropy_loss.item():.8g},"
                    f"{approx_kl.item():.8g},{float(np.mean(clipfracs)):.8g},"
                    f"{explained_var:.8g},{sps},{rollout_mean_reward:.8g}\n"
                )

        if iteration % int(cfg.log.log_interval) == 0:
            log.info(
                f"iteration={iteration}/{num_iterations} step={global_step} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"pg={pg_loss.item():.4f} v={v_loss.item():.4f} "
                f"ent={entropy_loss.item():.4f} kl={approx_kl.item():.4f} "
                f"clip={float(np.mean(clipfracs)):.4f} ev={float(explained_var):.4f} "
                f"sps={sps} r_mean={rollout_mean_reward:.4f}"
            )

        # Save agent (state_dict)
        if bool(cfg.save.save_agent):
            save_every = max(1, int(num_iterations) // max(1, int(cfg.save.save_num)))
            if (iteration % save_every == 0) or (iteration == int(num_iterations)):
                path = os.path.join("agents", f"agent_step_{global_step}.pt")
                torch.save(agent.state_dict(), path)
                log.info(f"saved={path}")
                if wandb_run is not None:
                    wandb_run.save(path)

    envs.close()
    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
