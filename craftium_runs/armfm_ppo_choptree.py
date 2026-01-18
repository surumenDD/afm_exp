# -*- coding: utf-8 -*-
# ARM-FM for Craftium ChopTree with PPO (CleanRL-style) + FM-generated LARM integration
# - Fully self-contained implementation that:
#   1) Generates a Language-Aligned Reward Machine (LARM) via an arbitrary FM API (LLM/VLM via API)
#   2) Compiles executable Python labeling functions for events (Generator–Critic loop with self-repair)
#   3) Embeds state instructions and conditions the policy on the current RM state's embedding
#   4) Injects dense, structured rewards from the LARM into PPO training
#   5) Logs all artifacts (prompts, generated specs, code, configs) for reproducibility
#
# Notes:
# - This script uses only API calls to an arbitrary model provider (OpenAI-compatible JSON APIs assumed).
# - If no API is configured, it falls back to a deterministic, local LARM for ChopTree so you can still run.
# - We do not assume access to Craftium's private internals; we expose a wrapper attribute `_armfm_last_reward`
#   and `_armfm_last_action` to the labeling functions so that they can detect events like "tree chopped".
# - The RM reward is added to the environment reward: R_total = R_env + R_RM (configurable scale).
#
# USAGE (common):
#   python armfm_ppo_choptree.py --env-id Craftium/ChopTree-v0 --total-timesteps 1000000 --num-envs 4
#
# FM API (set via CLI or env):
#   --fm-api-base https://api.openai.com/v1 --fm-api-key $OPENAI_API_KEY
#   --fm-chat-model gpt-4o-mini --fm-embed-model text-embedding-3-large
#
# Reproducibility:
#   - Saves all FM prompts, RM specs, generated code, and configs under runs/<run_name>/armfm_artifacts/
#   - Agent checkpoints (optional) saved under agents/<run_name>/
#
# Reference:
#   Base PPO loop adapted from CleanRL's ppo_atari.py
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
#
# ----------------------------------------------------------------------------------------------

# fmt: off
import os
import re
import json
import time
import random
import hashlib
import textwrap
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Callable, Tuple
from dotenv import load_dotenv
load_dotenv()

import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import requests

# Craftium must be installed and available
import craftium

# fmt: on

# --------------------------
# Arguments
# --------------------------


@dataclass
class Args:
    # Experiment meta
    exp_name: str = "armfm_ppo_choptree"
    seed: Optional[int] = None
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "craftium"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    # Craftium env runtime
    async_envs: bool = False
    mt_wd: str = os.getenv("WORK_DIR", "./craftium_runs")
    out_dir: str = os.getenv("OUT_DIR", "../output")
    frameskip: int = 4
    save_agent: bool = False
    save_num: int = 5
    mt_port: int = 49155
    fps_max: int = 200
    pmul: float = 1.0  # Optional[int] = None

    # PPO hyperparams
    env_id: str = "Craftium/ChopTree-v0"
    total_timesteps: int = int(1e6)
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # ARM-FM integration
    enable_armfm: bool = True
    # additional dense reward from RM; keep 0.0 for ChopTree (RM uses env signals)
    rm_reward_scale: float = 0.0
    z_dim: int = 256  # embedding dimension for RM state instructions
    z_project_dim: int = 128  # projection for policy conditioning
    # FM API configuration (OpenAI-compatible or similar)
    fm_api_base: Optional[str] = None
    fm_api_key: Optional[str] = None
    fm_chat_model: Optional[str] = None
    fm_embed_model: Optional[str] = None
    fm_timeout_s: float = 60.0

    # FM gen/critic loop
    larm_gen_rounds: int = 2
    larm_max_retries: int = 3

    # Prompts & mission (ChopTree)
    auto_build_larm: bool = True
    # If you want to provide a custom mission/spec, you can override below:
    mission_text: str = (
        "密林にスポーンしたプレイヤーが鋼の斧で木を切る。木を切るたびに報酬 +1.0 を得るため、"
        "エピソード中にできるだけ多くの木を切る。"
    )
    # See "ENV_SPEC_FOR_LABELING" constant below for labeling-allowed attributes

    # Derived at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Logging
    log_interval: int = 1          # 何iterationごとにコンソールへ出すか
    csv_log: bool = True           # CSV保存するか
    flush_tb: bool = True          # TBを頻繁にflushするか


# --------------------------
# FM Client (OpenAI-compatible)
# --------------------------

class FMClient:
    def __init__(self, api_base: str, api_key: str, chat_model: str, embed_model: Optional[str], timeout_s: float = 60.0):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}"})

    def _request_with_retry(self, method: str, url: str, payload: dict, max_retries: int = 5) -> dict:
        """Make HTTP request with exponential backoff retry for rate limits."""
        for attempt in range(max_retries):
            resp = self.session.post(url, json=payload, timeout=self.timeout_s)
            if resp.status_code == 429:
                # Rate limited - wait and retry with exponential backoff
                wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
                print(
                    f"[FM API] Rate limited (429). Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            if not resp.ok:
                # Print detailed error for debugging
                print(f"[FM API] Error {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        # Final attempt
        resp = self.session.post(url, json=payload, timeout=self.timeout_s)
        if not resp.ok:
            print(f"[FM API] Error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: List[Dict[str, str]], temperature: float = 1.0, max_completion_tokens: int = 2000) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        data = self._request_with_retry("POST", url, payload)
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embed_model:
            # Deterministic pseudo-embedding fallback (reproducible hash -> vector)
            return [self._hash_embed(t) for t in texts]
        url = f"{self.api_base}/embeddings"
        payload = {
            "model": self.embed_model,
            "input": texts,
        }
        data = self._request_with_retry("POST", url, payload)
        return [item["embedding"] for item in data["data"]]

    @staticmethod
    def _hash_embed(text: str, dim: int = 256) -> List[float]:
        # Reproducible hash to numeric vector
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v.tolist()


# --------------------------
# ARM-FM prompts (no omission)
# --------------------------

# Reward Machine (LARM) generator prompt template (strict format)
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
- Only list non-zero rewards in REWARD_FUNCTION; others are assumed to be 0.
- STATE_INSTRUCTIONS are short natural language strings (Japanese) that describe the subgoal at each state.

Hints for ChopTree:
- The wrapper exposes env._armfm_last_reward (last scalar env reward), env._armfm_last_action (last action id), and may store simple counters, e.g., env._armfm_chops_in_window.
- For densification, you can use events like "recently_chopped_tree" or "streak_progress" to encourage frequent chopping.
- Keep it compact (2~4 states).
"""

# Labeling function generator prompt (strict code output)
LABELING_GENERATOR_SYSTEM = """You are a Python labeling function generator for an RL Reward Machine.
You will implement one boolean function per event name in pure Python, using ONLY allowed attributes of the provided env wrapper.
Output ONLY valid Python code wrapped in ```python fences. No extra text or comments.
"""

# We explicitly define allowed attributes for the wrapper (we will actually provide them).
ENV_SPEC_FOR_LABELING = """Allowed env wrapper attributes and meaning:
- env._armfm_last_reward: float, last environment reward received on the previous step (0.0 or 1.0 in ChopTree).
- env._armfm_last_action: int, last action index taken (if available).
- env._armfm_step: int, current step counter in episode.
- env._armfm_recent_chops: int, chops counted in a short window (sliding window maintained by wrapper).
- env._armfm_recent_steps: int, steps counted in a short window.
- env._armfm_streak: int, current chopping streak length (wrapper maintained).
No imports allowed. Functions receive a single argument env and must return True/False.
Do not define any "else" event function.
"""

LABELING_GENERATOR_USER_TEMPLATE = """Reward Machine to implement events for:
{reward_machine_text}

Guidelines:
- Define exactly one function per event listed in the RM.
- Function name must match the event name exactly.
- Signature must be: def {{event}}(env): -> bool
- Use ONLY the allowed env wrapper attributes:
{env_spec}

Output rules:
- Only output Python code.
- No comments or extra text.
- Wrap code in ```python fences.
"""

# Critic prompt: fix broken RM or code with errors
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
# RM parsing and runtime
# --------------------------


class LARM:
    def __init__(self,
                 states: List[str],
                 initial_state: str,
                 transitions: Dict[Tuple[str, str], str],
                 rewards: Dict[Tuple[str, str, str], float],
                 state_instructions: Dict[str, str],
                 event_names: List[str],
                 event_funcs: Dict[str, Callable[[Any], bool]],
                 embeddings: Dict[str, List[float]]):
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
        # Evaluate events in order; if multiple events true, apply first (stable ordering by event_names)
        sigma = None
        for ev in self.event_names:
            try:
                if self.event_funcs[ev](env_wrapper):
                    sigma = ev
                    break
            except Exception:
                # robust fallback: ignore label errors
                continue
        next_state = self.transitions.get((self.current_state, sigma), None)
        if next_state is None:
            # else transition (if defined), else stay
            next_state = self.transitions.get(
                (self.current_state, "else"), self.current_state)
            r = self.rewards.get((self.current_state, "else", next_state), 0.0)
        else:
            r = self.rewards.get((self.current_state, sigma, next_state), 0.0)
        self.current_state = next_state
        return r

    def current_embedding(self) -> List[float]:
        if self.current_state in self.embeddings:
            return self.embeddings[self.current_state]
        example = next(iter(self.embeddings.values()), None)
        dim = len(example) if example is not None else 1
        return [0.0] * dim



class RMParser:
    RM_PATTERN = re.compile(
        r"STATES:\s*(?P<states>.+?)\s*INITIAL_STATE:\s*(?P<init>.+?)\s*TRANSITION_FUNCTION:\s*(?P<trans>.+?)\s*REWARD_FUNCTION:\s*(?P<rewards>.+?)\s*STATE_INSTRUCTIONS:\s*(?P<instr>.+)",
        re.S | re.I
    )

    @staticmethod
    def parse_reward_machine(text: str) -> Tuple[List[str], str, Dict[Tuple[str, str], str], Dict[Tuple[str, str, str], float], Dict[str, str], List[str]]:
        m = RMParser.RM_PATTERN.search(text)
        if not m:
            raise ValueError(
                "Failed to parse RM text. Ensure strict format and sections exist.")

        states_raw = m.group("states").strip()
        init_raw = m.group("init").strip()
        trans_raw = m.group("trans").strip()
        rewards_raw = m.group("rewards").strip()
        instr_raw = m.group("instr").strip()

        # STATES: u0, u1, ...
        states = [s.strip() for s in states_raw.split(",") if s.strip()]
        initial_state = init_raw

        # TRANSITION_FUNCTION lines
        transitions: Dict[Tuple[str, str], str] = {}
        event_names = set()
        for line in trans_raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            # format: (uX, event) -> uY
            m2 = re.match(r"\(([^,]+),\s*([^)]+)\)\s*->\s*(\S+)", line)
            if not m2:
                continue
            u_from = m2.group(1).strip()
            ev = m2.group(2).strip()
            u_to = m2.group(3).strip()
            transitions[(u_from, ev)] = u_to
            if ev != "else":
                event_names.add(ev)

        # REWARD_FUNCTION lines
        rewards: Dict[Tuple[str, str, str], float] = {}
        for line in rewards_raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            # format: (uX, event, uY) -> X
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

        # STATE_INSTRUCTIONS lines
        state_instructions: Dict[str, str] = {}
        for line in instr_raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            # format: uX: instruction text
            k, v = line.split(":", 1)
            state_instructions[k.strip()] = v.strip()

        return states, initial_state, transitions, rewards, state_instructions, sorted(event_names)


# --------------------------
# Generator–Critic LARM builder
# --------------------------

class LARMBuilder:
    def __init__(self, fm: Optional[FMClient], artifacts_dir: str, z_dim: int, max_retries: int = 3):
        self.fm = fm
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.z_dim = z_dim
        self.max_retries = max_retries

    def build_for_choptree(self, mission_text: str, gen_rounds: int) -> Tuple[LARM, Dict[str, Any]]:
        # 1) Generate RM spec (plaintext) with possible self-improvement loops
        rm_text, rm_gen_logs = self._generate_rm_text(mission_text, gen_rounds)

        # 2) Parse RM spec
        states, initial_state, transitions, rewards, state_instructions, event_names = RMParser.parse_reward_machine(
            rm_text)

        # 3) Generate labeling functions code
        label_code, code_logs = self._generate_labeling_code(
            rm_text, event_names)

        # 4) Compile labeling functions
        event_funcs = self._compile_labeling_functions(label_code, event_names)

        # 5) Embed instructions
        embeddings = self._embed_instructions(state_instructions)

        # 6) Construct LARM
        larm = LARM(
            states=states,
            initial_state=initial_state,
            transitions=transitions,
            rewards=rewards,
            state_instructions=state_instructions,
            event_names=event_names,
            event_funcs=event_funcs,
            embeddings=embeddings
        )

        # 7) Save artifacts
        with open(os.path.join(self.artifacts_dir, "rm_spec.txt"), "w", encoding="utf-8") as f:
            f.write(rm_text)
        with open(os.path.join(self.artifacts_dir, "labeling_code.py"), "w", encoding="utf-8") as f:
            f.write(label_code)
        with open(os.path.join(self.artifacts_dir, "state_instructions.json"), "w", encoding="utf-8") as f:
            json.dump(state_instructions, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.artifacts_dir, "event_names.json"), "w", encoding="utf-8") as f:
            json.dump(event_names, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.artifacts_dir, "rm_generation_logs.json"), "w", encoding="utf-8") as f:
            json.dump({"rm_gen_logs": rm_gen_logs, "code_gen_logs": code_logs},
                      f, ensure_ascii=False, indent=2)

        return larm, {"rm_text": rm_text, "label_code": label_code, "gen_logs": {"rm": rm_gen_logs, "code": code_logs}}

    def _generate_rm_text(self, mission_text: str, gen_rounds: int) -> Tuple[str, Dict[str, Any]]:
        logs = {"rounds": []}
        if self.fm is None:
            # Fallback deterministic RM (compact)
            rm_text = textwrap.dedent("""\
            ```plaintext
            STATES: u0
            INITIAL_STATE: u0
            TRANSITION_FUNCTION:
            (u0, chopped_tree_recently) -> u0
            (u0, else) -> u0
            REWARD_FUNCTION:
            (u0, chopped_tree_recently, u0) -> 0.0
            STATE_INSTRUCTIONS:
            u0: 近くの木を見つけて斧で切り続ける
            ```
            """).strip()
            logs["rounds"].append({"fallback": True, "rm_text": rm_text})
            return rm_text.strip("`"), logs

        sys = {"role": "system", "content": RM_GENERATOR_SYSTEM}
        user = {"role": "user", "content": RM_GENERATOR_USER_TEMPLATE.format(
            mission_text=mission_text)}
        content = self.fm.chat([sys, user])
        print(f"[FM] Raw RM response:\n{content}\n{'='*50}")
        logs["rounds"].append({"initial": content})
        rm_text = self._extract_fenced(content, "plaintext")
        if not rm_text:
            # Try without language specifier (just ```)
            rm_text = self._extract_fenced(content, "") or content
        print(f"[FM] Extracted RM text:\n{rm_text}\n{'='*50}")

        # Basic validator
        for r in range(1, gen_rounds):
            try:
                _ = RMParser.parse_reward_machine(rm_text)
                break
            except Exception as e:
                print(f"[FM] RM parse error (attempt {r}): {e}")
                if r >= self.max_retries:
                    break
                critic_sys = {"role": "system", "content": CRITIC_SYSTEM}
                critic_user = {
                    "role": "user", "content": CRITIC_USER_TEMPLATE_RM.format(error_text=str(e))}
                fixed = self.fm.chat([critic_sys, critic_user])
                fixed_txt = self._extract_fenced(fixed, "plaintext") or fixed
                rm_text = fixed_txt
                logs["rounds"].append({"repair": fixed_txt, "error": str(e)})
        return rm_text, logs

    def _generate_labeling_code(self, rm_text: str, event_names: List[str]) -> Tuple[str, Dict[str, Any]]:
        logs = {"attempts": []}
        if self.fm is None:
            # Deterministic default labeling: event 'chopped_tree_recently' checks last reward
            code = textwrap.dedent("""\
            ```python
            def chopped_tree_recently(env):
                # last reward == +1.0 means a chop occurred on previous step
                return float(getattr(env, "_armfm_last_reward", 0.0)) >= 1.0
            ```
            """).strip()
            logs["attempts"].append({"fallback": True, "code": code})
            return code.strip("`"), logs

        sys = {"role": "system", "content": LABELING_GENERATOR_SYSTEM}
        u = {"role": "user", "content": LABELING_GENERATOR_USER_TEMPLATE.format(
            reward_machine_text=rm_text, env_spec=ENV_SPEC_FOR_LABELING)}
        content = self.fm.chat([sys, u])
        code = self._extract_fenced(content, "python") or content
        logs["attempts"].append({"initial": code})

        # Try compile; if fails, repair via critic
        for k in range(self.max_retries):
            try:
                _ = self._compile_labeling_functions(code, event_names)
                break
            except Exception as e:
                critic_sys = {"role": "system", "content": CRITIC_SYSTEM}
                critic_user = {
                    "role": "user", "content": CRITIC_USER_TEMPLATE_CODE.format(error_text=str(e))}
                fixed = self.fm.chat([critic_sys, critic_user])
                code = self._extract_fenced(fixed, "python") or fixed
                logs["attempts"].append({"repair": code, "error": str(e)})
        return code, logs

    def _compile_labeling_functions(self, code: str, event_names: List[str]) -> Dict[str, Callable]:
        # Extract code body (strip fences if present)
        src = self._strip_fences(code)
        glb: Dict[str, Any] = {}
        loc: Dict[str, Any] = {}
        exec(src, glb, loc)
        funcs: Dict[str, Callable] = {}
        for ev in event_names:
            if ev == "else":
                continue
            if ev not in loc:
                raise ValueError(f"Missing labeling function for event: {ev}")
            if not callable(loc[ev]):
                raise ValueError(f"Event {ev} is not a function")
            funcs[ev] = loc[ev]
        return funcs

    def _embed_instructions(self, state_instructions: Dict[str, str]) -> Dict[str, List[float]]:
        keys = list(state_instructions.keys())
        texts = [state_instructions[k] for k in keys]
        if self.fm is None:
            vecs = [FMClient._hash_embed(t, dim=self.z_dim) for t in texts]
        else:
            vecs = self.fm.embed(texts)
            # ensure consistent dimension
            vecs = [v[:self.z_dim] + [0.0] *
                    max(0, self.z_dim - len(v)) for v in vecs]
        return {k: v for k, v in zip(keys, vecs)}

    @staticmethod
    def _extract_fenced(text: str, lang: str) -> Optional[str]:
        # Try with specific language first
        if lang:
            pat = re.compile(rf"```{lang}\s*\n?(.+?)\s*```", re.S | re.I)
            m = pat.search(text)
            if m:
                return m.group(1).strip()
        # Try any code fence (```...```)
        pat = re.compile(r"```\w*\s*\n?(.+?)\s*```", re.S)
        m = pat.search(text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            # remove first fence line
            text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = "\n".join(text.splitlines()[:-1])
        return text.strip()


# --------------------------
# ARM-FM environment wrapper for LARM execution
# --------------------------

class ARMFMEnvWrapper(gym.Wrapper):
    def __init__(self, env, larm: LARM, rm_reward_scale: float = 0.0, recent_window: int = 50):
        super().__init__(env)
        self.larm = larm
        self.rm_reward_scale = rm_reward_scale
        self.recent_window = recent_window

        # Exposed attributes for labeling functions
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
        self._armfm_last_action = int(action) if np.isscalar(
            action) or isinstance(action, (int, np.integer)) else -1
        self._armfm_last_reward = float(reward)
        # update rolling window stats for labeling
        self._recent_rewards.append(self._armfm_last_reward)
        if len(self._recent_rewards) > self.recent_window:
            self._recent_rewards.pop(0)
        self._armfm_recent_chops = sum(
            1 for r in self._recent_rewards if r >= 1.0)
        self._armfm_recent_steps = min(self._armfm_step, self.recent_window)
        self._armfm_streak = self._armfm_streak + \
            1 if self._armfm_last_reward >= 1.0 else 0

        rm_r = self.larm.step(self)
        total_r = reward + self.rm_reward_scale * rm_r
        # attach RM info
        info = info or {}
        info["armfm_rm_state"] = self.larm.current_state
        info["armfm_rm_reward"] = rm_r
        info["armfm_total_reward"] = total_r
        return obs, total_r, terminated, truncated, info

    def current_state_embedding(self) -> np.ndarray:
        z = self.larm.current_embedding()
        return np.asarray(z, dtype=np.float32)


# --------------------------
# PPO Agent (image + RM state embedding conditioning)
# --------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, z_dim: int, z_project_dim: int):
        super().__init__()
        # Visual encoder (FrameStack of 4 grayscale/RGB frames)
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

        # Infer flattened size by passing a dummy tensor once
        with torch.no_grad():
            dummy = torch.zeros(1, *envs.single_observation_space.shape)
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
        self.actor = layer_init(
            nn.Linear(512, envs.single_action_space.n), std=0.01)
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

class GrayscaleToRGBRenderWrapper(gym.Wrapper):
    """Wrapper that converts grayscale render output to RGB for video recording."""

    def render(self):
        frame = self.env.render()
        if frame is not None and frame.ndim == 2:
            # Convert grayscale (H, W) to RGB (H, W, 3)
            frame = np.stack([frame, frame, frame], axis=-1)
        return frame


def make_env(env_id, idx, fps_max, pmul, capture_video, run_name, mt_port, mt_wd, frameskip, seed, larm_wrapper_ctor):
    def thunk():
        craftium_kwargs = dict(
            run_dir_prefix=mt_wd,
            mt_port=mt_port,
            frameskip=frameskip,
            # Keep grayscale for all envs (consistent observation space)
            rgb_observations=False,
            seed=seed,
            fps_max=fps_max,
            pmul=pmul,
        )
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **craftium_kwargs)
            # Convert grayscale render to RGB
            env = GrayscaleToRGBRenderWrapper(env)
            env = gym.wrappers.RecordVideo(
                env, f"{args.out_dir}/videos/{run_name}")
        else:
            env = gym.make(env_id, **craftium_kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FrameStack(env, 4)
        if larm_wrapper_ctor is not None:
            env = larm_wrapper_ctor(env)
        return env
    return thunk


# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    # Derive sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = (args.total_timesteps //
                           args.frameskip) // args.batch_size

    # Seed
    tnow = int(time.time())
    if args.seed is None:
        args.seed = tnow
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda":
        print(f"[Device] GPU={torch.cuda.get_device_name(0)}")

    # Run name and logging
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{tnow}"
    # log dirs
    tb_dir = os.path.join(args.out_dir, "runs", run_name)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"[Run] {run_name}")
    print(f"[TensorBoard] {tb_dir}")

    writer = SummaryWriter(tb_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{k}|{v}|" for k, v in asdict(args).items()]))
    )

    csv_path = os.path.join(tb_dir, "train_metrics.csv")
    if args.csv_log:
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "global_step,iteration,lr,value_loss,policy_loss,entropy,approx_kl,clipfrac,explained_var,sps,mean_reward\n")

    # W&B
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Agent saving path
    if args.save_agent:
        agent_path = f"{args.out_dir}/agents/{run_name}"
        os.makedirs(agent_path, exist_ok=True)

    # FM client setup
    fm_client = None
    fm_config = {}
    if args.enable_armfm and args.auto_build_larm:
        # Prefer CLI args; fallback to env vars
        api_base = args.fm_api_base or os.getenv("FM_API_BASE")
        api_key = args.fm_api_key or os.getenv("FM_API_KEY")
        chat_model = args.fm_chat_model or os.getenv("FM_CHAT_MODEL")
        embed_model = args.fm_embed_model or os.getenv("FM_EMBED_MODEL")
        if api_base and api_key and chat_model:
            fm_client = FMClient(api_base=api_base, api_key=api_key, chat_model=chat_model,
                                 embed_model=embed_model, timeout_s=args.fm_timeout_s)
            fm_config = {"api_base": api_base,
                         "chat_model": chat_model, "embed_model": embed_model}
        else:
            fm_client = None  # fallback local deterministic
            fm_config = {"api_base": None,
                         "chat_model": None, "embed_model": None}

    # Build LARM (or fallback)
    larm = None
    larm_info = {}
    artifacts_dir = os.path.join("runs", run_name, "armfm_artifacts")
    if args.enable_armfm:
        builder = LARMBuilder(fm=fm_client, artifacts_dir=artifacts_dir,
                              z_dim=args.z_dim, max_retries=args.larm_max_retries)
        larm, larm_info = builder.build_for_choptree(
            args.mission_text, gen_rounds=args.larm_gen_rounds)
        # Save prompts and FM config
        with open(os.path.join(artifacts_dir, "prompts_and_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "rm_generator_system": RM_GENERATOR_SYSTEM,
                "rm_generator_user_template": RM_GENERATOR_USER_TEMPLATE,
                "labeling_generator_system": LABELING_GENERATOR_SYSTEM,
                "labeling_env_spec": ENV_SPEC_FOR_LABELING,
                "labeling_generator_user_template": LABELING_GENERATOR_USER_TEMPLATE,
                "critic_system": CRITIC_SYSTEM,
                "critic_user_template_rm": CRITIC_USER_TEMPLATE_RM,
                "critic_user_template_code": CRITIC_USER_TEMPLATE_CODE,
                "mission_text": args.mission_text,
                "fm_config": fm_config
            }, f, ensure_ascii=False, indent=2)

    # Construct a wrapper ctor for envs
    def larm_wrapper_ctor(env):
        if not args.enable_armfm:
            return env
        return ARMFMEnvWrapper(env, larm=larm, rm_reward_scale=args.rm_reward_scale)

    # Vectorized envs
    vector_env = gym.vector.SyncVectorEnv if not args.async_envs else gym.vector.AsyncVectorEnv
    envs = vector_env([
        make_env(
            args.env_id, i, args.fps_max, args.pmul, args.capture_video, run_name,
            args.mt_port + i, args.mt_wd, args.frameskip, args.seed, larm_wrapper_ctor
        )
        for i in range(args.num_envs)
    ])

    assert isinstance(envs.single_action_space,
                      gym.spaces.Discrete), "only discrete action space is supported"

    # Agent
    agent = Agent(envs, z_dim=args.z_dim,
                  z_project_dim=args.z_project_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    zs = torch.zeros((args.num_steps, args.num_envs,
                     args.z_dim), device=device)

    # Reset
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    # Helper to read current z from first env in vector
    def collect_current_z():
        if not args.enable_armfm:
            return np.zeros((args.num_envs, args.z_dim), dtype=np.float32)

        zs_arr = envs.call("current_state_embedding")
        return np.stack(zs_arr, axis=0).astype(np.float32)


    # PPO loop
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs * args.frameskip
            obs[step] = next_obs
            dones[step] = next_done

            # collect z for each env
            if args.enable_armfm:
                with torch.no_grad():
                    zs_np = collect_current_z()
                z_t = torch.tensor(zs_np, device=device)
            else:
                z_t = torch.zeros((args.num_envs, args.z_dim), device=device)
            zs[step] = z_t

            # Action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, z_t)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Env step
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy())
            next_done = torch.tensor(np.logical_or(
                terminations, truncations), device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_r = info["episode"]["r"]
                        ep_l = info["episode"]["l"] * args.frameskip
                        print(
                            f"global_step={global_step}, episodic_return={ep_r}")
                        writer.add_scalar(
                            "charts/episodic_return", ep_r, global_step)
                        writer.add_scalar(
                            "charts/episodic_length", ep_l, global_step)
                        # log RM state if present
                        if "armfm_rm_state" in info:
                            writer.add_text(
                                "armfm/last_rm_state", str(info["armfm_rm_state"]), global_step)

        # GAE
        with torch.no_grad():
            # next z
            if args.enable_armfm:
                zs_np = collect_current_z()
                next_z = torch.tensor(zs_np, device=device)
            else:
                next_z = torch.zeros(
                    (args.num_envs, args.z_dim), device=device)
            next_value = agent.get_value(next_obs, next_z).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_zs = zs.reshape((-1, args.z_dim))

        # Optimize
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -
                        args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * \
                        torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Metrics
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac",
                          float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)

        rollout_mean_reward = rewards.mean().item()

        if iteration % args.log_interval == 0:
            print(
                f"[Iter {iteration:5d}/{args.num_iterations}] "
                f"step={global_step} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"pg={pg_loss.item():.4f} "
                f"v={v_loss.item():.4f} "
                f"ent={entropy_loss.item():.4f} "
                f"kl={approx_kl.item():.4f} "
                f"clip={float(np.mean(clipfracs)):.4f} "
                f"ev={explained_var:.4f} "
                f"sps={int(global_step / (time.time() - start_time))} "
                f"r_mean={rollout_mean_reward:.4f}"
            )

        # CSV追記
        if args.csv_log:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step},{iteration},"
                    f"{optimizer.param_groups[0]['lr']:.8g},"
                    f"{v_loss.item():.8g},"
                    f"{pg_loss.item():.8g},"
                    f"{entropy_loss.item():.8g},"
                    f"{approx_kl.item():.8g},"
                    f"{float(np.mean(clipfracs)):.8g},"
                    f"{explained_var:.8g},"
                    f"{int(global_step / (time.time() - start_time))},"
                    f"{rollout_mean_reward:.8g}\n"
                )

        # TBを頻繁にflushしたい場合
        if args.flush_tb:
            writer.flush()

        # Save agent
        if args.save_agent and (iteration % max(1, (args.num_iterations // max(1, args.save_num))) == 0 or iteration == args.num_iterations):
            print("Saving agent...")
            torch.save(agent.state_dict(),
                       f"{agent_path}/agent_step_{global_step}.pt")
            print("Agent saved.")

    envs.close()
    writer.close()
