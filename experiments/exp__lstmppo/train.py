# -*- coding: utf-8 -*-
# Craftium PPO(LSTM) Experiment Runner
#
# 機能:
# - Hydraによる設定管理
# - WandBによる実験トラッキング
# - ログ出力およびタイムスタンプ付きの成果物保存
#
# Base implementation: CleanRL ppo_atari_lstm.py
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py

import os
import random
import time
import datetime
import logging
from typing import Optional, Tuple, Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import hydra
from omegaconf import DictConfig, OmegaConf

import craftium


# --------------------------
# Env builder
# --------------------------
class AddChannelDim(gym.ObservationWrapper):
    """(H,W) -> (1,H,W)"""
    def __init__(self, env):
        super().__init__(env)
        assert len(env.observation_space.shape) == 2
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, h, w),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return obs[None, :, :]


def make_env(cfg, idx, run_name):
    def thunk():
        craftium_kwargs = dict(
            run_dir_prefix=cfg.mt_wd,
            mt_port=cfg.mt_port + idx,
            frameskip=cfg.frameskip,
            rgb_observations=True,
            mt_listen_timeout=300_000,
            seed=int(cfg.seed) + idx,
        )

        if cfg.capture_video and idx == 0:
            env = gym.make(cfg.env_id, render_mode="rgb_array", **craftium_kwargs)

            video_folder = "videos"
            os.makedirs(video_folder, exist_ok=True)

            # "/" が含まれるとディレクトリ階層として解釈されてエラーになるため、置換する
            safe_prefix = run_name.replace("/", "__")

            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix=safe_prefix,
            )
        else:
            env = gym.make(cfg.env_id, **craftium_kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, 84)
        env = AddChannelDim(env)   # (84, 84) -> (1, 84, 84)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
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

        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size)).float()
        done = done.float()

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

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 二重登録を防ぐ（Hydra環境での再実行対策）
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
# Checkpoint Helpers
# --------------------------
def _find_latest_checkpoint_in_dir(ckpt_dir: str) -> Optional[str]:
    if not ckpt_dir or (not os.path.isdir(ckpt_dir)):
        return None

    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None

    # "ckpt_iter_000123_step_000000001234.pt" を想定して辞書順でもほぼOK
    files = sorted(files)
    return os.path.join(ckpt_dir, files[-1])


def resolve_resume_path(resume_path: str) -> Optional[str]:
    """
    resume_path が
      - 空文字: None
      - ファイル: そのまま
      - ディレクトリ: 中の最新ptを探す
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
    """
    先に run_name / wandb_run_id などだけ欲しい用途。
    重いstate_dictも読むが、ここは頻繁に呼ばない想定。
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
        meta = {}
        for k in ["run_name", "wandb_run_id", "global_step", "iteration"]:
            if k in ckpt:
                meta[k] = ckpt[k]
        return meta
    except Exception:
        return {}


def save_checkpoint(
    path: str,
    agent: nn.Module,
    optimizer: optim.Optimizer,
    global_step: int,
    iteration: int,
    cfg: DictConfig,
    run_name: str,
    wandb_run_id: Optional[str],
    next_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    ckpt = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "iteration": int(iteration),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "run_name": str(run_name),
        "wandb_run_id": wandb_run_id,
        # 乱数状態（続きの挙動を安定させたい場合）
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # LSTM state（任意）
        "next_lstm_state": (
            (next_lstm_state[0].detach().cpu(), next_lstm_state[1].detach().cpu())
            if next_lstm_state is not None
            else None
        ),
    }
    torch.save(ckpt, path)

    # 「最新」を上書きで持つ（探しやすくする）
    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    try:
        torch.save(ckpt, latest_path)
    except Exception:
        pass


def load_checkpoint(
    path: str,
    agent: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device)

    agent.load_state_dict(ckpt["agent_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    global_step = int(ckpt.get("global_step", 0))
    iteration = int(ckpt.get("iteration", 0))
    run_name = str(ckpt.get("run_name", ""))
    wandb_run_id = ckpt.get("wandb_run_id", None)

    # 乱数復元
    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
    if "rng_numpy" in ckpt:
        np.random.set_state(ckpt["rng_numpy"])
    if "rng_torch" in ckpt:
        torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

    next_lstm_state = None
    if ckpt.get("next_lstm_state", None) is not None:
        h, c = ckpt["next_lstm_state"]
        next_lstm_state = (h.to(device), c.to(device))

    return global_step, iteration, run_name, wandb_run_id, next_lstm_state


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    log = logging.getLogger(__name__)

    # seed決定（nullなら実行時刻をseedにする）
    t = int(time.time())
    if cfg.seed is None:
        cfg.seed = t

    # --------------------------
    # Sanity check: LSTM PPOでは num_minibatches は num_envs の約数である必要がある
    # - num_envs=1, num_minibatches>1 の場合、必ず AssertionError になるため自動補正する
    # --------------------------
    cfg.num_envs = int(cfg.num_envs)
    cfg.num_steps = int(cfg.num_steps)
    cfg.frameskip = int(cfg.frameskip)
    cfg.num_minibatches = int(cfg.num_minibatches)

    if cfg.num_envs < 1:
        log.warning(f"num_envs adjusted: {cfg.num_envs} -> 1")
        cfg.num_envs = 1

    req_num_minibatches = int(cfg.num_minibatches)
    cfg.num_minibatches = max(1, min(int(cfg.num_minibatches), int(cfg.num_envs)))
    while int(cfg.num_envs) % int(cfg.num_minibatches) != 0:
        cfg.num_minibatches -= 1
        if int(cfg.num_minibatches) <= 1:
            cfg.num_minibatches = 1
            break

    if int(cfg.num_minibatches) != req_num_minibatches:
        log.warning(
            f"num_minibatches adjusted: {req_num_minibatches} -> {int(cfg.num_minibatches)} "
            f"(num_envs={int(cfg.num_envs)} must be divisible)"
        )

    # runtime計算
    batch_size = int(cfg.num_envs * cfg.num_steps)
    minibatch_size = int(batch_size // cfg.num_minibatches)
    num_iterations = (int(cfg.total_timesteps) // int(cfg.frameskip)) // int(batch_size)

    # 通常のrun_name（resume時はcheckpoint側を優先できる）
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{t}"

    # 出力構造（Hydraがrun.dirにchdir済み）
    os.makedirs("agents", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # --------------------------
    # resume_pathの解決
    # --------------------------
    resume_ckpt_path = None
    if bool(getattr(cfg, "resume", False)):
        # 指定がなければ、このrunのcheckpoints/latest.ptを優先して探す
        rp = str(getattr(cfg, "resume_path", "")).strip()
        if rp == "":
            # まずはこのディレクトリ内
            local_latest = resolve_resume_path(os.path.join("checkpoints", "latest.pt"))
            if local_latest is None:
                # checkpoints/ の中の最新も探す
                local_latest = resolve_resume_path("checkpoints")
            resume_ckpt_path = local_latest
        else:
            resume_ckpt_path = resolve_resume_path(rp)

        if resume_ckpt_path is None:
            log.warning("resume=true but no checkpoint found. start from scratch.")
        else:
            meta = peek_checkpoint_metadata(resume_ckpt_path)
            if "run_name" in meta and isinstance(meta["run_name"], str) and meta["run_name"] != "":
                run_name = meta["run_name"]

    log.info(f"run_name={run_name}")
    log.info(f"output_dir={os.getcwd()}")

    # --------------------------
    # W&B（resume対応：）
    # --------------------------
    wandb_run = None
    wandb_run_id_for_resume = None

    if cfg.track:
        import wandb

        # HydraのDictConfigはそのままwandb.configへ渡すと事故るのでdictへ変換する
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)

        # resume時は checkpoint から wandb_run_id を読む（可能なら同じRunへ復帰）
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
                # settings=wandb.Settings(start_method="thread"),
            )

    # TRY NOT TO MODIFY: seeding
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch.backends.cudnn.deterministic = bool(cfg.torch_deterministic)

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
    log.info(f"device={device}")

    # env setup
    vector_env = gym.vector.SyncVectorEnv if not cfg.async_envs else gym.vector.AsyncVectorEnv
    envs = vector_env([make_env(cfg, i, run_name) for i in range(int(cfg.num_envs))])

    log.info(f"obs_shape={envs.single_observation_space.shape}")

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(cfg.learning_rate), eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((int(cfg.num_steps), int(cfg.num_envs)) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((int(cfg.num_steps), int(cfg.num_envs)) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((int(cfg.num_steps), int(cfg.num_envs))).to(device)
    rewards = torch.zeros((int(cfg.num_steps), int(cfg.num_envs))).to(device)
    dones = torch.zeros((int(cfg.num_steps), int(cfg.num_envs))).to(device)
    values = torch.zeros((int(cfg.num_steps), int(cfg.num_envs))).to(device)

    # env reset
    next_obs, _ = envs.reset(seed=int(cfg.seed))
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros(int(cfg.num_envs), device=device, dtype=torch.float32)

    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, int(cfg.num_envs), agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, int(cfg.num_envs), agent.lstm.hidden_size).to(device),
    )

    # --------------------------
    # Resume: agent, optimizer, global_step, iteration を復元
    # --------------------------
    global_step = 0
    resume_iteration = 0
    wandb_run_id_to_save = wandb_run.id if wandb_run is not None else None

    if bool(getattr(cfg, "resume", False)) and resume_ckpt_path is not None:
        log.info(f"loading checkpoint: {resume_ckpt_path}")
        try:
            global_step, resume_iteration, ckpt_run_name, ckpt_wandb_id, loaded_lstm = load_checkpoint(
                resume_ckpt_path, agent, optimizer, device
            )
            if ckpt_run_name:
                run_name = ckpt_run_name
            if ckpt_wandb_id is not None:
                wandb_run_id_to_save = ckpt_wandb_id

            if loaded_lstm is not None:
                next_lstm_state = loaded_lstm

            log.info(f"resumed: iteration={resume_iteration} global_step={global_step}")
        except Exception as e:
            log.warning(f"failed to load checkpoint: {e}. start from scratch.")
            global_step = 0
            resume_iteration = 0

    start_time = time.time()

    # --------------------------
    # main loop
    # --------------------------
    start_iteration = int(resume_iteration) + 1

    for iteration in range(start_iteration, num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # LR anneal
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * float(cfg.learning_rate)
            optimizer.param_groups[0]["lr"] = lrnow

        # rollout
        for step in range(int(cfg.num_steps)):
            global_step += int(cfg.num_envs) * int(cfg.frameskip)
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.tensor(next_done, device=device).float()

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # 配列の場合は .item() で値を取り出す
                        ep_r = info["episode"]["r"]
                        if isinstance(ep_r, np.ndarray):
                            ep_r = ep_r.item()
                        ep_r = float(ep_r)

                        ep_l = info["episode"]["l"]
                        if isinstance(ep_l, np.ndarray):
                            ep_l = ep_l.item()
                        ep_l = int(ep_l) * int(cfg.frameskip)

                        log.info(f"global_step={global_step} episodic_return={ep_r} episodic_length={ep_l}")

                        if wandb_run is not None:
                            wandb_run.log(
                                {"charts/episodic_return": ep_r, "charts/episodic_length": ep_l},
                                step=global_step,
                            )

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t_ in reversed(range(int(cfg.num_steps))):
                if t_ == int(cfg.num_steps) - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t_ + 1]
                    nextvalues = values[t_ + 1]
                delta = rewards[t_] + float(cfg.gamma) * nextvalues * nextnonterminal - values[t_]
                advantages[t_] = lastgaelam = (
                    delta + float(cfg.gamma) * float(cfg.gae_lambda) * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update
        envsperbatch = int(cfg.num_envs) // int(cfg.num_minibatches)
        envinds = np.arange(int(cfg.num_envs))
        flatinds = np.arange(batch_size).reshape(int(cfg.num_steps), int(cfg.num_envs))

        clipfracs = []
        for epoch in range(int(cfg.update_epochs)):
            np.random.shuffle(envinds)
            for start in range(0, int(cfg.num_envs), envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > float(cfg.clip_coef)).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - float(cfg.clip_coef), 1 + float(cfg.clip_coef))
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -float(cfg.clip_coef),
                        float(cfg.clip_coef),
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - float(cfg.ent_coef) * entropy_loss + v_loss * float(cfg.vf_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), float(cfg.max_grad_norm))
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > float(cfg.target_kl):
                break

        # logging
        current_time = time.time()
        elapsed_seconds = current_time - start_time
        # 秒数を "H:MM:SS" 形式の文字列に変換
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

        sps = int(global_step / (time.time() - start_time))
        metrics = {
            "charts/SPS": sps,
            "charts/elapsed_time": elapsed_seconds,
            "losses/value_loss": float(v_loss.item()),
            "losses/policy_loss": float(pg_loss.item()),
            "losses/entropy": float(entropy_loss.item()),
            "losses/old_approx_kl": float(old_approx_kl.item()),
            "losses/approx_kl": float(approx_kl.item()),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "charts/learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        log.info(f"iteration={iteration}/{num_iterations} SPS={sps}")

        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)

        # save agent（既存）
        if cfg.save_agent:
            save_every = max(1, num_iterations // int(cfg.save_num))
            if (iteration % save_every == 0) or (iteration == num_iterations):
                path = f"agents/agent_step_{global_step}.pt"
                torch.save(agent.state_dict(), path)
                log.info(f"saved={path}")
                if wandb_run is not None:
                    wandb_run.save(path)

        # Save Checkpoint
        if bool(getattr(cfg, "save_checkpoint", True)):
            ckpt_every = int(getattr(cfg, "checkpoint_every", 1))
            if ckpt_every <= 0:
                ckpt_every = 1

            if (iteration % ckpt_every == 0) or (iteration == num_iterations):
                ckpt_path = f"checkpoints/ckpt_iter_{iteration:06d}_step_{global_step:012d}.pt"
                save_checkpoint(
                    ckpt_path,
                    agent=agent,
                    optimizer=optimizer,
                    global_step=global_step,
                    iteration=iteration,
                    cfg=cfg,
                    run_name=run_name,
                    wandb_run_id=wandb_run_id_to_save,
                    next_lstm_state=next_lstm_state,
                )
                log.info(f"checkpoint_saved={ckpt_path}")
                if wandb_run is not None:
                    wandb_run.save(ckpt_path)

    envs.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
