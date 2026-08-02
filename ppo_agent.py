# ──────────────────────────────────────────────────────────
# ppo_agent.py  ←  PPO Actor-Critic network
# ──────────────────────────────────────────────────────────
#
#  Observation fed to the PPO agent:
#      z       : (B, VAE_LATENT_DIM)   – from VAE encoder μ
#      state   : (B, PPO_STATE_DIM)    – [speed/50, steer, dist_wp, cos_hdg, sin_hdg]
#
#  Action space  : continuous  [steer, accel]  (accel > 0 = throttle, < 0 = brake)
#  Actor output  : Gaussian μ + log_std  per action dim  (all tanh-squashed)
#  Critic output : scalar V(s)
# ──────────────────────────────────────────────────────────

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from parameters import (
    PPO_LATENT_DIM, PPO_STATE_DIM, PPO_ACTION_DIM,
    PPO_ACTOR_HIDDEN, PPO_CRITIC_HIDDEN,
    PPO_LR_ACTOR, PPO_LR_CRITIC,
    PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS,
    PPO_ENTROPY_COEF, PPO_VALUE_COEF, PPO_MAX_GRAD_NORM,
    PPO_EPOCHS_PER_UPDATE, PPO_MINIBATCH_SIZE,
    PPO_LIDAR_FEAT_DIM, PPO_MAX_LIDAR_POINTS
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLP_INPUT_DIM = PPO_LATENT_DIM + PPO_STATE_DIM + PPO_LIDAR_FEAT_DIM
FLAT_OBS_DIM  = PPO_LATENT_DIM + PPO_STATE_DIM + (PPO_MAX_LIDAR_POINTS * 3)


def flat_obs_dim(latent_dim: int) -> int:
    return int(latent_dim) + PPO_STATE_DIM + (PPO_MAX_LIDAR_POINTS * 3)


def squashed_log_prob(dist: Normal, raw: torch.Tensor) -> torch.Tensor:
    """
    log π(a|s) for a = tanh(u) with u ~ Normal(μ,σ).
    Change of variables: log π(a) = log π(u) − Σ log |da_i/du_i|.
    All action dims use tanh squashing.
    """
    lp = dist.log_prob(raw).sum(dim=-1)
    squashed = torch.tanh(raw)
    correction = torch.log((1.0 - squashed * squashed).clamp_min(1e-6)).sum(dim=-1)
    return lp - correction


# ── Shared helper ────────────────────────────────────────
def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


# ── PointNet Extractor ────────────────────────────────────
class PointNetExtractor(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        # x shape: (Batch, N, 3)
        features = self.mlp(x)
        global_feature, _ = torch.max(features, dim=1)
        return global_feature

# ── Actor ─────────────────────────────────────────────────
class Actor(nn.Module):
    """
    Gaussian actor:  outputs mean and log_std for each action dim.
    Actions are sampled during training and taken as mean at test time.
    """

    LOG_STD_MIN = -2.0      # std ≥ 0.13 — prevents deterministic collapse
    LOG_STD_MAX =  0.5

    def __init__(self, latent_dim: Optional[int] = None):
        super().__init__()
        ld = int(PPO_LATENT_DIM if latent_dim is None else latent_dim)
        self.latent_dim = ld
        mlp_in = ld + PPO_STATE_DIM + PPO_LIDAR_FEAT_DIM
        self.lidar_extractor = PointNetExtractor(out_dim=PPO_LIDAR_FEAT_DIM)
        self.net = _mlp(mlp_in, PPO_ACTOR_HIDDEN, PPO_ACTION_DIM)
        # Initialize log_std to a slightly negative value (-0.5) 
        # so initial exploration variance is gentler.
        self.log_std = nn.Parameter(torch.full((PPO_ACTION_DIM,), -0.5))   # learnable

    def forward(self, obs):
        core_dim = self.latent_dim + PPO_STATE_DIM
        core_obs = obs[:, :core_dim]
        lidar_flat = obs[:, core_dim:]
        lidar_pts = lidar_flat.view(-1, PPO_MAX_LIDAR_POINTS, 3)

        lidar_feats = self.lidar_extractor(lidar_pts)

        mlp_in = torch.cat([core_obs, lidar_feats], dim=1)

        mu = self.net(mlp_in)
        log = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, obs, deterministic=False):
        """
        Returns
        -------
        action     : np.ndarray  shape (action_dim,)
        log_prob   : float
        """
        dist = self(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.sample()

        log_prob = squashed_log_prob(dist, raw)

        # All actions use tanh: steer ∈ [-1,1], accel ∈ [-1,1]
        action = torch.tanh(raw)

        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def evaluate(self, obs, raw_action):
        """
        Used during PPO update; re-computes log_prob and entropy.
        raw_action is the *pre-squash* action stored during rollout.
        """
        dist     = self(obs)
        log_prob = squashed_log_prob(dist, raw_action)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ── Critic ─────────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, latent_dim: Optional[int] = None):
        super().__init__()
        ld = int(PPO_LATENT_DIM if latent_dim is None else latent_dim)
        self.latent_dim = ld
        mlp_in = ld + PPO_STATE_DIM + PPO_LIDAR_FEAT_DIM
        self.lidar_extractor = PointNetExtractor(out_dim=PPO_LIDAR_FEAT_DIM)
        self.net = _mlp(mlp_in, PPO_CRITIC_HIDDEN, 1)

    def forward(self, obs):
        core_dim = self.latent_dim + PPO_STATE_DIM
        core_obs = obs[:, :core_dim]
        lidar_flat = obs[:, core_dim:]
        lidar_pts = lidar_flat.view(-1, PPO_MAX_LIDAR_POINTS, 3)
        
        lidar_feats = self.lidar_extractor(lidar_pts)
        
        mlp_in = torch.cat([core_obs, lidar_feats], dim=1)
        return self.net(mlp_in).squeeze(-1)


# ── Rollout Buffer ─────────────────────────────────────────
class RolloutBuffer:
    """Stores a fixed-length trajectory and computes advantages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs        = []   # torch tensors
        self.actions    = []   # raw (pre-squash) torch tensors
        self.log_probs  = []   # float
        self.rewards    = []   # float
        self.values     = []   # float
        self.dones      = []   # bool

    def store(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self, last_value, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
        """GAE-λ advantage estimation."""
        advantages = []
        gae = 0.0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns    = advantages + torch.tensor(self.values, dtype=torch.float32)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def get_tensors(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.tensor(self.log_probs, dtype=torch.float32),
        )


# ── PPO Agent ─────────────────────────────────────────────
class PPOAgent:
    """
    Manages Actor + Critic, the rollout buffer, and the PPO update.

    External usage
    ──────────────
    agent = PPOAgent()
    obs_tensor = make_obs(z, state)          # shape (1, INPUT_DIM)
    action_np, log_prob, value = agent.select_action(obs_tensor)
    agent.buffer.store(...)
    if len(agent.buffer.rewards) >= PPO_STEPS_PER_UPDATE:
        agent.update(last_obs_tensor)
    """

    def __init__(self, latent_dim: Optional[int] = None):
        ld = int(PPO_LATENT_DIM if latent_dim is None else latent_dim)
        self.latent_dim = ld
        self.actor = Actor(ld).to(DEVICE)
        self.critic = Critic(ld).to(DEVICE)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=PPO_LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=PPO_LR_CRITIC)
        self.buffer = RolloutBuffer()

    # ── obs construction ──────────────────────────────────
    @staticmethod
    def make_obs(z: torch.Tensor, state: np.ndarray, lidar: np.ndarray) -> torch.Tensor:
        """
        Combine VAE latent z (1, latent_dim), numpy state (PPO_STATE_DIM,), and lidar (N, 3)
        into a single observation tensor on DEVICE.
        """
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        l = torch.tensor(lidar.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return torch.cat([z, s, l], dim=1)   # (1, FLAT_OBS_DIM)

    # ── step ──────────────────────────────────────────────
    def select_action(self, obs: torch.Tensor, deterministic=False):
        """
        Returns
        -------
        action    : np.ndarray  (3,)
        raw       : torch.Tensor (1, 3)  pre-squash for buffer
        log_prob  : float
        value     : float
        """
        with torch.no_grad():
            dist  = self.actor(obs)
            raw   = dist.mean if deterministic else dist.sample()
            log_p = squashed_log_prob(dist, raw).item()
            val   = self.critic(obs).item()

        # All actions use tanh: steer ∈ [-1,1], accel ∈ [-1,1]
        action = torch.tanh(raw).squeeze(0).cpu().numpy()

        return action, raw.squeeze(0), log_p, val

    # ── PPO update ────────────────────────────────────────
    def update(self, last_obs: torch.Tensor):
        with torch.no_grad():
            last_value = self.critic(last_obs).item()

        advantages, returns = self.buffer.compute_advantages(last_value)
        obs_t, act_t, old_lp_t = self.buffer.get_tensors()

        obs_t    = obs_t.to(DEVICE)
        act_t    = act_t.to(DEVICE)
        old_lp_t = old_lp_t.to(DEVICE)
        adv_t    = advantages.to(DEVICE)
        ret_t    = returns.to(DEVICE)

        n = len(ret_t)

        for _ in range(PPO_EPOCHS_PER_UPDATE):
            indices = torch.randperm(n)
            for start in range(0, n, PPO_MINIBATCH_SIZE):
                idx = indices[start: start + PPO_MINIBATCH_SIZE]

                # actor
                new_logp, entropy = self.actor.evaluate(obs_t[idx], act_t[idx])
                ratio  = (new_logp - old_lp_t[idx]).exp()
                clip   = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS)
                actor_loss = -torch.min(ratio * adv_t[idx], clip * adv_t[idx]).mean()
                actor_loss -= PPO_ENTROPY_COEF * entropy.mean()

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), PPO_MAX_GRAD_NORM)
                self.opt_actor.step()

                # critic
                value_pred  = self.critic(obs_t[idx])
                critic_loss = PPO_VALUE_COEF * nn.functional.mse_loss(value_pred, ret_t[idx])

                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), PPO_MAX_GRAD_NORM)
                self.opt_critic.step()

        self.buffer.reset()

    # ── checkpointing ─────────────────────────────────────
    def save(self, path: str, run_id=None):
        data = {
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        if run_id is not None:
            data["run_id"] = run_id
        torch.save(data, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return False, None
        try:
            ckpt = torch.load(path, map_location=DEVICE)
            self.actor.load_state_dict(ckpt["actor"], strict=True)
            self.critic.load_state_dict(ckpt["critic"], strict=True)
            print(f"PPO checkpoint loaded: {path}")
            return True, ckpt.get("run_id")
        except Exception as e:
            print(
                f"PPO checkpoint skipped (incompatible or corrupt): {e}\n"
                f"  Training from scratch. Remove old file or match architecture "
                f"(FLAT_OBS_DIM={flat_obs_dim(self.latent_dim)})."
            )
            return False, None

