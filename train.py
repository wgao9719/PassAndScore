import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import PassAndScoreEnv

# -------------- Factorized Actor (two Categorical heads) --------------
class FactorizedPolicy(nn.Module):
    """
    Centralized observations -> shared torso -> two softmax heads (A, B).
    Each head outputs logits over {Stay, Up, Down, Left, Right}.
    """
    def __init__(self, obs_dim, n_each=5, hidden=128):
        super().__init__()
        self.n_each = n_each
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.headA = nn.Linear(hidden, n_each)
        self.headB = nn.Linear(hidden, n_each)

    def forward(self, x):
        h = self.body(x)
        return self.headA(h), self.headB(h)

    @torch.no_grad()
    def act(self, obs):
        """
        Returns: (aA, aB), logp_sum
        """
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logitsA, logitsB = self.forward(x)
        distA = torch.distributions.Categorical(logits=logitsA)
        distB = torch.distributions.Categorical(logits=logitsB)
        aA = distA.sample()
        aB = distB.sample()
        logp = distA.log_prob(aA) + distB.log_prob(aB)
        return (int(aA.item()), int(aB.item())), float(logp.item())

# -------------- Critic --------------
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

CKPT_PATH = "policy.pth"

def atomic_torch_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX & Windows 10+

# ----- Generalized Advantage Estimation -----
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T] value(s_t)
    dones:   [T] bools (True if episode ended at t)
    last_value: V(s_{T}) for bootstrapping (0 if done)
    returns (advantages[T], targets[T]) where targets are V-train targets
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    targets = adv + values
    return adv, targets

# ---------------- Side-based progress shaper ----------------
import math

class SideProgressShaper:
    """
    Side-based, progress-only shaping.

    - If the ball is on the LEFT side (Region A side), reward progress toward
      the *center of Region B* with a saturation radius (pass goal).
      Uses episode-capped potential so moving away from center early on
      doesn't penalize exploration.

    - If the ball is on the RIGHT side (Region B side), reward progress toward
      the *goalmouth segment* (shoot goal) using distance-to-goal WITHOUT the
      initial-distance cap. Still progress-only (no penalty for regress).

    The side boundary is the midpoint between Region A's max-x and Region B's min-x.
    """
    def __init__(self, gamma=0.99, w_goal=1.0, w_pass=1.0, progress_only=True, pass_radius=0.5):
        self.gamma = float(gamma)
        self.w_goal = float(w_goal)
        self.w_pass = float(w_pass)
        self.progress_only = bool(progress_only)
        self.pass_radius = float(pass_radius)

        # Episode caches
        self.center_B = None     # (cx, cy)
        self.split_x  = None     # side boundary
        self.d0_pass  = None     # initial saturated distance to Region B center (for pass side)

    def reset(self, env):
        # Region B center
        cx = 0.5 * (env.region_b.xmin + env.region_b.xmax)
        cy = 0.5 * (env.region_b.ymin + env.region_b.ymax)
        self.center_B = (float(cx), float(cy))

        # Side boundary (between regions, includes "surrounding" notion)
        self.split_x = 0.5 * (env.region_a.xmax + env.region_b.xmin)

        # Initial pass distance (saturated)
        pball = env.state["pball"]
        self.d0_pass = self._sat_pass_distance(float(pball[0]), float(pball[1]))

    def _sat_pass_distance(self, bx: float, by: float) -> float:
        """Saturated distance to Region B center: max(0, ||ball - center_B|| - pass_radius)."""
        dx = bx - self.center_B[0]
        dy = by - self.center_B[1]
        return max(0.0, math.hypot(dx, dy) - self.pass_radius)

    def _progress_delta(self, phi_prev: float, phi_curr: float) -> float:
        return (self.gamma * phi_curr) - phi_prev

    def _clip_if_progress_only(self, delta: float) -> float:
        return max(0.0, delta) if self.progress_only else delta

    def step(self, info: dict) -> float:
        if not (isinstance(info, dict) and "before" in info and "after" in info):
            return 0.0

        before, after = info["before"], info["after"]
        bx_prev, by_prev = float(before.get("ball_x", 0.0)), float(before.get("ball_y", 0.0))
        bx_curr, by_curr = float(after.get("ball_x",  0.0)), float(after.get("ball_y",  0.0))

        # Decide side by CURRENT ball position
        on_left_side = (bx_curr < self.split_x)

        if on_left_side:
            # ---------- PASS SIDE (episode-capped, progress-only) ----------
            d_prev = self._sat_pass_distance(bx_prev, by_prev)
            d_curr = self._sat_pass_distance(bx_curr, by_curr)
            # capped potential: phi = max(0, d0_pass - d)
            phi_prev = max(0.0, self.d0_pass - d_prev)
            phi_curr = max(0.0, self.d0_pass - d_curr)
            delta = self._progress_delta(phi_prev, phi_curr)
            return self.w_pass * self._clip_if_progress_only(delta)
        else:
            # ---------- SHOOT SIDE (NO initial cap, progress-only) ----------
            d_prev = float(before.get("dist_ball_to_goal", 0.0))
            d_curr = float(after.get("dist_ball_to_goal",  0.0))
            # use phi = -d (no cap); progress if distance decreases
            phi_prev = -d_prev
            phi_curr = -d_curr
            delta = self._progress_delta(phi_prev, phi_curr)   # = d_prev - gamma*d_curr
            return self.w_goal * self._clip_if_progress_only(delta)

def train(
    seed=0,
    episodes_per_update=10,
    updates=200,
    max_steps=400,
    gamma=0.99,
    lam=0.95,               # GAE(λ)
    lr_actor=3e-4,
    lr_critic=1e-3,
    entropy_coef=0.01,
    value_coef=0.5,         # (kept for compatibility; we use separate optims)
    max_grad_norm=0.5,
    hidden=128,
    device="cpu",
    # PPO-specific
    clip_ratio=0.2,
    ppo_epochs=4,
    minibatch_size=2048,
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = PassAndScoreEnv(centralized=True, seed=seed)

    obs_dim = 12
    n_each = 5  # per-agent action size

    actor  = FactorizedPolicy(obs_dim, n_each=n_each, hidden=hidden).to(device)
    critic = ValueNet(obs_dim, hidden=hidden).to(device)
    opt_actor  = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    # Load from existing checkpoint if available (must match architecture)
    if os.path.isfile(CKPT_PATH):
        print(f"Loading checkpoint from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        actor.load_state_dict(ckpt["actor"], strict=True)
        critic.load_state_dict(ckpt["critic"], strict=True)
        print("Checkpoint loaded.")

    for it in range(1, updates + 1):
        # ======= Collect trajectories (on-policy) =======
        obs_buf, actA_buf, actB_buf = [], [], []
        logp_buf, val_tgt_buf, adv_buf = [], [], []
        ep_returns = []

        for _ in range(episodes_per_update):
            obs, _ = env.reset()
            shaper = SideProgressShaper(gamma=gamma, w_goal=1.0, w_pass=1.0,
                                        progress_only=True, pass_radius=0.5)
            shaper.reset(env)

            ep_rews, ep_obs, ep_aA, ep_aB, ep_vals, ep_dones, ep_logp = [], [], [], [], [], [], []
            done, steps = False, 0

            while not done and steps < max_steps:
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logitsA, logitsB = actor(x)
                    distA = torch.distributions.Categorical(logits=logitsA)
                    distB = torch.distributions.Categorical(logits=logitsB)
                    aA = int(distA.sample().item())
                    aB = int(distB.sample().item())
                    logp = float(distA.log_prob(torch.tensor(aA, device=device)) +
                                 distB.log_prob(torch.tensor(aB, device=device)))
                    value = float(critic(x).item())

                nxt, r, term, trunc, info = env.step((aA, aB))
                r += shaper.step(info)

                ep_obs.append(obs)
                ep_aA.append(aA)
                ep_aB.append(aB)
                ep_rews.append(float(r))
                ep_vals.append(value)
                ep_dones.append(bool(term or trunc))
                ep_logp.append(logp)

                obs = nxt
                steps += 1
                done = term or trunc

            # Bootstrap value for final state
            if ep_dones and ep_dones[-1]:
                last_v = 0.0
            else:
                x_last = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    last_v = float(critic(x_last).item())

            # GAE for this episode
            adv, tgt = compute_gae(
                rewards=np.asarray(ep_rews, dtype=np.float32),
                values=np.asarray(ep_vals, dtype=np.float32),
                dones=np.asarray(ep_dones, dtype=np.bool_),
                last_value=last_v,
                gamma=gamma,
                lam=lam,
            )

            obs_buf.extend(ep_obs)
            actA_buf.extend(ep_aA)
            actB_buf.extend(ep_aB)
            logp_buf.extend(ep_logp)
            val_tgt_buf.extend(tgt.tolist())   # critic targets
            adv_buf.extend(adv.tolist())       # store GAE advantages

            ep_returns.append(float(np.sum(ep_rews)))

        # ======= Prepare tensors =======
        O        = torch.as_tensor(np.array(obs_buf, dtype=np.float32), device=device)
        A_A      = torch.as_tensor(np.array(actA_buf, dtype=np.int64), device=device)
        A_B      = torch.as_tensor(np.array(actB_buf, dtype=np.int64), device=device)
        LOGP_OLD = torch.as_tensor(np.array(logp_buf, dtype=np.float32), device=device)
        Vt       = torch.as_tensor(np.array(val_tgt_buf, dtype=np.float32), device=device)
        Adv      = torch.as_tensor(np.array(adv_buf, dtype=np.float32), device=device)

        # Normalize advantages
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)

        # ======= PPO updates =======
        N = O.shape[0]
        for _ in range(ppo_epochs):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, minibatch_size):
                idx = perm[i:i+minibatch_size]

                logitsA, logitsB = actor(O[idx])
                distA = torch.distributions.Categorical(logits=logitsA)
                distB = torch.distributions.Categorical(logits=logitsB)

                logp = distA.log_prob(A_A[idx]) + distB.log_prob(A_B[idx])
                entropy = (distA.entropy() + distB.entropy()).mean()

                ratio = torch.exp(logp - LOGP_OLD[idx])
                surr1 = ratio * Adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * Adv[idx]
                actor_loss = -(torch.min(surr1, surr2)).mean() - entropy_coef * entropy

                # Critic update on the same minibatch
                V = critic(O[idx])
                critic_loss = torch.mean((V - Vt[idx]) ** 2)

                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt_actor.step()

                opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                opt_critic.step()

        # ======= Logging (approximate, computed post-update on full batch) =======
        with torch.no_grad():
            logitsA_all, logitsB_all = actor(O)
            distA_all = torch.distributions.Categorical(logits=logitsA_all)
            distB_all = torch.distributions.Categorical(logits=logitsB_all)
            entropy_all = (distA_all.entropy() + distB_all.entropy()).mean().item()
            V_all = critic(O)
            critic_loss_full = torch.mean((V_all - Vt) ** 2).item()

        print(f"[{it:04d}] avg_return={np.mean(ep_returns):.3f} "
              f"entropy={entropy_all:.3f} "
              f"critic_loss={critic_loss_full:.3f}")

        if it % 100 == 0:
            atomic_torch_save({"actor": actor.state_dict(),
                               "critic": critic.state_dict(),
                               "obs_dim": obs_dim,
                               "n_each": n_each}, CKPT_PATH)

    print("Training complete.")
    torch.save({"actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "obs_dim": obs_dim,
                "n_each": n_each}, 'policy.pth')

if __name__ == "__main__":
    train(
        seed=0,
        episodes_per_update=32,
        updates=4000,
        max_steps=1000,
        gamma=0.99,
        lam=0.95,
        lr_actor=1e-4,
        lr_critic=1e-3,
        entropy_coef=0.01,
        value_coef=0.5,
        hidden=128,
        device="cpu",
        clip_ratio=0.2,
        ppo_epochs=4,
        minibatch_size=2048,
    )
