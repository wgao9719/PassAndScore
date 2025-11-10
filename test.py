#!/usr/bin/env python3
import argparse, os, numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from env import PassAndScoreEnv
from train import FactorizedPolicy, ValueNet  # new factorized actor + critic

# ---------- Load actor & critic from PPO/factorized checkpoint ----------
def load_actor_critic(ckpt_path, hidden=128, device="cpu"):
    """
    Expects a checkpoint saved by the new PPO + factorized-policy trainer, with keys:
      {"actor": state_dict, "critic": state_dict, "obs_dim": int, "n_each": int}
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if not (isinstance(ckpt, dict) and "actor" in ckpt and "critic" in ckpt):
        raise ValueError("Checkpoint must contain 'actor' and 'critic' state dicts.")

    obs_dim = int(ckpt["obs_dim"])
    n_each  = int(ckpt["n_each"])

    actor  = FactorizedPolicy(obs_dim, n_each=n_each, hidden=hidden).to(device)
    critic = ValueNet(obs_dim, hidden=hidden).to(device)
    actor.load_state_dict(ckpt["actor"], strict=True)
    critic.load_state_dict(ckpt["critic"], strict=True)
    actor.eval(); critic.eval()
    return actor, critic, obs_dim, n_each

@torch.no_grad()
def run_one_episode(actor, critic, seed=0, render_fps=30, max_steps=400, device="cpu",
                    stochastic=True, print_prefix="[eval]"):
    env = PassAndScoreEnv(centralized=True, seed=seed)
    obs, _ = env.reset()

    plt.ion()
    dt = 1.0 / render_fps
    ret = 0.0

    # Create the figure once and watch for close events
    env.render()                     # open the window once
    fig = plt.gcf()
    closed = False

    def _on_close(evt):
        nonlocal closed
        closed = True

    cid = fig.canvas.mpl_connect('close_event', _on_close)

    for t in range(max_steps):
        # If the user closed the window, stop BEFORE rendering again (so it won't reopen)
        if closed or not plt.fignum_exists(fig.number):
            print("\nWindow closed, exiting...")
            break

        # Compute value BEFORE acting (V(s_t))
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        V = float(critic(x).item())

        # Update render and overlay V(s)
        env.render()  # update existing window
        try:
            ax = fig.axes[0] if fig.axes else None
            if ax is not None:
                title = ax.get_title()
                ax.set_title(f"{title}   |   V(s)={V:.3f}" if title else f"V(s)={V:.3f}")
        except Exception:
            pass
        plt.pause(dt)  # non-blocking GUI update

        # Act with factorized policy (two Categorical heads)
        logitsA, logitsB = actor(x)
        if stochastic:
            distA = torch.distributions.Categorical(logits=logitsA)
            distB = torch.distributions.Categorical(logits=logitsB)
            aA = int(distA.sample().item())
            aB = int(distB.sample().item())
        else:
            aA = int(torch.argmax(logitsA, dim=-1).item())
            aB = int(torch.argmax(logitsB, dim=-1).item())

        # Step
        obs, r, term, trunc, info = env.step((aA, aB))
        ret += r
        print(f"\r{print_prefix} step={t+1}/{max_steps}  reward={r:.3f}  return={ret:.3f}  V={V:.3f}",
              end="", flush=True)
        if term or trunc:
            break

    # Cleanup
    plt.ioff()
    try:
        fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass
    try:
        plt.close(fig)
    except Exception:
        pass
    env.close()

    if not (closed or term or trunc):
        # Only print final return if we finished naturally
        print(f"\n{print_prefix} return={ret:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to PPO checkpoint (policy.pth)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None, help="Episode seed (default: random)")
    p.add_argument("--fps", type=int, default=30, help="Render FPS")
    p.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    p.add_argument("--deterministic", action="store_true", help="Greedy actions at eval time")
    p.add_argument("--hidden", type=int, default=128, help="Hidden size (must match training)")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    actor, critic, _, _ = load_actor_critic(
        args.ckpt, hidden=args.hidden, device=args.device
    )
    seed = args.seed if args.seed is not None else int(np.random.randint(0, 1_000_000))
    run_one_episode(
        actor, critic,
        seed=seed,
        render_fps=args.fps,
        max_steps=args.max_steps,
        device=args.device,
        stochastic=(not args.deterministic),
        print_prefix=f"[eval seed={seed}]",
    )

if __name__ == "__main__":
    main()
