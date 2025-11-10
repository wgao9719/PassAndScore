#!/usr/bin/env python3
import argparse, os, numpy as np, torch
from env import PassAndScoreEnv
from train import FactorizedPolicy, ValueNet  # uses your PPO factorized policy

@torch.no_grad()
def load_actor(ckpt_path, hidden=128, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    n_each  = int(ckpt["n_each"])
    actor = FactorizedPolicy(obs_dim, n_each=n_each, hidden=hidden).to(device)
    actor.load_state_dict(ckpt["actor"], strict=True)
    actor.eval()
    return actor, obs_dim, n_each

@torch.no_grad()
def run_episode(env, actor, max_steps=1000, device="cpu", deterministic=False):
    obs, _ = env.reset()
    steps = 0
    scored_goal = False
    left_regions = False
    truncated = False

    while steps < max_steps:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logitsA, logitsB = actor(x)
        if deterministic:
            aA = int(torch.argmax(logitsA, dim=-1).item())
            aB = int(torch.argmax(logitsB, dim=-1).item())
        else:
            distA = torch.distributions.Categorical(logits=logitsA)
            distB = torch.distributions.Categorical(logits=logitsB)
            aA = int(distA.sample().item())
            aB = int(distB.sample().item())

        obs, r, term, trunc, info = env.step((aA, aB))
        steps += 1

        after = info.get("after", {})
        # These flags are provided by the env patch above
        if after.get("scored_goal", False):
            scored_goal = True
        if after.get("left_regions", False):
            left_regions = True

        if term or trunc:
            truncated = bool(trunc and not term)
            break

    return {
        "scored_goal": scored_goal,
        "left_regions": left_regions,
        "truncated": truncated,
        "steps": steps,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (policy.pth)")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=128, help="Hidden size used in training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Greedy eval (default: stochastic)")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    actor, obs_dim, n_each = load_actor(args.ckpt, hidden=args.hidden, device=args.device)

    goals = 0
    leaves = 0
    truncs = 0
    steps_to_goal = []

    for ep in range(args.episodes):
        # use a different seed per episode for variety
        env = PassAndScoreEnv(centralized=True, seed=(args.seed + ep))
        result = run_episode(env, actor,
                             max_steps=args.max_steps,
                             device=args.device,
                             deterministic=args.deterministic)
        env.close()

        goals += 1 if result["scored_goal"] else 0
        leaves += 1 if result["left_regions"] else 0
        truncs += 1 if result["truncated"] else 0
        if result["scored_goal"]:
            steps_to_goal.append(result["steps"])

    n = float(args.episodes)
    avg_goals = goals / n
    avg_leaves = leaves / n
    avg_truncs = truncs / n
    avg_steps_to_goal = (float(np.mean(steps_to_goal)) if steps_to_goal else float("nan"))

    print("\n=== EVAL SUMMARY ===")
    print(f"Episodes:              {int(n)}")
    print(f"Avg # goals/episode:   {avg_goals:.3f}")
    print(f"Avg # leaves/episode:  {avg_leaves:.3f}")
    print(f"Avg # truncs/episode:  {avg_truncs:.3f}")
    if steps_to_goal:
        print(f"Avg steps-to-goal:     {avg_steps_to_goal:.1f}  (over {len(steps_to_goal)} scoring eps)")
    else:
        print("Avg steps-to-goal:     N/A (no scoring episodes)")

if __name__ == "__main__":
    main()
