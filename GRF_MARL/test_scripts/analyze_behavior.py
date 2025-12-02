#!/usr/bin/env python
"""
Behavioral DNA Analysis

Analyzes action distributions per strategy to identify behavioral specialization.

Action groups in GRF:
  - 0: Idle
  - 1-8: Movement (directions)
  - 9-11: Passing (low, high, short)
  - 12: Shooting
  - 13-15: Dribbling/Trapping
  - 16-18: Sprint/Slide

Usage:
    python analyze_behavior.py --checkpoint ./logs/.../epoch_300
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import Counter, defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gfootball.env as football_env
from light_malib.model.gr_football.passandscore_factorized import FiLMGenerator


# Action groupings
ACTION_GROUPS = {
    "Idle": [0],
    "Move": list(range(1, 9)),
    "Pass": [9, 10, 11],
    "Shoot": [12],
    "Dribble": [13, 14, 15],
    "Sprint/Slide": [16, 17, 18]
}


def create_env():
    """Create GRF environment."""
    env = football_env.create_environment(
        env_name="5_vs_5",
        stacked=False,
        representation="simple115v2",
        rewards="scoring,checkpoints",
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=0,
        logdir="",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
    )
    return env


def load_policy(checkpoint_dir):
    """Load policy from checkpoint."""
    actor_path = os.path.join(checkpoint_dir, "actor.pt")
    
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"No actor.pt in {checkpoint_dir}")
    
    actor_state = torch.load(actor_path, map_location="cpu")
    
    # Infer dimensions
    input_dim = 133
    hidden_dim = 128
    action_dim = 19
    num_strategies = 8
    
    for k, v in actor_state.items():
        if 'body.0.weight' in k:
            input_dim = v.shape[1]
            hidden_dim = v.shape[0]
        if 'head.weight' in k:
            action_dim = v.shape[0]
        if 'film_generator.strategy_embedding.weight' in k:
            num_strategies = v.shape[0]
    
    # Create FiLM
    film = FiLMGenerator(num_strategies=num_strategies, hidden_dim=hidden_dim)
    film_state = {k.replace('film_generator.', ''): v 
                  for k, v in actor_state.items() if 'film_generator' in k}
    film.load_state_dict(film_state)
    
    # Create body MLP
    body = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
    )
    body_state = {k.replace('body.', ''): v for k, v in actor_state.items() if k.startswith('body.')}
    body.load_state_dict(body_state)
    
    # Create action head
    head = torch.nn.Linear(hidden_dim, action_dim)
    head_state = {k.replace('head.', ''): v for k, v in actor_state.items() if k.startswith('head.')}
    head.load_state_dict(head_state)
    
    class PolicyWrapper:
        def __init__(self, body, film, head, input_dim):
            self.body = body.eval()
            self.film = film.eval()
            self.head = head.eval()
            self.input_dim = input_dim
        
        def get_action(self, obs, strategy_code):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                num_agents = obs_t.shape[0]
                obs_dim = obs_t.shape[1]
                
                # Pad to match input_dim
                if self.input_dim > obs_dim:
                    pooled = obs_t.mean(dim=0, keepdim=True).expand(num_agents, -1)
                    extra_dim = self.input_dim - obs_dim
                    extra = pooled[:, :extra_dim] if extra_dim <= obs_dim else torch.zeros(num_agents, extra_dim)
                    obs_t = torch.cat([obs_t, extra], dim=1)
                
                strat_t = torch.full((num_agents,), strategy_code, dtype=torch.long)
                
                x = self.body(obs_t)
                x = self.film.modulate(strategy_code=strat_t, features=x)
                logits = self.head(x)
                
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                return actions.cpu().numpy()
    
    return PolicyWrapper(body, film, head, input_dim)


def get_action_group(action):
    """Get the group name for an action."""
    for group, indices in ACTION_GROUPS.items():
        if action in indices:
            return group
    return "Unknown"


def compute_energy_proxy(action):
    """Compute energy proxy based on action type."""
    if action in ACTION_GROUPS["Sprint/Slide"]:
        return 1.0
    elif action in ACTION_GROUPS["Move"]:
        return 0.5
    elif action in ACTION_GROUPS["Shoot"]:
        return 0.8
    elif action in ACTION_GROUPS["Dribble"]:
        return 0.6
    elif action in ACTION_GROUPS["Pass"]:
        return 0.3
    else:  # Idle
        return 0.0


def analyze_strategy(env, policy, strategy_id, num_episodes=3, max_steps=300):
    """Analyze behavior for a single strategy."""
    action_counts = Counter()
    energy_values = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        for step in range(max_steps):
            # Get actions for all agents
            actions = policy.get_action(obs, strategy_id)
            
            # Record action DNA for each agent
            for action in actions:
                action = int(action)
                group = get_action_group(action)
                action_counts[group] += 1
                energy_values.append(compute_energy_proxy(action))
            
            # Step environment
            obs, _, done, _ = env.step(actions.tolist())
            
            if isinstance(done, (list, np.ndarray)):
                if all(done):
                    break
            elif done:
                break
    
    return action_counts, energy_values


def print_strategy_profile(strategy_id, action_counts, energy_values):
    """Print behavioral profile for a strategy."""
    total = sum(action_counts.values())
    if total == 0:
        print(f"\nStrategy {strategy_id}: No data collected")
        return {}
    
    avg_energy = np.mean(energy_values) if energy_values else 0
    
    # Build profile
    profile = {}
    for group in ACTION_GROUPS.keys():
        pct = (action_counts.get(group, 0) / total) * 100
        profile[group] = pct
    
    # Determine dominant trait
    dominant = max(profile, key=profile.get)
    
    # Energy-based classification
    if avg_energy > 0.7:
        energy_class = "🔥 High Energy"
    elif avg_energy > 0.4:
        energy_class = "⚡ Medium Energy"
    else:
        energy_class = "🧊 Low Energy"
    
    print(f"\n┌─────────────────────────────────────────────────────")
    print(f"│ Strategy {strategy_id}: {energy_class} | Dominant: {dominant}")
    print(f"├─────────────────────────────────────────────────────")
    print(f"│ Energy Score: {avg_energy:.3f}")
    print(f"│ Action Distribution:")
    
    # Print bar chart
    for group, pct in sorted(profile.items(), key=lambda x: -x[1]):
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"│   {group:12s} [{bar}] {pct:5.1f}%")
    
    print(f"└─────────────────────────────────────────────────────")
    
    return {"energy": avg_energy, "dominant": dominant, "profile": profile}


def diagnose_strategies(all_profiles):
    """Provide overall diagnosis of strategy diversity."""
    print("\n" + "="*60)
    print("BEHAVIORAL DIAGNOSIS")
    print("="*60)
    
    # Check for mode collapse (all strategies have same dominant action)
    dominants = [p["dominant"] for p in all_profiles.values()]
    unique_dominants = set(dominants)
    
    if len(unique_dominants) == 1:
        print("🔴 MODE COLLAPSE: All strategies have same dominant action!")
        print(f"   All strategies dominated by: {dominants[0]}")
    elif len(unique_dominants) <= 2:
        print("🟡 LOW DIVERSITY: Only 2 distinct dominant actions")
        print(f"   Dominants: {unique_dominants}")
    else:
        print("🟢 GOOD DIVERSITY: Multiple distinct dominant actions")
        print(f"   Dominants: {unique_dominants}")
    
    # Check energy spread
    energies = [p["energy"] for p in all_profiles.values()]
    energy_spread = max(energies) - min(energies)
    
    print(f"\nEnergy Spread: {energy_spread:.3f}")
    if energy_spread < 0.1:
        print("🔴 All strategies have similar energy levels")
    elif energy_spread < 0.2:
        print("🟡 Moderate energy variation")
    else:
        print("🟢 Good energy variation across strategies")
    
    # Identify specialized strategies
    print("\nStrategy Specializations:")
    for strat_id, prof in all_profiles.items():
        profile = prof["profile"]
        specializations = []
        
        if profile.get("Shoot", 0) > 10:
            specializations.append("🎯 Attacker")
        if profile.get("Pass", 0) > 25:
            specializations.append("🎭 Playmaker")
        if profile.get("Sprint/Slide", 0) > 15:
            specializations.append("🏃 Presser")
        if profile.get("Idle", 0) > 15:
            specializations.append("🛡️ Holder")
        if profile.get("Move", 0) > 60:
            specializations.append("🔄 Mover")
        if profile.get("Dribble", 0) > 15:
            specializations.append("⚽ Dribbler")
        
        if specializations:
            print(f"  Strategy {strat_id}: {', '.join(specializations)}")
        else:
            print(f"  Strategy {strat_id}: 🔲 Generic")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--num_strategies", type=int, default=8)
    args = parser.parse_args()
    
    print("="*60)
    print("BEHAVIORAL DNA ANALYSIS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes per strategy: {args.episodes}")
    
    # Create env and load policy
    print("\n📦 Creating environment...")
    env = create_env()
    
    print(f"\n🔧 Loading policy...")
    policy = load_policy(args.checkpoint)
    print("✅ Policy loaded")
    
    # Analyze each strategy
    all_profiles = {}
    
    for strat_id in range(args.num_strategies):
        print(f"\n🎮 Analyzing Strategy {strat_id}...")
        action_counts, energy_values = analyze_strategy(
            env, policy, strat_id, 
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )
        profile = print_strategy_profile(strat_id, action_counts, energy_values)
        all_profiles[strat_id] = profile
    
    env.close()
    
    # Overall diagnosis
    diagnose_strategies(all_profiles)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

