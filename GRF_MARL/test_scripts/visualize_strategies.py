"""
Strategy Heatmap Visualization

Diagnose Phase 1 training quality:
- "Corner Camper" (Bad): Tight dots in corners - hiding from game
- "Blob" (Bad): All strategies identical - mode collapse  
- "Playstyle" (Good): Different spatial distributions per strategy

Usage:
    python visualize_strategies.py --checkpoint /path/to/phase1/checkpoint
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gfootball.env as football_env


def create_env():
    """Create GRF environment matching training config."""
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
        extra_players=None,
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
    )
    return env


def load_policy(checkpoint_dir, device="cpu"):
    """Load Phase 1 policy directly from state dicts with minimal model reconstruction."""
    from light_malib.model.gr_football.passandscore_factorized import FiLMGenerator
    
    # Load state dicts
    actor_path = os.path.join(checkpoint_dir, "actor.pt")
    
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"No actor.pt in {checkpoint_dir}")
    
    actor_state = torch.load(actor_path, map_location="cpu")
    print(f"✅ Loaded actor state ({len(actor_state)} params)")
    
    # Print keys for debugging
    print(f"   Keys: {list(actor_state.keys())}")
    
    # Infer dimensions from state dict
    # Actor structure: body.0/2 (MLP), film_generator.*, head (action output)
    input_dim = 133  # obs + pooled features
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
    
    print(f"   Inferred: input={input_dim}, hidden={hidden_dim}, actions={action_dim}, strategies={num_strategies}")
    
    # Create FiLM generator
    film = FiLMGenerator(num_strategies=num_strategies, hidden_dim=hidden_dim)
    film_state = {k.replace('film_generator.', ''): v 
                  for k, v in actor_state.items() if 'film_generator' in k}
    film.load_state_dict(film_state)
    print(f"✅ Loaded FiLM generator")
    
    # Create body MLP (matches actor_state keys: body.0.*, body.2.*)
    body = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
    )
    body_state = {k.replace('body.', ''): v for k, v in actor_state.items() if k.startswith('body.')}
    body.load_state_dict(body_state)
    print(f"✅ Loaded body MLP")
    
    # Create action head
    head = torch.nn.Linear(hidden_dim, action_dim)
    head_state = {k.replace('head.', ''): v for k, v in actor_state.items() if k.startswith('head.')}
    head.load_state_dict(head_state)
    print(f"✅ Loaded action head")
    
    class PolicyWrapper:
        def __init__(self, body, film, head, input_dim):
            self.body = body.eval()
            self.film = film.eval()
            self.head = head.eval()
            self.input_dim = input_dim
            self.device = torch.device("cpu")
        
        def get_action(self, obs, strategy_code):
            """Get actions for batch of observations with fixed strategy.
            
            Args:
                obs: numpy array of shape [num_agents, obs_dim] or list of obs
                strategy_code: integer strategy code
            Returns:
                actions: numpy array of shape [num_agents]
            """
            with torch.no_grad():
                # Convert obs to tensor - shape should be [num_agents, obs_dim]
                if isinstance(obs, list):
                    obs_t = torch.tensor(np.array(obs), dtype=torch.float32)
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                
                num_agents = obs_t.shape[0]
                obs_dim = obs_t.shape[1]
                
                # The actor expects 133 features = obs (115) + pooled obs (18 extra)
                # For factorized actor: concatenate local obs with mean-pooled obs
                pooled = obs_t.mean(dim=0, keepdim=True).expand(num_agents, -1)
                
                # Concatenate local + pooled (or just pad if dimensions don't match)
                if self.input_dim > obs_dim:
                    # Add pooled features
                    extra_dim = self.input_dim - obs_dim
                    if extra_dim <= obs_dim:
                        # Use first extra_dim features from pooled
                        extra = pooled[:, :extra_dim]
                    else:
                        # Pad with zeros
                        extra = torch.zeros(num_agents, extra_dim)
                    obs_t = torch.cat([obs_t, extra], dim=1)
                elif obs_t.shape[1] > self.input_dim:
                    obs_t = obs_t[:, :self.input_dim]
                
                strat_t = torch.full((num_agents,), strategy_code, dtype=torch.long)
                
                # Forward: obs -> body MLP -> FiLM modulate -> head
                x = self.body(obs_t)
                x = self.film.modulate(strategy_code=strat_t, features=x)
                logits = self.head(x)
                
                # Sample actions
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                return actions.cpu().numpy()
    
    return PolicyWrapper(body, film, head, input_dim)


def get_active_player_positions(obs, num_agents=4):
    """
    Extract positions from observation.
    
    GRF simple115v2 layout:
    - indices 0-21: left team positions (11 players * 2)
    - indices 88-89: ball position
    
    We track controlled players (first 4 of left team).
    """
    positions = []
    
    # Left team positions: indices 0-21, reshape to [11, 2]
    if len(obs) >= 22:
        left_team = obs[:22].reshape(11, 2)
        # Our controlled players are typically indices 0-3
        for i in range(min(num_agents, 4)):
            positions.append(left_team[i])
    
    return positions


def get_ball_position(obs):
    """Get ball position from observation."""
    if len(obs) >= 90:
        return obs[88:90]
    return np.array([0, 0])


def run_episodes_with_strategy(env, policy, strategy_id, num_episodes=5, max_steps=500):
    """
    Run episodes with a fixed strategy and collect positions.
    
    Returns:
        positions: list of (x, y) tuples for active players
        ball_positions: list of (x, y) tuples for ball
    """
    positions = []
    ball_positions = []
    
    for ep in range(num_episodes):
        obs_list = env.reset()
        
        for step in range(max_steps):
            # Collect positions from first agent's observation
            # obs_list shape: [num_agents, obs_dim]
            if isinstance(obs_list, np.ndarray) and obs_list.ndim == 2:
                raw_obs = obs_list[0]  # First agent's view
            elif isinstance(obs_list, list):
                raw_obs = obs_list[0]
            else:
                raw_obs = obs_list
            
            # Get player positions
            player_pos = get_active_player_positions(raw_obs)
            positions.extend(player_pos)
            
            # Get ball position
            ball_pos = get_ball_position(raw_obs)
            ball_positions.append(ball_pos)
            
            # Get actions from policy with fixed strategy
            # obs_list is [num_agents, obs_dim] numpy array
            actions = policy.get_action(obs_list, strategy_id)
            
            # Step environment (GRF expects list of ints)
            obs_list, rewards, dones, infos = env.step(actions.tolist())
            
            if isinstance(dones, (list, np.ndarray)):
                if all(dones):
                    break
            elif dones:
                break
    
    return np.array(positions), np.array(ball_positions)


def plot_heatmaps(all_positions, all_ball_positions, output_path="strategy_heatmaps.png"):
    """Generate heatmap visualization for all strategies."""
    
    num_strategies = len(all_positions)
    cols = 4
    rows = (num_strategies + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()
    
    # GRF pitch dimensions (normalized)
    x_range = [-1.0, 1.0]
    y_range = [-0.42, 0.42]
    
    for strat_id in range(num_strategies):
        ax = axes[strat_id]
        positions = all_positions[strat_id]
        
        if len(positions) > 0:
            # Player position heatmap
            h = ax.hist2d(
                positions[:, 0], 
                positions[:, 1], 
                bins=30, 
                range=[x_range, y_range], 
                cmap='Reds',
                density=True
            )
            plt.colorbar(h[3], ax=ax, label='Density')
            
            # Draw pitch lines
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
            
            # Goal areas
            ax.axvline(x=-0.7, color='white', linestyle=':', alpha=0.3)
            ax.axvline(x=0.7, color='white', linestyle=':', alpha=0.3)
        
        ax.set_title(f"Strategy {strat_id}", fontsize=14, fontweight='bold')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel("X (Left Goal ← → Right Goal)")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        
        # Add statistics
        if len(positions) > 0:
            mean_x = positions[:, 0].mean()
            mean_y = positions[:, 1].mean()
            std_x = positions[:, 0].std()
            std_y = positions[:, 1].std()
            ax.text(0.02, 0.98, f"μ=({mean_x:.2f},{mean_y:.2f})\nσ=({std_x:.2f},{std_y:.2f})", 
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_strategies, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Strategy Heatmaps - Player Positions\n(Good: Different patterns | Bad: All identical or corner-camping)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved heatmaps to: {output_path}")
    
    return fig


def compute_strategy_diversity(all_positions):
    """
    Compute metrics to quantify strategy diversity.
    
    Returns:
        diversity_score: Higher = more diverse strategies
        diagnosis: String describing the result
    """
    num_strategies = len(all_positions)
    
    # Compute mean position for each strategy
    means = []
    stds = []
    for strat_id in range(num_strategies):
        pos = all_positions[strat_id]
        if len(pos) > 0:
            means.append(pos.mean(axis=0))
            stds.append(pos.std(axis=0))
        else:
            means.append(np.array([0, 0]))
            stds.append(np.array([0, 0]))
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Inter-strategy variance (how different are the mean positions?)
    inter_variance = np.var(means, axis=0).sum()
    
    # Intra-strategy variance (how spread out is each strategy?)
    intra_variance = np.mean(stds ** 2)
    
    # Diversity ratio
    diversity_score = inter_variance / (intra_variance + 1e-6)
    
    # Diagnose
    print("\n" + "="*60)
    print("STRATEGY DIVERSITY ANALYSIS")
    print("="*60)
    
    print(f"\nMean positions per strategy:")
    for i, m in enumerate(means):
        print(f"  Strategy {i}: x={m[0]:+.3f}, y={m[1]:+.3f}")
    
    print(f"\nMetrics:")
    print(f"  Inter-strategy variance: {inter_variance:.4f}")
    print(f"  Intra-strategy variance: {intra_variance:.4f}")
    print(f"  Diversity ratio: {diversity_score:.4f}")
    
    # Diagnosis
    if inter_variance < 0.01:
        diagnosis = "🔴 MODE COLLAPSE - All strategies produce identical behavior!"
    elif intra_variance < 0.001:
        diagnosis = "🔴 CORNER CAMPING - Players stuck in fixed positions!"
    elif diversity_score > 0.5:
        diagnosis = "🟢 GOOD DIVERSITY - Strategies show different spatial patterns!"
    elif diversity_score > 0.1:
        diagnosis = "🟡 MODERATE DIVERSITY - Some differentiation, could be better."
    else:
        diagnosis = "🟠 LOW DIVERSITY - Strategies are too similar."
    
    print(f"\nDiagnosis: {diagnosis}")
    print("="*60)
    
    return diversity_score, diagnosis


def main():
    parser = argparse.ArgumentParser(description="Visualize strategy heatmaps from Phase 1")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to Phase 1 checkpoint directory")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per strategy (default: 5)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--output", type=str, default="strategy_heatmaps.png",
                        help="Output image path")
    parser.add_argument("--num_strategies", type=int, default=8,
                        help="Number of strategies to test (default: 8)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cpu)")
    args = parser.parse_args()
    
    print("="*60)
    print("STRATEGY HEATMAP VISUALIZATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes per strategy: {args.episodes}")
    print(f"Strategies to test: {args.num_strategies}")
    
    # Create environment
    print("\n📦 Creating environment...")
    env = create_env()
    
    # Load policy
    print(f"\n🔧 Loading policy from {args.checkpoint}...")
    try:
        policy = load_policy(args.checkpoint, device=args.device)
        print("✅ Policy loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        print("\nFalling back to manual loading...")
        # Manual fallback - just test env without policy
        policy = None
    
    if policy is None:
        print("\n⚠️ Running without policy (random actions) - just to test env setup")
        # Run with random actions to verify env works
        for strat in range(2):
            obs = env.reset()
            for _ in range(10):
                # GRF multi-agent: sample individual actions (0-18), not array
                actions = [np.random.randint(0, 19) for _ in range(4)]
                obs, _, done, _ = env.step(actions)
                if isinstance(done, (list, np.ndarray)):
                    done = all(done)
                if done:
                    break
        print("✅ Environment works. Fix policy loading to generate real heatmaps.")
        env.close()
        return
    
    # Collect data for each strategy
    all_positions = {}
    all_ball_positions = {}
    
    for strat_id in range(args.num_strategies):
        print(f"\n🎮 Running Strategy {strat_id}...")
        positions, ball_positions = run_episodes_with_strategy(
            env, policy, strat_id, 
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )
        all_positions[strat_id] = positions
        all_ball_positions[strat_id] = ball_positions
        print(f"   Collected {len(positions)} position samples")
    
    env.close()
    
    # Generate visualizations
    print("\n📊 Generating heatmaps...")
    plot_heatmaps(all_positions, all_ball_positions, args.output)
    
    # Compute diversity metrics
    compute_strategy_diversity(all_positions)
    
    print("\n✅ Done! Check the output image for visual diagnosis.")


if __name__ == "__main__":
    main()

