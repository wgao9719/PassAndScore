#!/usr/bin/env python
"""
Quick Strategy Diversity Test

Tests if Phase 1 trained different strategies or collapsed to one behavior.

Usage:
    cd GRF_MARL
    python test_strategy_diversity.py --checkpoint ./results/phase1/latest
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_checkpoint(checkpoint_dir, device="cpu"):
    """Load actor and backbone from checkpoint."""
    
    actor_path = os.path.join(checkpoint_dir, "actor.pt")
    backbone_path = os.path.join(checkpoint_dir, "backbone.pt")
    
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"No actor.pt found in {checkpoint_dir}")
    
    print(f"Loading from: {checkpoint_dir}")
    # Always load to CPU first, then move if needed
    actor_state = torch.load(actor_path, map_location="cpu")
    backbone_state = torch.load(backbone_path, map_location="cpu") if os.path.exists(backbone_path) else None
    
    print(f"  Actor keys: {len(actor_state)} parameters")
    if backbone_state:
        print(f"  Backbone keys: {len(backbone_state)} parameters")
    
    return actor_state, backbone_state


def test_film_diversity(actor_state, num_strategies=8, device="cpu"):
    """
    Test if FiLM layers produce different outputs for different strategies.
    This is a quick sanity check without running the full environment.
    """
    
    # Find FiLM-related weights
    film_keys = [k for k in actor_state.keys() if 'film' in k.lower()]
    
    if not film_keys:
        print("\n⚠️ No FiLM layers found in actor. Strategy conditioning may not be implemented.")
        return None
    
    print(f"\nFiLM parameters found: {film_keys}")
    
    # Look for the embedding layer (maps strategy code → embedding)
    embed_key = None
    for k in actor_state.keys():
        if 'embed' in k.lower() and 'weight' in k.lower():
            embed_key = k
            break
    
    if embed_key:
        embeddings = actor_state[embed_key]
        print(f"\nStrategy embeddings shape: {embeddings.shape}")
        
        # Compute pairwise distances between strategy embeddings
        num_strat = min(embeddings.shape[0], num_strategies)
        
        print("\nStrategy Embedding Similarity Matrix:")
        print("(Lower = more different, Higher = more similar)")
        print("-" * 50)
        
        # Normalize embeddings
        embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Print similarity matrix
        print("     ", end="")
        for i in range(num_strat):
            print(f"  S{i}  ", end="")
        print()
        
        for i in range(num_strat):
            print(f"S{i}  ", end="")
            for j in range(num_strat):
                sim = similarity[i, j].item()
                if i == j:
                    print(f" [{sim:.2f}]", end="")
                else:
                    print(f"  {sim:.2f} ", end="")
            print()
        
        # Compute diversity score
        off_diagonal = similarity[~torch.eye(num_strat, dtype=bool)].mean().item()
        diversity = 1.0 - off_diagonal
        
        print(f"\n📊 Diversity Score: {diversity:.3f}")
        print(f"   (0.0 = all identical, 1.0 = maximally different)")
        
        if diversity < 0.1:
            print("🔴 CRITICAL: Strategy embeddings are nearly identical!")
            print("   → Mode collapse likely. Check discriminator loss.")
        elif diversity < 0.3:
            print("🟡 WARNING: Low diversity between strategies.")
        else:
            print("🟢 GOOD: Strategy embeddings show meaningful differences.")
        
        return diversity
    else:
        print("No strategy embedding layer found.")
        return None


def test_film_output_diversity(checkpoint_dir, device="cpu"):
    """
    Actually run strategies through FiLM and check output diversity.
    """
    
    # Try to load the full model
    try:
        from light_malib.model.gr_football.passandscore_factorized import FiLMGenerator
        
        # Check if we can find FiLM params
        actor_path = os.path.join(checkpoint_dir, "actor.pt")
        actor_state = torch.load(actor_path, map_location="cpu")
        
        # Find FiLM network parameters
        film_state = {}
        for k, v in actor_state.items():
            if 'film' in k.lower():
                # Remove prefix to get FiLM-only state dict
                new_key = k.split('film_generator.')[-1] if 'film_generator.' in k else k
                film_state[new_key] = v
        
        if not film_state:
            return
        
        # Create FiLM generator
        # Infer dimensions from weights
        for k, v in film_state.items():
            if 'film_net.0.weight' in k:
                num_strategies = v.shape[1]
                embed_dim = v.shape[0]
            if 'film_net.2.weight' in k:
                hidden_dim = v.shape[0] // 2
        
        print(f"\nInferred: {num_strategies} strategies, hidden_dim={hidden_dim}")
        
        film = FiLMGenerator(
            num_strategies=num_strategies,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim if 'embed_dim' in dir() else 32
        )  # Keep on CPU
        
        # Load state (try to match keys)
        try:
            film.load_state_dict(film_state, strict=False)
        except:
            print("Could not load FiLM state dict exactly, checking outputs anyway...")
        
        # Test outputs
        print("\nFiLM (gamma, beta) outputs per strategy:")
        print("-" * 60)
        
        gammas = []
        betas = []
        
        for s in range(min(num_strategies, 8)):
            code = torch.tensor([s])  # CPU tensor
            gamma, beta = film(code)
            gammas.append(gamma.mean().item())
            betas.append(beta.mean().item())
            print(f"Strategy {s}: γ_mean={gamma.mean():.3f}, β_mean={beta.mean():.3f}")
        
        gamma_spread = max(gammas) - min(gammas)
        beta_spread = max(betas) - min(betas)
        
        print(f"\nγ spread: {gamma_spread:.3f}")
        print(f"β spread: {beta_spread:.3f}")
        
        if gamma_spread < 0.05 and beta_spread < 0.05:
            print("🔴 FiLM outputs nearly identical across strategies!")
        elif gamma_spread > 0.1 or beta_spread > 0.1:
            print("🟢 FiLM shows meaningful variation across strategies.")
        
    except ImportError as e:
        print(f"Could not import FiLM module: {e}")
    except Exception as e:
        print(f"FiLM output test failed: {e}")


def plot_embedding_space(actor_state, output_path="strategy_embeddings.png"):
    """Visualize strategy embeddings in 2D."""
    
    # Find embedding layer
    embed_key = None
    for k in actor_state.keys():
        if 'embed' in k.lower() and 'weight' in k.lower():
            embed_key = k
            break
    
    if embed_key is None:
        print("No embeddings to visualize")
        return
    
    embeddings = actor_state[embed_key].cpu().numpy()
    num_strategies = embeddings.shape[0]
    
    # PCA to 2D
    from sklearn.decomposition import PCA
    
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        variance_explained = sum(pca.explained_variance_ratio_)
    else:
        embeddings_2d = embeddings
        variance_explained = 1.0
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    
    for i in range(num_strategies):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                   c=[colors[i]], s=200, label=f'Strategy {i}', 
                   edgecolors='black', linewidths=2)
        plt.annotate(f'S{i}', (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=12, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Strategy Embeddings (PCA, {variance_explained:.1%} variance explained)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 Saved embedding visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Phase 1 checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print("="*60)
    print("STRATEGY DIVERSITY DIAGNOSTIC")
    print("="*60)
    
    # Load checkpoint
    actor_state, backbone_state = load_checkpoint(args.checkpoint, args.device)
    
    # Test 1: Embedding diversity
    print("\n" + "="*60)
    print("TEST 1: Strategy Embedding Diversity")
    print("="*60)
    diversity = test_film_diversity(actor_state, device=args.device)
    
    # Test 2: FiLM output diversity  
    print("\n" + "="*60)
    print("TEST 2: FiLM Output Diversity")
    print("="*60)
    test_film_output_diversity(args.checkpoint, args.device)
    
    # Visualize embeddings
    try:
        plot_embedding_space(actor_state)
    except ImportError:
        print("\n(sklearn not available for PCA visualization)")
    except Exception as e:
        print(f"\nVisualization failed: {e}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
If diversity is LOW:
  1. Check discriminator loss - is it decreasing?
  2. Increase discriminator learning rate
  3. Add discriminator gradient penalty
  4. Use stronger intrinsic reward coefficient

If diversity is OK but behavior is bad:
  1. Run full heatmap visualization (visualize_strategies.py)
  2. Check if strategies map to useful behaviors
  3. May need more Phase 1 training epochs
""")


if __name__ == "__main__":
    main()

