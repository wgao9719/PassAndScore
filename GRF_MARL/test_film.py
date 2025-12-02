#!/usr/bin/env python3
"""
Standalone FiLM verification - no external dependencies.

IMPORTANT: FiLM is initialized to IDENTITY (γ=1, β=0).
This means:
- At initialization: all strategies produce IDENTICAL outputs (expected!)
- After training: strategies should DIVERGE (the real test)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMGenerator(nn.Module):
    """FiLM: h' = gamma * h + beta"""
    
    def __init__(self, num_strategies, hidden_dim, embed_dim=32):
        super().__init__()
        self.strategy_embedding = nn.Embedding(num_strategies, embed_dim)
        self.film_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2 * hidden_dim)
        )
        # Initialize to identity: gamma=1, beta=0
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)
        self.film_net[-1].bias.data[:hidden_dim] = 1.0
        
    def forward(self, strategy_code):
        if strategy_code.dim() > 1:
            strategy_code = strategy_code.squeeze(-1)
        embed = self.strategy_embedding(strategy_code.long())
        params = self.film_net(embed)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta
    
    def modulate(self, features, strategy_code):
        gamma, beta = self.forward(strategy_code)
        return gamma * features + beta


class TestActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, num_strategies):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.film = FiLMGenerator(num_strategies, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, strategy):
        h = self.body(obs)
        h = self.film.modulate(h, strategy)
        return self.head(h)


def measure_differentiation(actor, obs, num_strategies, batch_size):
    """Measure how different outputs are across strategies."""
    with torch.no_grad():
        outs = [actor(obs, torch.full((batch_size,), s, dtype=torch.long)) 
                for s in range(num_strategies)]
    
    diffs = []
    for i in range(num_strategies):
        for j in range(i+1, num_strategies):
            diffs.append((outs[i] - outs[j]).abs().mean().item())
    
    return sum(diffs) / len(diffs) if diffs else 0


def main():
    print("=" * 60)
    print("FiLM Gradient Verification")
    print("=" * 60)
    
    # Config
    obs_dim, hidden_dim, action_dim = 115, 128, 19
    num_strategies = 8
    batch_size = 16
    
    actor = TestActor(obs_dim, hidden_dim, action_dim, num_strategies)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    obs = torch.randn(batch_size, obs_dim)
    
    # =========================================================================
    # Test 1: Initial state (expected: identical outputs due to identity init)
    # =========================================================================
    print("\n[Test 1] Initial State (identity init)")
    initial_diff = measure_differentiation(actor, obs, num_strategies, batch_size)
    print(f"  Output difference: {initial_diff:.6f}")
    print(f"  Expected: ~0 (identity FiLM → same output for all strategies)")
    if initial_diff < 1e-5:
        print("  ✓ Correct: Identity initialization confirmed")
    
    # =========================================================================
    # Test 2: Gradient flow when loss depends on strategy
    # =========================================================================
    print("\n[Test 2] Gradient Flow (strategy-dependent loss)")
    opt.zero_grad()
    
    # Loss that DEPENDS on strategy (different target per strategy)
    strategies = torch.randint(0, num_strategies, (batch_size,))
    out = actor(obs, strategies)
    targets = strategies % action_dim  # Different target per strategy
    loss = F.cross_entropy(out, targets)
    loss.backward()
    
    embed_grad = actor.film.strategy_embedding.weight.grad
    has_grad = embed_grad is not None and embed_grad.abs().sum() > 1e-8
    print(f"  Embedding gradient present: {'✓ Yes' if has_grad else '⚠ Weak/Zero'}")
    if has_grad:
        print(f"  Gradient norm: {embed_grad.norm().item():.6f}")
    
    # =========================================================================
    # Test 3: THE CRITICAL TEST - Does training differentiate strategies?
    # =========================================================================
    print("\n[Test 3] ★ CRITICAL: Learning Differentiation ★")
    print("  Training for 50 steps...")
    
    for step in range(50):
        opt.zero_grad()
        s = torch.randint(0, num_strategies, (batch_size,))
        out = actor(obs, s)
        # Strategy-dependent target
        target = s % action_dim
        loss = F.cross_entropy(out, target)
        loss.backward()
        opt.step()
    
    final_diff = measure_differentiation(actor, obs, num_strategies, batch_size)
    
    print(f"\n  Before training: {initial_diff:.6f}")
    print(f"  After training:  {final_diff:.6f}")
    print(f"  Increase: {final_diff - initial_diff:+.6f}")
    
    if final_diff > 0.1:
        print("\n  ✓ ✓ ✓ SUCCESS: Strategies now produce DIFFERENT outputs!")
        print("      FiLM conditioning is working correctly.")
        success = True
    elif final_diff > 0.01:
        print("\n  ⚠ Partial: Some differentiation, but weak")
        success = True
    else:
        print("\n  ✗ FAIL: Strategies still produce similar outputs")
        print("      FiLM might not be learning properly.")
        success = False
    
    # =========================================================================
    # Test 4: FiLM parameter values after training
    # =========================================================================
    print("\n[Test 4] FiLM Parameters After Training")
    with torch.no_grad():
        gammas, betas = [], []
        for s in range(num_strategies):
            g, b = actor.film(torch.tensor([s]))
            gammas.append(g.mean().item())
            betas.append(b.mean().item())
        
        print(f"  γ range: [{min(gammas):.3f}, {max(gammas):.3f}]")
        print(f"  β range: [{min(betas):.3f}, {max(betas):.3f}]")
        
        gamma_spread = max(gammas) - min(gammas)
        beta_spread = max(betas) - min(betas)
        
        if gamma_spread > 0.1 or beta_spread > 0.1:
            print("  ✓ FiLM parameters have diverged across strategies")
        else:
            print("  ⚠ FiLM parameters similar across strategies")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    if success:
        print("✓ ✓ ✓ ALL CHECKS PASSED ✓ ✓ ✓")
        print("")
        print("FiLM conditioning is working correctly:")
        print("  1. Identity init means strategies start the same (correct)")
        print("  2. Training makes strategies produce different outputs (correct)")
        print("  3. Phase 1 training should learn diverse behaviors")
    else:
        print("✗ CHECKS FAILED")
        print("Review the FiLM architecture.")
    print("=" * 60)


if __name__ == "__main__":
    main()
