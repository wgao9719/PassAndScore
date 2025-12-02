# Copyright 2024
#
# Factorized Actor-Critic for Multi-Agent PPO with CTDE/DTE support.
#
# CTDE (Centralized Training, Decentralized Execution):
#   - Actor: uses local observations only (factorized_actor_use_global=False)
#   - Critic: uses global/pooled observations (factorized_critic_use_local=False)
#
# DTE (Decentralized Training and Execution):
#   - Actor: uses local observations only (factorized_actor_use_global=False)
#   - Critic: uses local observations only (factorized_critic_use_local=True, use pooled=False)
#
# Fully Centralized (for comparison):
#   - Actor: uses local + global (factorized_actor_use_global=True)
#   - Critic: uses global + local (factorized_critic_use_local=True)
#
# Phase 1 Strategy Conditioning:
#   - FiLM conditioning on categorical strategy code c
#   - Discriminator predicts c from trajectory -> intrinsic reward
#   - Maximizes mutual information I(c; trajectory)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete

from light_malib.algorithm.utils import init_fc_weights
from light_malib.utils.preprocessor import get_preprocessor

# Export all classes for proper pickling
__all__ = [
    'FiLMGenerator',
    'TransformerEncoder', 
    'Supervisor',
    'StrategyDiscriminator',
    'Backbone',
    'Actor',
    'Critic',
    'FeatureEncoder',
]

# Use the standard GRF FeatureEncoder for encoding State objects
from light_malib.envs.gr_football.encoders.encoder_basic import FeatureEncoder


# =============================================================================
# FiLM Conditioning: Feature-wise Linear Modulation
# =============================================================================

class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from a categorical strategy code.
    
    FiLM: h' = gamma * h + beta
    
    Args:
        num_strategies: Number of discrete strategy codes K
        hidden_dim: Dimension of the features to modulate
        embed_dim: Dimension of strategy embedding (optional)
    """
    
    def __init__(self, num_strategies: int, hidden_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_strategies = num_strategies
        self.hidden_dim = hidden_dim
        
        # Learnable embedding for each strategy code
        self.strategy_embedding = nn.Embedding(num_strategies, embed_dim)
        
        # Generate FiLM parameters from embedding
        self.film_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2 * hidden_dim)  # gamma and beta
        )
        
        # Initialize to identity transform: gamma=1, beta=0
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)
        self.film_net[-1].bias.data[:hidden_dim] = 1.0  # gamma = 1
        
    def forward(self, strategy_code: torch.Tensor) -> tuple:
        """
        Args:
            strategy_code: [batch] or [batch, 1] tensor of strategy indices
        Returns:
            gamma: [batch, hidden_dim] scale factors
            beta: [batch, hidden_dim] shift factors
        """
        if strategy_code.dim() > 1:
            strategy_code = strategy_code.squeeze(-1)
        strategy_code = strategy_code.long()
        
        # Get embedding and generate FiLM params
        embed = self.strategy_embedding(strategy_code)  # [batch, embed_dim]
        film_params = self.film_net(embed)  # [batch, 2*hidden_dim]
        
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma, beta
    
    def modulate(self, features: torch.Tensor, strategy_code: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: h' = gamma * h + beta"""
        gamma, beta = self.forward(strategy_code)
        
        # Expand gamma/beta to match features shape
        while gamma.dim() < features.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return gamma * features + beta


# =============================================================================
# Strategy Discriminator: Predicts strategy code from trajectory
# =============================================================================

# =============================================================================
# Supervisor Network: Selects strategy code from global state (Phase 2)
# =============================================================================

class TransformerEncoder(nn.Module):
    """
    Small Transformer encoder for processing entity lists.
    Captures relational information between entities (ball, players).
    """
    
    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Project entity features to hidden dim
        self.input_proj = nn.Linear(entity_dim, hidden_dim)
        
        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entities: [batch, num_entities, entity_dim]
        Returns:
            hidden: [batch, hidden_dim] aggregated representation
        """
        batch_size = entities.shape[0]
        
        # Project to hidden dim
        x = self.input_proj(entities)  # [batch, num_entities, hidden_dim]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1 + num_entities, hidden_dim]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, 1 + num_entities, hidden_dim]
        
        # Take CLS token output as aggregated representation
        hidden = self.norm(x[:, 0])  # [batch, hidden_dim]
        
        return hidden


class Supervisor(nn.Module):
    """
    "God-View" Supervisor policy that selects strategy codes.
    
    Phase 2: Players are frozen, supervisor learns which strategy to use.
    
    c_t = F_φ(s_t^global)
    
    KEY INSIGHT: Don't pool observations! Use a Transformer to preserve
    spatial/relational information between entities.
    
    Input: Raw entity states (Ball + 22 Players) as (23, Features)
    Architecture: Transformer Encoder (2 layers, 4 heads)
    
    Why Transformer: Attention allows reasoning like:
      "If [Striker] near [Defender] but [Winger] far from [Defender] -> Strategy 2"
    Pooling destroys this relational information.
    
    Args:
        entity_dim: Dimension of per-entity features
        num_entities: Number of entities (default 23 = ball + 22 players)
        num_strategies: Number of discrete strategy codes K
        hidden_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        use_transformer: If False, falls back to MLP (for comparison)
    """
    
    def __init__(
        self,
        obs_dim: int = 115,  # Kept for interface compatibility
        num_strategies: int = 8,
        hidden_dim: int = 128,
        num_agents: int = 4,
        # God-View specific params
        entity_dim: int = 6,  # (x, y, vx, vy, direction, tired) per entity
        num_entities: int = 23,  # ball + 22 players
        num_heads: int = 4,
        num_layers: int = 2,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_strategies = num_strategies
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.entity_dim = entity_dim
        self.num_entities = num_entities
        self.use_transformer = use_transformer
        
        # Placeholder RNN states (interface compatibility)
        self.rnn_layer_num = 1
        self.rnn_state_size = 1
        
        if use_transformer:
            # Transformer-based "God-View" encoder
            # Preserves spatial/relational information
            self.encoder = TransformerEncoder(
                entity_dim=entity_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            encoder_output_dim = hidden_dim
        else:
            # Fallback: MLP on concatenated observations (loses relations)
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            encoder_output_dim = hidden_dim
        
        # Policy head: outputs distribution over strategies
        self.policy_head = nn.Linear(encoder_output_dim, num_strategies)
        
        # Value head: estimates expected team return
        self.value_head = nn.Linear(encoder_output_dim, 1)
        
        # Initialize with small weights for stable start
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def _extract_entities(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract entity features from raw GRF observation.
        
        GRF observation layout (simplified, adjust indices as needed):
          - Ball: position (x, y), velocity (vx, vy)
          - Left team (11 players): position, velocity, direction, tired
          - Right team (11 players): position, velocity, direction, tired
        
        Returns:
            entities: [batch, 23, entity_dim] tensor
        """
        batch_size = obs.shape[0]
        
        # GRF observation indices (approximate, may need adjustment)
        # These are based on typical GRF "simple115" representation
        # Ball: indices 88-91 (x, y, z, ownership)
        # Left team positions: indices 0-21 (11 * 2)
        # Left team directions: indices 44-54
        # Right team positions: indices 22-43 (11 * 2)
        # Right team directions: indices 55-65
        
        entities = []
        
        # Ball (entity 0)
        if obs.shape[-1] >= 92:
            ball_pos = obs[..., 88:90]  # x, y
            ball_vel = torch.zeros(batch_size, 2, device=obs.device)  # placeholder
            ball_dir = torch.zeros(batch_size, 1, device=obs.device)
            ball_tired = torch.zeros(batch_size, 1, device=obs.device)
            ball = torch.cat([ball_pos, ball_vel, ball_dir, ball_tired], dim=-1)
            entities.append(ball)
        else:
            # Fallback: just use zeros
            entities.append(torch.zeros(batch_size, self.entity_dim, device=obs.device))
        
        # Left team (entities 1-11)
        for i in range(11):
            if obs.shape[-1] >= 66:
                pos = obs[..., i*2:(i+1)*2]  # positions
                vel = torch.zeros(batch_size, 2, device=obs.device)
                direction = obs[..., 44+i:45+i] if obs.shape[-1] > 44+i else torch.zeros(batch_size, 1, device=obs.device)
                tired = torch.zeros(batch_size, 1, device=obs.device)
                player = torch.cat([pos, vel, direction, tired], dim=-1)
            else:
                player = torch.zeros(batch_size, self.entity_dim, device=obs.device)
            entities.append(player)
        
        # Right team (entities 12-22)
        for i in range(11):
            if obs.shape[-1] >= 66:
                pos = obs[..., 22+i*2:22+(i+1)*2]  # positions
                vel = torch.zeros(batch_size, 2, device=obs.device)
                direction = obs[..., 55+i:56+i] if obs.shape[-1] > 55+i else torch.zeros(batch_size, 1, device=obs.device)
                tired = torch.zeros(batch_size, 1, device=obs.device)
                player = torch.cat([pos, vel, direction, tired], dim=-1)
            else:
                player = torch.zeros(batch_size, self.entity_dim, device=obs.device)
            entities.append(player)
        
        # Stack all entities: [batch, 23, entity_dim]
        entities = torch.stack(entities, dim=1)
        
        return entities
    
    def forward(
        self,
        global_obs: torch.Tensor,
        explore: bool = True,
        strategy: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass for supervisor policy.
        
        Args:
            global_obs: [batch, obs_dim] observation (will be reshaped to entities)
                        OR [batch, num_entities, entity_dim] pre-extracted entities
            explore: Whether to sample (True) or take argmax (False)
            strategy: [batch] optional, for computing log_prob of given strategy
        
        Returns:
            strategy: [batch] selected strategy codes
            log_prob: [batch] log probability of selected strategy
            value: [batch, 1] estimated state value
            entropy: [batch] entropy of strategy distribution (if strategy provided)
        """
        if self.use_transformer:
            # Extract entities from observation if needed
            if global_obs.dim() == 2:
                entities = self._extract_entities(global_obs)
            else:
                entities = global_obs  # Already in entity format
            hidden = self.encoder(entities)
        else:
            # Fallback MLP encoder
            hidden = self.encoder(global_obs)
        
        # Policy: distribution over strategies
        logits = self.policy_head(hidden)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Value estimate
        value = self.value_head(hidden)
        
        if strategy is None:
            # Sampling mode
            if explore:
                strategy = dist.sample()
            else:
                strategy = logits.argmax(dim=-1)
            log_prob = dist.log_prob(strategy)
            entropy = None
        else:
            # Evaluation mode
            log_prob = dist.log_prob(strategy)
            entropy = dist.entropy()
        
        return strategy, log_prob, value, entropy
    
    def get_value(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for global observation."""
        if self.use_transformer:
            if global_obs.dim() == 2:
                entities = self._extract_entities(global_obs)
            else:
                entities = global_obs
            hidden = self.encoder(entities)
        else:
            hidden = self.encoder(global_obs)
        return self.value_head(hidden)


class StrategyDiscriminator(nn.Module):
    """
    Discriminator that predicts the strategy code c from a trajectory window.
    
    Uses a GRU to process structured features (NOT blind pooling), then
    outputs a distribution over strategy codes.
    
    KEY INSIGHT: Don't pool blindly!
    - Mean pooling loses "attacking left" vs "attacking right" distinction
    - Instead: use structured features that preserve spatial information
    
    Input Features (per timestep):
    - Ball position (2D)
    - Ball velocity (2D)  
    - Team centroid (2D) - where is the team as a whole?
    - Team spread (1D) - how spread out is the team?
    - Controlled player position (2D)
    - Closest teammate relative position (2D)
    - Actions (one-hot, optional)
    
    Window Size: 20-50 steps is the sweet spot for GRF tactics
    - Too short (5): only sees velocity/jitter
    - Too long (100): gradient propagation is hard
    
    Args:
        obs_dim: Dimension of observations
        num_strategies: Number of discrete strategy codes K
        hidden_dim: GRU hidden dimension
        window_size: Number of past steps to consider (recommended: 20-50)
        num_agents: Number of controlled agents
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_strategies: int,
        hidden_dim: int = 64,
        window_size: int = 32,  # Changed default: 32 is in sweet spot
        num_agents: int = 4,
        use_actions: bool = True,
        action_dim: int = 19,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_strategies = num_strategies
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_agents = num_agents
        self.use_actions = use_actions
        self.action_dim = action_dim
        
        # Structured feature dimensions (per timestep):
        # - Ball: position(2) + velocity(2) = 4
        # - Team: centroid(2) + spread(1) = 3
        # - Per-agent: position(2) = 2 * num_agents
        # - Relative features: closest teammate per agent (2) * num_agents
        # Total without actions: 4 + 3 + 4*num_agents
        self.structured_feature_dim = 4 + 3 + 4 * num_agents
        
        input_dim = self.structured_feature_dim
        if use_actions:
            input_dim += action_dim * num_agents
        
        # Feature projection (normalize scales)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Temporal encoder: GRU over the trajectory window
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,  # Increased depth
            batch_first=True,
            dropout=0.1,
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_strategies),
        )
    
    def _extract_structured_features(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extract structured features from observations.
        
        IMPORTANT: Don't blindly pool! Preserve spatial information.
        
        Args:
            obs_seq: [batch, window, agents, obs_dim]
            action_seq: [batch, window, agents] (optional)
        
        Returns:
            features: [batch, window, feature_dim]
        """
        batch_size = obs_seq.shape[0]
        window_size = obs_seq.shape[1]
        num_agents = obs_seq.shape[2]
        device = obs_seq.device
        
        # GRF observation layout (simple115 format, approximate):
        # 0-21: left team positions (11 * 2)
        # 22-43: right team positions (11 * 2)
        # 44-54: left team directions (11)
        # 55-65: right team directions (11)
        # 66-87: various features
        # 88-90: ball position (x, y, z)
        # 91: ball ownership
        # etc.
        
        features_list = []
        
        # 1. Ball features (position + velocity proxy)
        # Ball position from first agent's observation
        if obs_seq.shape[-1] > 90:
            ball_pos = obs_seq[:, :, 0, 88:90]  # [batch, window, 2]
            # Ball velocity: difference between consecutive positions
            ball_vel = torch.zeros_like(ball_pos)
            ball_vel[:, 1:] = ball_pos[:, 1:] - ball_pos[:, :-1]
        else:
            ball_pos = torch.zeros(batch_size, window_size, 2, device=device)
            ball_vel = torch.zeros(batch_size, window_size, 2, device=device)
        
        features_list.append(ball_pos)
        features_list.append(ball_vel)
        
        # 2. Agent positions (preserve individual positions, don't pool!)
        # First 2 features of each agent's obs should be their position
        agent_positions = obs_seq[:, :, :, 0:2]  # [batch, window, agents, 2]
        agent_positions_flat = agent_positions.reshape(batch_size, window_size, -1)  # [batch, window, agents*2]
        features_list.append(agent_positions_flat)
        
        # 3. Team centroid (where is the team as a whole?)
        team_centroid = agent_positions.mean(dim=2)  # [batch, window, 2]
        features_list.append(team_centroid)
        
        # 4. Team spread (how spread out? compact vs wide formation)
        team_spread = ((agent_positions - team_centroid.unsqueeze(2)) ** 2).sum(dim=-1).mean(dim=2, keepdim=True)
        team_spread = torch.sqrt(team_spread + 1e-6)  # [batch, window, 1]
        features_list.append(team_spread)
        
        # 5. Closest teammate relative positions (per agent)
        # For each agent, find relative position to closest teammate
        closest_rel = torch.zeros(batch_size, window_size, num_agents, 2, device=device)
        for i in range(num_agents):
            agent_pos = agent_positions[:, :, i:i+1, :]  # [batch, window, 1, 2]
            other_pos = torch.cat([
                agent_positions[:, :, :i, :],
                agent_positions[:, :, i+1:, :]
            ], dim=2)  # [batch, window, agents-1, 2]
            
            if other_pos.shape[2] > 0:
                dists = ((agent_pos - other_pos) ** 2).sum(dim=-1)  # [batch, window, agents-1]
                closest_idx = dists.argmin(dim=-1)  # [batch, window]
                
                # Gather closest teammate relative position
                for b in range(batch_size):
                    for t in range(window_size):
                        idx = closest_idx[b, t].item()
                        closest_rel[b, t, i] = other_pos[b, t, idx] - agent_pos[b, t, 0]
        
        closest_rel_flat = closest_rel.reshape(batch_size, window_size, -1)  # [batch, window, agents*2]
        features_list.append(closest_rel_flat)
        
        # 6. Actions (if using)
        if self.use_actions and action_seq is not None:
            actions_onehot = F.one_hot(action_seq.long(), num_classes=self.action_dim)
            actions_flat = actions_onehot.reshape(batch_size, window_size, -1).float()
            features_list.append(actions_flat)
        
        # Concatenate all features
        features = torch.cat(features_list, dim=-1)  # [batch, window, feature_dim]
        
        return features
        
    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_seq: [batch, window_size, num_agents, obs_dim] observations
            action_seq: [batch, window_size, num_agents] actions (optional)
        
        Returns:
            logits: [batch, num_strategies] unnormalized log probabilities
        """
        # Extract structured features (no blind pooling!)
        features = self._extract_structured_features(obs_seq, action_seq)
        
        # Project features
        features = self.feature_proj(features)
        
        # Pass through GRU
        _, hidden = self.gru(features)  # hidden: [2, batch, hidden_dim]
        hidden = hidden[-1]  # Take last layer: [batch, hidden_dim]
        
        # Classify
        logits = self.classifier(hidden)  # [batch, num_strategies]
        return logits
    
    def log_prob(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        strategy_code: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of the true strategy code.
        
        This is used for the intrinsic reward: r_intrinsic = log q(c | trajectory)
        
        Returns:
            log_probs: [batch] log probabilities
        """
        logits = self.forward(obs_seq, action_seq)
        log_probs = F.log_softmax(logits, dim=-1)
        
        if strategy_code.dim() > 1:
            strategy_code = strategy_code.squeeze(-1)
        
        # Gather log prob for the true strategy
        return log_probs.gather(1, strategy_code.long().unsqueeze(-1)).squeeze(-1)
    
    def predict(self, obs_seq: torch.Tensor, action_seq: torch.Tensor = None) -> torch.Tensor:
        """Predict most likely strategy code."""
        logits = self.forward(obs_seq, action_seq)
        return logits.argmax(dim=-1)
    
    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        strategy_code: torch.Tensor,
    ) -> tuple:
        """
        Compute cross-entropy loss for discriminator training.
        
        Returns:
            loss: scalar loss
            accuracy: classification accuracy
        """
        logits = self.forward(obs_seq, action_seq)
        
        if strategy_code.dim() > 1:
            strategy_code = strategy_code.squeeze(-1)
        
        loss = F.cross_entropy(logits, strategy_code.long())
        
        # Compute accuracy for monitoring
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == strategy_code.long()).float().mean()
        
        return loss, accuracy


def _build_mlp(input_dim, layers_cfg, initialization):
    """Build a simple MLP body with configurable layers."""
    modules = []
    last_dim = input_dim
    
    for layer in layers_cfg or []:
        units = int(layer["units"])
        linear = nn.Linear(last_dim, units)
        modules.append(linear)
        act_name = layer.get("activation")
        if act_name and hasattr(nn, act_name):
            modules.append(getattr(nn, act_name)())
        last_dim = units

    seq = nn.Sequential(*modules) if modules else nn.Identity()

    use_orthogonal = initialization.get("use_orthogonal", False)
    gain = initialization.get("gain", 1.0)
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(use_orthogonal)]

    def _init(m):
        if isinstance(m, nn.Linear):
            init_fc_weights(m, init_method, gain)

    seq.apply(_init)
    return seq, last_dim


class Backbone(nn.Module):
    """
    Shared backbone that processes per-agent observations and creates
    pooled/global representations for CTDE/DTE.
    
    Outputs a dictionary with:
      - "local": per-agent observations [batch*num_agents, obs_dim]
      - "global": mean-pooled obs broadcast to all agents [batch*num_agents, obs_dim]
      - "pooled": raw mean-pooled observations [batch, obs_dim]
    """

    def __init__(
        self,
        model_config,
        global_observation_space,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()

        self.num_agents = int(custom_config.get("num_agents", 2))
        if self.num_agents <= 0:
            raise ValueError("custom_config.num_agents must be > 0 for factorized model.")

        if not isinstance(observation_space, Box):
            raise TypeError("Backbone expects a Box observation space.")

        self.obs_dim = get_preprocessor(observation_space)(observation_space).size
        
        # Optional: learned embedding before pooling
        embed_dim = model_config.get("embed_dim", None)
        if embed_dim is not None:
            self.embed = nn.Linear(self.obs_dim, embed_dim)
            self.embed_dim = embed_dim
        else:
            self.embed = None
            self.embed_dim = self.obs_dim

    def forward(self, states, observations, critic_rnn_states, rnn_masks):
        # Handle both tensor and numpy inputs
        if isinstance(observations, np.ndarray):
            obs = torch.from_numpy(observations).float()
        else:
            obs = observations.float()
        
        # Move to same device as parameters if we have any
        if self.embed is not None:
            obs = obs.to(self.embed.weight.device)
        
        # Reshape: [batch * num_agents, obs_dim] -> [batch, num_agents, obs_dim]
        batch = obs.shape[0] // self.num_agents
        obs = obs.view(batch, self.num_agents, -1)
        
        # Optional embedding
        if self.embed is not None:
            obs = self.embed(obs)
        
        # Mean pooling across agents for global representation
        pooled = obs.mean(dim=1)  # [batch, embed_dim]
        
        # Broadcast pooled back to all agents
        pooled_rep = pooled.unsqueeze(1).expand(-1, self.num_agents, -1)  # [batch, num_agents, embed_dim]

        return {
            "local": obs.reshape(batch * self.num_agents, -1),
            "global": pooled_rep.reshape(batch * self.num_agents, -1),
            "pooled": pooled,
        }


class Actor(nn.Module):
    """
    Factorized actor for multi-agent PPO with FiLM strategy conditioning.
    
    CTDE Mode (factorized_actor_use_global=False):
      - Input: local observations only
      - Each agent acts based only on its own observation
      
    Centralized Mode (factorized_actor_use_global=True):
      - Input: local + global (pooled) observations
      - Each agent sees team-level information
      
    Strategy Conditioning (Phase 1):
      - FiLM modulation: h' = gamma(c) * h + beta(c)
      - Enables diverse behaviors conditioned on strategy code c
    """

    def __init__(
        self,
        model_config,
        action_space,
        custom_config,
        initialization,
        backbone,
    ):
        super().__init__()

        self.backbone = backbone
        self.action_space = action_space
        
        # CTDE: use_global=False (decentralized execution)
        # Centralized: use_global=True
        self.use_global = custom_config.get("factorized_actor_use_global", False)

        self.flash_dim = int(custom_config.get("flash_dim", 0))
        self.use_flash = bool(custom_config.get("factorized_actor_use_flash", True))
        
        # Strategy conditioning (Phase 1 pre-training)
        self.use_strategy_conditioning = custom_config.get("use_strategy_conditioning", False)
        self.num_strategies = int(custom_config.get("num_strategies", 8))
        self.strategy_embed_dim = int(custom_config.get("strategy_embed_dim", 32))
        
        # Placeholder RNN states (not using RNN, but interface requires them)
        self.rnn_layer_num = 1
        self.rnn_state_size = 1

        act_dim = get_preprocessor(action_space)(action_space).size
        
        # Input dimension depends on whether we use global features
        if backbone.embed is not None:
            base_dim = backbone.embed_dim
        else:
            base_dim = backbone.obs_dim
            
        input_dim = base_dim + (base_dim if self.use_global else 0)
        if self.use_flash and self.flash_dim > 0:
            input_dim += self.flash_dim

        self.body, last_dim = _build_mlp(
            input_dim, model_config.get("layers", []), initialization
        )
        
        # FiLM generator for strategy conditioning
        # Applies modulation AFTER the MLP body, BEFORE the policy head
        if self.use_strategy_conditioning:
            self.film_generator = FiLMGenerator(
                num_strategies=self.num_strategies,
                hidden_dim=last_dim,
                embed_dim=self.strategy_embed_dim,
            )
        else:
            self.film_generator = None
        
        self.head = nn.Linear(last_dim, act_dim)

        use_orthogonal = initialization.get("use_orthogonal", False)
        gain = initialization.get("gain", 0.01)  # Smaller gain for policy head
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(use_orthogonal)]
        init_fc_weights(self.head, init_method, gain)

    def forward(
        self,
        observations,
        actor_rnn_states,
        rnn_masks,
        action_masks,
        explore,
        actions,
        strategy_code=None,
    ):
        local = observations["local"]
        
        if self.use_global:
            features = torch.cat([local, observations["global"]], dim=-1)
        else:
            features = local

        if self.use_flash and self.flash_dim > 0:
            flash = observations.get("flash_token")
            if flash is None:
                flash = torch.zeros(
                    features.shape[0],
                    self.flash_dim,
                    dtype=features.dtype,
                    device=features.device,
                )
            else:
                flash = torch.as_tensor(
                    flash, dtype=features.dtype, device=features.device
                )
                flash = flash.view(features.shape[0], self.flash_dim)
            features = torch.cat([features, flash], dim=-1)

        hidden = self.body(features)
        
        # Apply FiLM conditioning if enabled and strategy code is provided
        if self.use_strategy_conditioning and self.film_generator is not None:
            if strategy_code is None:
                # Default to strategy 0 if not provided (e.g., during evaluation)
                strategy_code = torch.zeros(
                    hidden.shape[0], dtype=torch.long, device=hidden.device
                )
            else:
                strategy_code = torch.as_tensor(
                    strategy_code, dtype=torch.long, device=hidden.device
                )
                if strategy_code.dim() == 0:
                    strategy_code = strategy_code.expand(hidden.shape[0])
                elif strategy_code.shape[0] != hidden.shape[0]:
                    # Expand strategy code to match batch size (same code for all agents)
                    strategy_code = strategy_code.repeat_interleave(
                        hidden.shape[0] // strategy_code.shape[0]
                    )
            
            hidden = self.film_generator.modulate(hidden, strategy_code)
        
        logits = self.head(hidden)

        # Apply action mask (mask out illegal actions)
        if action_masks is not None:
            illegal_action_mask = 1 - action_masks
            logits = logits - 1e10 * illegal_action_mask

        dist = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            # Sampling mode
            sampled = dist.sample() if explore else dist.probs.argmax(dim=-1)
            action_log_probs = dist.log_prob(sampled)
            entropy = None
            actions = sampled
        else:
            # Evaluation mode (computing log probs for given actions)
            action_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return actions, actor_rnn_states, action_log_probs, entropy


class Critic(nn.Module):
    """
    Value network for multi-agent PPO.
    
    CTDE Mode (factorized_critic_use_local=False):
      - Input: global/pooled observations only
      - Centralized critic sees team-level state
      
    DTE Mode (factorized_critic_use_local=True):
      - Input: local observations (+ optionally pooled)
      - Decentralized critic per agent
      
    Output:
      - V(s): scalar state value (use_q_head=False, default for PPO)
      - Q(s,a): per-action values (use_q_head=True)
    """

    def __init__(
        self,
        model_config,
        observation_space,  # This is action_space if use_q_head, else Discrete(1)
        custom_config,
        initialization,
        backbone,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_agents = backbone.num_agents
        
        self.flash_dim = int(custom_config.get("flash_dim", 0))
        self.use_flash = bool(custom_config.get("factorized_critic_use_flash", False))
        # CTDE: include_local=False (centralized critic with global state)
        # DTE: include_local=True (decentralized critic)
        self.include_local = custom_config.get("factorized_critic_use_local", False)
        
        # Whether critic uses pooled (global) features
        # For pure DTE, set both include_local=True and use_pooled=False
        self.use_pooled = custom_config.get("factorized_critic_use_pooled", True)
        
        # Placeholder RNN states
        self.rnn_layer_num = 1
        self.rnn_state_size = 1

        # Output dimension: 1 for V(s), action_dim for Q(s,a)
        use_q_head = custom_config.get("use_q_head", False)
        if use_q_head:
            output_dim = get_preprocessor(observation_space)(observation_space).size
        else:
            output_dim = 1  # Scalar V(s) for PPO
        
        # Input dimension
        if backbone.embed is not None:
            base_dim = backbone.embed_dim
        else:
            base_dim = backbone.obs_dim
        
        input_dim = 0
        if self.use_pooled:
            input_dim += base_dim
        if self.include_local:
            input_dim += base_dim
        
        # Fallback: if neither, use pooled
        if input_dim == 0:
            input_dim = base_dim
            self.use_pooled = True
        if self.use_flash and self.flash_dim > 0:
            input_dim += self.flash_dim

        self.body, last_dim = _build_mlp(
            input_dim, model_config.get("layers", []), initialization
        )
        self.head = nn.Linear(last_dim, output_dim)

        use_orthogonal = initialization.get("use_orthogonal", False)
        gain = initialization.get("gain", 1.0)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(use_orthogonal)]
        init_fc_weights(self.head, init_method, gain)

    def forward(self, observations, critic_rnn_states, rnn_masks):
        features_list = []
        
        if self.use_pooled:
            # Expand pooled to match per-agent batch size
            pooled = observations["pooled"]  # [batch, dim]
            pooled = pooled.repeat_interleave(self.num_agents, dim=0)  # [batch*num_agents, dim]
            features_list.append(pooled)
        
        if self.include_local:
            features_list.append(observations["local"])
        
        if len(features_list) > 1:
            features = torch.cat(features_list, dim=-1)
        else:
            features = features_list[0]

        if self.use_flash and self.flash_dim > 0:
            flash = observations.get("flash_token")
            if flash is None:
                flash = torch.zeros(
                    features.shape[0],
                    self.flash_dim,
                    dtype=features.dtype,
                    device=features.device,
                )
            else:
                flash = torch.as_tensor(
                    flash, dtype=features.dtype, device=features.device
                )
                flash = flash.view(features.shape[0], self.flash_dim)
            features = torch.cat([features, flash], dim=-1)

        hidden = self.body(features)
        values = self.head(hidden)
        
        return values, critic_rnn_states
