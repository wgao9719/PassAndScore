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

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete

from light_malib.algorithm.utils import init_fc_weights
from light_malib.utils.preprocessor import get_preprocessor

# Use the standard GRF FeatureEncoder for encoding State objects
from light_malib.envs.gr_football.encoders.encoder_basic import FeatureEncoder


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
    Factorized actor for multi-agent PPO.
    
    CTDE Mode (factorized_actor_use_global=False):
      - Input: local observations only
      - Each agent acts based only on its own observation
      
    Centralized Mode (factorized_actor_use_global=True):
      - Input: local + global (pooled) observations
      - Each agent sees team-level information
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

        self.body, last_dim = _build_mlp(
            input_dim, model_config.get("layers", []), initialization
        )
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
    ):
        local = observations["local"]
        
        if self.use_global:
            features = torch.cat([local, observations["global"]], dim=-1)
        else:
            features = local

        hidden = self.body(features)
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

        hidden = self.body(features)
        values = self.head(hidden)
        
        return values, critic_rnn_states
