# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import OrderedDict
import numpy as np
from light_malib.utils.logger import Logger
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.base_env import BaseEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_table_name
from light_malib.utils.episode import EpisodeKey


class FlashTokenManager:
    def __init__(self, cfg, env):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.dim = int(self.cfg.get("dim", 0))
        if self.dim <= 0:
            self.enabled = False
        self.mode = str(self.cfg.get("mode", "off")).lower()
        self.std = float(self.cfg.get("std", 1.0))
        self.rng = np.random.default_rng(self.cfg.get("seed", None))
        self.num_players = getattr(env, "num_players", {})
        self.current = {}
        self.episode_base = None

    def _zeros(self):
        tokens = {}
        for agent_id, n in self.num_players.items():
            tokens[agent_id] = np.zeros((n, self.dim), dtype=np.float32) if self.dim > 0 else None
        return tokens

    def _sample_vector(self):
        return self.rng.normal(0.0, self.std, size=(1, self.dim)).astype(np.float32)

    def reset(self):
        self.episode_base = None
        if self.mode == "gaussian_episode" and self.enabled:
            self.episode_base = {
                agent_id: self._sample_vector() for agent_id in self.num_players
            }
        self.current = self._zeros() if not self.enabled else {}

    def sample(self):
        if not self.enabled:
            if not self.current:
                self.current = self._zeros()
            return self.current

        tokens = {}
        for agent_id, n in self.num_players.items():
            if self.mode == "gaussian_episode":
                base = self.episode_base[agent_id]
            else:  # gaussian_step or default
                base = self._sample_vector()
            tokens[agent_id] = np.repeat(base, n, axis=0)
        self.current = tokens
        return tokens

    def attach(self, step_data):
        tokens = self.sample()
        for agent_id, token in tokens.items():
            if token is None:
                step_data[agent_id].pop(EpisodeKey.FLASH_TOKEN, None)
            else:
                step_data[agent_id][EpisodeKey.FLASH_TOKEN] = token
        return step_data


class TacticalRewardComputer:
    """
    Computes dense tactical rewards for Supervisor training (Phase 2).
    
    Problem: Sparse goal rewards cause posterior collapse (supervisor picks same strategy).
    Solution: Add intermediate signals:
      - Ball progression: Did ball move toward enemy goal?
      - Possession: Did we keep the ball?
    
    These rewards are ONLY for supervisor, not player training.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        
        # Reward coefficients
        self.goal_coef = float(self.cfg.get("goal_coef", 1.0))
        self.progression_coef = float(self.cfg.get("progression_coef", 0.1))
        self.possession_coef = float(self.cfg.get("possession_coef", 0.05))
        
        # Track previous ball position for progression calculation
        self.prev_ball_x = None
        
        # GRF field: x goes from -1 (our goal) to +1 (enemy goal)
        # We want to reward moving the ball toward +1
        
    def reset(self):
        """Reset at episode start."""
        self.prev_ball_x = None
    
    def compute(self, obs, reward, info=None):
        """
        Compute dense tactical rewards.
        
        Args:
            obs: Current observation (should contain ball position)
            reward: Game reward (goals, etc.)
            info: Optional info dict from environment
        
        Returns:
            dict with:
              - supervisor_reward: Combined tactical reward
              - ball_progression: Ball movement toward enemy goal
              - possession_reward: Whether we have the ball
        """
        if not self.enabled:
            return {
                EpisodeKey.SUPERVISOR_REWARD: reward,
                EpisodeKey.BALL_PROGRESSION: 0.0,
                EpisodeKey.POSSESSION_REWARD: 0.0,
            }
        
        # Extract ball x position from observation
        # GRF observation layout varies, but ball is typically around index 88-89
        # Adjust indices based on your feature encoder
        ball_x = self._extract_ball_x(obs)
        
        # Ball progression: positive if ball moved toward enemy goal
        if self.prev_ball_x is not None and ball_x is not None:
            progression = ball_x - self.prev_ball_x  # +1 direction is enemy goal
            progression = np.clip(progression, -0.1, 0.1)  # Clip extreme values
        else:
            progression = 0.0
        
        self.prev_ball_x = ball_x
        
        # Possession: 1 if we have the ball, 0 otherwise
        # This can be extracted from observation or info
        possession = self._extract_possession(obs, info)
        
        # Combine rewards for supervisor
        supervisor_reward = (
            reward * self.goal_coef +
            progression * self.progression_coef +
            possession * self.possession_coef
        )
        
        return {
            EpisodeKey.SUPERVISOR_REWARD: float(supervisor_reward),
            EpisodeKey.BALL_PROGRESSION: float(progression),
            EpisodeKey.POSSESSION_REWARD: float(possession),
        }
    
    def _extract_ball_x(self, obs):
        """Extract ball x position from observation."""
        if obs is None:
            return None
        
        # Handle different observation shapes
        if isinstance(obs, dict):
            # Get from first agent
            obs = next(iter(obs.values()))
        
        if hasattr(obs, 'shape'):
            # Flatten if needed
            if obs.ndim > 1:
                obs = obs.flatten()
            
            # Ball x is typically at index 88 in GRF simple115 representation
            # Adjust this index based on your actual feature encoder
            if len(obs) > 88:
                return float(obs[88])
        
        return None
    
    def _extract_possession(self, obs, info=None):
        """Extract possession from observation or info."""
        # Option 1: From info dict (if available)
        if info is not None:
            if isinstance(info, dict):
                if 'ball_owned_team' in info:
                    return 1.0 if info['ball_owned_team'] == 0 else 0.0  # 0 = left team
        
        # Option 2: Infer from observation
        # In GRF, ball_owned_team might be encoded in the observation
        # This is a heuristic - adjust based on your feature encoder
        if obs is not None:
            if isinstance(obs, dict):
                obs = next(iter(obs.values()))
            if hasattr(obs, 'shape') and obs.ndim > 1:
                obs = obs.flatten()
            
            # Ball ownership might be around index 91 in simple115
            if len(obs) > 91:
                ownership = obs[91]
                if ownership > 0.5:  # Left team has ball
                    return 1.0
                elif ownership < -0.5:  # Right team has ball
                    return 0.0
                else:  # Ball is loose
                    return 0.5
        
        return 0.5  # Unknown, neutral value


class StrategyConditioningManager:
    """
    Manages strategy code sampling and intrinsic reward computation.
    
    Phase 1 (training_phase="phase1"):
      - Sample random c ~ Categorical(K) per episode
      - Compute intrinsic reward from discriminator
      - Players learn diverse behaviors conditioned on c
    
    Phase 2 (training_phase="phase2"):
      - Supervisor selects c = F_φ(s_global) every K steps (temporal abstraction)
      - Players are frozen, supervisor learns optimal strategy selection
      - No intrinsic reward (pure environment reward)
    
    Temporal Abstraction (Phase 2):
      - Supervisor acts at lower frequency (e.g., every 50 steps = ~3-5 sec in GRF)
      - Prevents "jitter" where strategies change too fast to manifest
      - Allows latent options time to unfold (e.g., players sprinting to press positions)
    
    The intrinsic reward (Phase 1 only):
      r_intrinsic = log q(c | trajectory_window)
    """
    
    def __init__(self, cfg, env, discriminator=None, supervisor=None, training_phase="phase1"):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.num_strategies = int(self.cfg.get("num_strategies", 8))
        self.intrinsic_reward_coef = float(self.cfg.get("intrinsic_reward_coef", 0.1))
        self.window_size = int(self.cfg.get("window_size", 16))
        self.rng = np.random.default_rng(self.cfg.get("seed", None))
        self.num_players = getattr(env, "num_players", {})
        
        # Discriminator for computing intrinsic rewards (Phase 1)
        self.discriminator = discriminator
        
        # Supervisor for strategy selection (Phase 2)
        self.supervisor = supervisor
        self.training_phase = training_phase
        
        # Temporal abstraction: supervisor acts every K steps (Phase 2)
        # Default 50 steps = ~3-5 seconds in GRF at 10Hz
        self.supervisor_interval = int(self.cfg.get("supervisor_interval", 50))
        self.steps_since_supervisor = 0
        
        # Current episode's strategy code
        self.current_strategy = None
        
        # Trajectory buffer for intrinsic reward computation
        self.obs_buffer = []
        self.action_buffer = []
        
        # Supervisor outputs (Phase 2)
        self.supervisor_log_prob = None
        self.supervisor_value = None
        
    def reset(self):
        """Reset at episode start: sample new strategy code (Phase 1 only)."""
        if not self.enabled:
            self.current_strategy = None
            return
        
        if self.training_phase == "phase1":
            # Phase 1: Sample random strategy for the episode
            self.current_strategy = self.rng.integers(0, self.num_strategies)
        else:
            # Phase 2: Supervisor will select strategy (with temporal abstraction)
            self.current_strategy = None
            self.steps_since_supervisor = self.supervisor_interval  # Force immediate selection on first step
        
        # Clear trajectory buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.supervisor_log_prob = None
        self.supervisor_value = None
    
    def should_select_strategy(self):
        """Check if supervisor should select a new strategy (temporal abstraction)."""
        if self.training_phase != "phase2":
            return False
        return self.steps_since_supervisor >= self.supervisor_interval
    
    def step(self):
        """Increment step counter for temporal abstraction (Phase 2 only)."""
        if self.training_phase == "phase2":
            self.steps_since_supervisor += 1
    
    def select_strategy_with_supervisor(self, global_obs, explore=True, device="cpu"):
        """
        Use supervisor to select strategy code (Phase 2).
        
        Implements temporal abstraction: resets step counter after selection.
        Strategy will be held for `supervisor_interval` steps.
        
        Args:
            global_obs: Pooled observation [obs_dim] or [batch, obs_dim]
            explore: Whether to sample or take argmax
            device: Torch device
        
        Returns:
            strategy_code: int
        """
        if self.supervisor is None:
            raise ValueError("Supervisor not available for Phase 2")
        
        import torch
        
        if isinstance(global_obs, np.ndarray):
            global_obs = torch.from_numpy(global_obs).float().to(device)
        
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)
        
        with torch.no_grad():
            strategy, log_prob, value, _ = self.supervisor(global_obs, explore=explore)
        
        self.current_strategy = strategy.item()
        self.supervisor_log_prob = log_prob.cpu().numpy()
        self.supervisor_value = value.cpu().numpy()
        
        # Reset counter - strategy will be held for supervisor_interval steps
        self.steps_since_supervisor = 0
        
        return self.current_strategy
        
    def get_strategy_code(self):
        """Get current strategy code as dict per agent."""
        if not self.enabled or self.current_strategy is None:
            return {}
        
        codes = {}
        for agent_id, n in self.num_players.items():
            # Same strategy code for all agents in the team
            codes[agent_id] = np.full((n,), self.current_strategy, dtype=np.int64)
        return codes
    
    def attach(self, step_data):
        """Attach strategy code to step data for all agents."""
        if not self.enabled:
            return step_data
        
        codes = self.get_strategy_code()
        for agent_id, code in codes.items():
            if agent_id in step_data:
                step_data[agent_id][EpisodeKey.STRATEGY_CODE] = code
                
                # Also attach supervisor outputs for Phase 2 training
                if self.training_phase == "phase2":
                    if self.supervisor_log_prob is not None:
                        step_data[agent_id][EpisodeKey.SUPERVISOR_LOG_PROB] = self.supervisor_log_prob
                    if self.supervisor_value is not None:
                        step_data[agent_id][EpisodeKey.SUPERVISOR_VALUE] = self.supervisor_value
        
        return step_data
    
    def update_buffer(self, obs, actions):
        """
        Update trajectory buffer with new observations and actions.
        
        Args:
            obs: dict[agent_id -> array of shape (num_agents, obs_dim)]
            actions: dict[agent_id -> array of shape (num_agents,)]
        """
        if not self.enabled:
            return
        
        # Stack observations from first agent (assuming shared policy)
        first_agent = next(iter(obs.keys()))
        self.obs_buffer.append(obs[first_agent])
        if actions:
            self.action_buffer.append(actions.get(first_agent, np.zeros((obs[first_agent].shape[0],))))
        
        # Keep only the last window_size steps
        if len(self.obs_buffer) > self.window_size:
            self.obs_buffer = self.obs_buffer[-self.window_size:]
            self.action_buffer = self.action_buffer[-self.window_size:]
    
    def compute_intrinsic_reward(self, device="cpu"):
        """
        Compute intrinsic reward based on discriminator's prediction (Phase 1 only).
        
        Uses asymmetric clamping to prevent reward explosion:
        - Full upside for correct predictions (discriminator confident)
        - Capped downside for confusion (discriminator unsure)
        
        Raw reward = log q(c|τ) - log(1/K)  (PMI: pointwise mutual information)
        
        Without clamping:
          - If discriminator is correct: log q ≈ 0 → reward ≈ +log(K)
          - If discriminator is confused: log q → -∞ → reward → -∞ (BAD!)
        
        With asymmetric clamping:
          - Upside: full range up to +log(K) (e.g., +2.08 for K=8)
          - Downside: capped at -0.5 (prevents value function destruction)
        
        Returns:
            intrinsic_rewards: dict[agent_id -> array of shape (num_agents, 1)]
        """
        # No intrinsic reward in Phase 2
        if self.training_phase != "phase1":
            return {}
        
        if not self.enabled or self.discriminator is None:
            return {}
        
        if len(self.obs_buffer) < self.window_size:
            # Not enough history yet
            return {}
        
        import torch
        
        # Prepare trajectory tensors
        # obs_buffer: list of (num_agents, obs_dim) arrays
        obs_seq = np.stack(self.obs_buffer[-self.window_size:], axis=0)  # (window, agents, obs_dim)
        action_seq = np.stack(self.action_buffer[-self.window_size:], axis=0)  # (window, agents)
        
        # Add batch dimension
        obs_seq = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)  # (1, window, agents, obs_dim)
        action_seq = torch.from_numpy(action_seq).long().unsqueeze(0).to(device)  # (1, window, agents)
        strategy = torch.tensor([self.current_strategy], dtype=torch.long, device=device)
        
        # Compute intrinsic reward with asymmetric clamping
        with torch.no_grad():
            # Get log probability of the actual active strategy
            log_q_c = self.discriminator.log_prob(obs_seq, action_seq, strategy)
            
            # Baseline: random chance = 1/K
            # For K=8: log_baseline = log(1/8) = -2.08
            log_baseline = torch.log(torch.tensor(1.0 / self.num_strategies, device=device))
            
            # Asymmetric clamping: cap the downside, allow full upside
            # Minimum log_q is baseline - 0.5, so minimum reward is -0.5
            # Maximum log_q is ~0 (perfect prediction), so max reward is +log(K)
            min_log_q = log_baseline - 0.5
            clamped_log_q = torch.max(log_q_c, min_log_q)
            
            # PMI-style reward: advantage over random baseline
            # Range: [-0.5, +log(K)] e.g., [-0.5, +2.08] for K=8
            raw_reward = clamped_log_q - log_baseline
            
            # Apply coefficient
            intrinsic_reward = raw_reward.item() * self.intrinsic_reward_coef
        
        # Return as dict with rewards for each agent
        rewards = {}
        for agent_id, n in self.num_players.items():
            rewards[agent_id] = np.full((n, 1), intrinsic_reward, dtype=np.float32)
        
        return rewards


def rename_fields(data, fields, new_fields):
    assert len(fields)==len(new_fields)
    for agent_id, agent_data in data.items():
        for field, new_field in zip(fields,new_fields):
            if field in agent_data:
                field_data = agent_data.pop(field)
                agent_data[new_field] = field_data
    return data


def select_fields(data, fields):
    rets = {
        agent_id: {field: agent_data[field] for field in fields if field in agent_data}
        for agent_id, agent_data in data.items()
    }
    return rets


def update_fields(data1, data2):
    def update_dict(dict1, dict2):
        d = {}
        d.update(dict1)
        d.update(dict2)
        return d

    rets = {
        agent_id: update_dict(data1[agent_id], data2[agent_id]) for agent_id in data1
    }
    return rets


def stack_step_data(step_data_list, bootstrap_data):
    episode_data = {}
    for field in step_data_list[0]:
        data_list = [step_data[field] for step_data in step_data_list]
        if field in bootstrap_data:
            data_list.append(bootstrap_data[field])
        try:
            episode_data[field] = np.stack(data_list)
        except Exception as e:
            import traceback
            Logger.error(traceback.format_exc())
            first_shape=data_list[0].shape
            for idx,data in enumerate(data_list):
                if data.shape!=first_shape:
                    Logger.error("field {}: first_shape: {}, mismatched_shape: {}, mismatched_idx: {}".format(field,first_shape,data.shape,idx))
                    break
            raise e
    return episode_data


def credit_reassign(episode, info, reward_reassignment_cfg, s_idx, e_idx):
    _reward = episode["reward"]
    goal_info = info["goal_info"]
    assist_info = info["assist_info"]
    loseball_info = info["loseball_info"]
    halt_loseball_info = info["halt_loseball_info"]
    gainball_info = info["gainball_info"]
    tag_dict = {
        "goal": goal_info,
        "assist": assist_info,
        "loseball": loseball_info,
        "gainball": gainball_info,
    }
    for tag, i in tag_dict.items():
        for value in i:
            t_idx = value["t"] - s_idx
            if 0 <= t_idx < len(_reward):
                # if s_idx<=value['t']<=e_idx:
                #     t_idx = value['t']-s_idx
                player_idx = value["player"] - 1
                _reward[t_idx, player_idx, 0] += reward_reassignment_cfg[tag]

    episode["reward"] = _reward
    return episode

def pull_policies(rollout_worker,policy_ids):
    rollout_worker.pull_policies(policy_ids)
    behavior_policies = rollout_worker.get_policies(policy_ids)
    return behavior_policies

def env_reset(env, behavior_policies, custom_reset_config):
    global_timer.record("env_step_start")
    env_rets = env.reset(custom_reset_config)
    global_timer.time("env_step_start", "env_step_end", "env_step")

    init_rnn_states = {
        agent_id: behavior_policies[agent_id][1].get_initial_state(
            batch_size=env.num_players[agent_id]
        )
        for agent_id in env.agent_ids
    }

    step_data = update_fields(env_rets, init_rnn_states)
    return step_data

def submit_traj(data_server,step_data_list,last_step_data,rollout_desc,s_idx=None,e_idx=None,credit_reassign_cfg=None,assist_info=None):
    bootstrap_data = select_fields(
        last_step_data,
        [
            EpisodeKey.NEXT_OBS,
            EpisodeKey.DONE,
            EpisodeKey.CRITIC_RNN_STATE,
            EpisodeKey.NEXT_STATE,
            EpisodeKey.FLASH_TOKEN,
            EpisodeKey.STRATEGY_CODE,
        ],
    )
    bootstrap_data = rename_fields(bootstrap_data, [EpisodeKey.NEXT_OBS,EpisodeKey.NEXT_STATE], [EpisodeKey.CUR_OBS,EpisodeKey.CUR_OBS])
    bootstrap_data = bootstrap_data[rollout_desc.agent_id]
    
    _episode = stack_step_data(
        step_data_list[s_idx:e_idx],
        # TODO CUR_STATE is not supported now
        bootstrap_data,
    )

    if credit_reassign_cfg is not None and assist_info is not None:
        episode = credit_reassign(
            _episode,
            assist_info,
            credit_reassign_cfg,
            s_idx,
            e_idx,
        )
    else:
        episode = _episode

    # submit data:
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            [episode],
        )
    else:
        data_server.save(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            [episode],
        )

def submit_batches(data_server,episode, rollout_desc,credit_reassign_cfg=None,assist_info=None):
    transitions = []
    for step in range(len(episode) - 1):
        transition = {
            EpisodeKey.CUR_OBS: episode[step][EpisodeKey.CUR_OBS],  # [np.newaxis, ...],
            EpisodeKey.ACTION_MASK: episode[step][EpisodeKey.ACTION_MASK],  # [np.newaxis, ...],
            EpisodeKey.ACTION: episode[step][EpisodeKey.ACTION],  # [np.newaxis, ...],
            EpisodeKey.REWARD: episode[step][EpisodeKey.REWARD],  # [np.newaxis, ...],
            EpisodeKey.DONE: episode[step][EpisodeKey.DONE],  # [np.newaxis, ...],
            EpisodeKey.NEXT_OBS: episode[step + 1][EpisodeKey.CUR_OBS],  # [np.newaxis, ...],
            EpisodeKey.NEXT_ACTION_MASK: episode[step + 1][EpisodeKey.ACTION_MASK],  # [np.newaxis, ...]
            EpisodeKey.CRITIC_RNN_STATE: episode[step][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.NEXT_CRITIC_RNN_STATE: episode[step + 1][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.GLOBAL_STATE: episode[step][EpisodeKey.GLOBAL_STATE],
            EpisodeKey.NEXT_GLOBAL_STATE: episode[step + 1][EpisodeKey.GLOBAL_STATE],
            EpisodeKey.FLASH_TOKEN: episode[step].get(EpisodeKey.FLASH_TOKEN),
        }
        transitions.append(transition)
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            transitions
        )
    else:
        data_server.save(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            transitions
        )




def rollout_func(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: BaseEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs
):
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """

    sample_length = kwargs.get("sample_length", rollout_length)
    render = kwargs.get("render", False)
    if render:
        env.render()

    episode_mode = kwargs.get('episode_mode','traj')
    record_value = kwargs.get("record_value", False)
    if record_value:
        value_list = []
        pos_list = []

    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policy.eval()

    custom_reset_config = {
        "feature_encoders": feature_encoders,
        "main_agent_id": rollout_desc.agent_id,
        "rollout_length": rollout_length,
    }

    step_data = env_reset(env,behavior_policies,custom_reset_config)

    flash_manager = FlashTokenManager(getattr(rollout_worker, "flash_cfg", None), env)
    flash_manager.reset()
    
    # Strategy conditioning for Phase 1/2 training
    strategy_cfg = getattr(rollout_worker, "strategy_cfg", None)
    training_phase = strategy_cfg.get("training_phase", "phase1") if strategy_cfg else "phase1"
    
    # Get discriminator and supervisor from the main agent's policy
    main_policy = behavior_policies.get(rollout_desc.agent_id, (None, None))[1]
    discriminator = getattr(main_policy, "discriminator", None) if main_policy else None
    supervisor = getattr(main_policy, "supervisor", None) if main_policy else None
    
    strategy_manager = StrategyConditioningManager(
        strategy_cfg, env, 
        discriminator=discriminator,
        supervisor=supervisor,
        training_phase=training_phase
    )
    strategy_manager.reset()
    
    # Tactical reward computer for supervisor (Phase 2 dense rewards)
    tactical_reward_cfg = strategy_cfg.get("tactical_rewards", {}) if strategy_cfg else {}
    tactical_reward_computer = TacticalRewardComputer(tactical_reward_cfg)
    tactical_reward_computer.reset()

    step = 0
    step_data_list = []
    results = []
    # collect until rollout_length
    while step <= rollout_length:
        step_data = flash_manager.attach(step_data)
        
        # Phase 2: Use supervisor to select strategy (with temporal abstraction)
        # Supervisor only acts every supervisor_interval steps to prevent "jitter"
        if (strategy_manager.enabled and 
            strategy_manager.training_phase == "phase2" and 
            strategy_manager.supervisor is not None and
            strategy_manager.should_select_strategy()):
            # Get pooled global observation for supervisor
            first_agent = next(iter(step_data.keys()))
            global_obs = step_data[first_agent].get(EpisodeKey.NEXT_OBS, step_data[first_agent].get(EpisodeKey.CUR_OBS))
            if global_obs is not None:
                # Pool across agents: take mean
                pooled_obs = global_obs.mean(axis=0) if global_obs.ndim > 1 else global_obs
                strategy_manager.select_strategy_with_supervisor(
                    pooled_obs, 
                    explore=not eval,
                    device=main_policy.device if main_policy else "cpu"
                )
        
        step_data = strategy_manager.attach(step_data)
        
        # Increment temporal abstraction counter
        strategy_manager.step()
        
        # prepare policy input
        policy_inputs = rename_fields(step_data, [EpisodeKey.NEXT_OBS,EpisodeKey.NEXT_STATE], [EpisodeKey.CUR_OBS,EpisodeKey.CUR_OBS])
        for agent_id in policy_inputs:
            if EpisodeKey.FLASH_TOKEN in step_data[agent_id]:
                policy_inputs[agent_id][EpisodeKey.FLASH_TOKEN] = step_data[agent_id][EpisodeKey.FLASH_TOKEN]
            if EpisodeKey.STRATEGY_CODE in step_data[agent_id]:
                policy_inputs[agent_id][EpisodeKey.STRATEGY_CODE] = step_data[agent_id][EpisodeKey.STRATEGY_CODE]
        policy_outputs = {}
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            policy_outputs[agent_id] = policy.compute_action(
                inference=True, 
                explore=not eval,
                to_numpy=True,
                step = kwargs.get('rollout_epoch', 0),
                **policy_inputs[agent_id]
            )
            if record_value and agent_id == "agent_0":
                value_list.append(policy_outputs[agent_id][EpisodeKey.STATE_VALUE])
                pos_list.append(policy_inputs[agent_id][EpisodeKey.CUR_OBS][:, 114:116])

        global_timer.time("inference_start", "inference_end", "inference")

        actions = select_fields(policy_outputs, [EpisodeKey.ACTION])

        global_timer.record("env_step_start")
        env_rets = env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")
        
        # Update strategy manager's trajectory buffer for intrinsic reward
        if strategy_manager.enabled:
            obs_for_buffer = {
                agent_id: policy_inputs[agent_id][EpisodeKey.CUR_OBS]
                for agent_id in policy_inputs
            }
            actions_for_buffer = {
                agent_id: policy_outputs[agent_id][EpisodeKey.ACTION]
                for agent_id in policy_outputs
            }
            strategy_manager.update_buffer(obs_for_buffer, actions_for_buffer)
            
            # Compute and add intrinsic rewards (Phase 1 training only)
            if not eval:
                intrinsic_rewards = strategy_manager.compute_intrinsic_reward()
                for agent_id, intrinsic_r in intrinsic_rewards.items():
                    if agent_id in env_rets and EpisodeKey.REWARD in env_rets[agent_id]:
                        # Add intrinsic reward to environment reward
                        env_rets[agent_id][EpisodeKey.REWARD] = (
                            env_rets[agent_id][EpisodeKey.REWARD] + intrinsic_r
                        )
                        # Also store intrinsic reward separately for logging
                        env_rets[agent_id][EpisodeKey.INTRINSIC_REWARD] = intrinsic_r
        
        # Compute dense tactical rewards for supervisor (Phase 2)
        # Prevents posterior collapse from sparse goal rewards
        if strategy_manager.enabled and strategy_manager.training_phase == "phase2":
            first_agent = next(iter(env_rets.keys()))
            game_reward = env_rets[first_agent].get(EpisodeKey.REWARD, np.zeros((1,)))
            if hasattr(game_reward, 'mean'):
                game_reward = float(game_reward.mean())
            else:
                game_reward = float(game_reward)
            
            # Get observation for tactical reward computation
            obs_for_tactical = policy_inputs.get(first_agent, {}).get(EpisodeKey.CUR_OBS)
            
            tactical_rewards = tactical_reward_computer.compute(
                obs=obs_for_tactical,
                reward=game_reward,
            )
            
            # Attach supervisor reward to step data for Phase 2 training
            for agent_id in env_rets:
                env_rets[agent_id][EpisodeKey.SUPERVISOR_REWARD] = tactical_rewards[EpisodeKey.SUPERVISOR_REWARD]
                env_rets[agent_id][EpisodeKey.BALL_PROGRESSION] = tactical_rewards[EpisodeKey.BALL_PROGRESSION]
                env_rets[agent_id][EpisodeKey.POSSESSION_REWARD] = tactical_rewards[EpisodeKey.POSSESSION_REWARD]

        # record data after env step
        step_data = update_fields(
            step_data, select_fields(env_rets, [EpisodeKey.REWARD, EpisodeKey.DONE])
        )
        step_data = update_fields(
            step_data,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTION, EpisodeKey.ACTION_LOG_PROB, EpisodeKey.STATE_VALUE],
            ),
        )

        # save data of trained agent for training
        step_data_list.append(step_data[rollout_desc.agent_id])

        # record data for next step
        step_data = update_fields(
            env_rets,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE],
            ),
        )

        step += 1
        
        ##### submit samples to server #####
        # used for the full game
        if not eval:
            if episode_mode == 'traj':
                # used for on-policy algorithms
                assert data_server is not None            
                if sample_length > 0 and step % sample_length == 0:
                    assist_info = env.get_AssistInfo()

                    submit_ctr = step // sample_length
                    submit_max_num = rollout_length // sample_length

                    s_idx = sample_length * (submit_ctr - 1)
                    e_idx = sample_length * submit_ctr

                    submit_traj(data_server,step_data_list,step_data,rollout_desc,s_idx,e_idx,
                                credit_reassign_cfg=kwargs.get("credit_reassign_cfg"),
                                assist_info=assist_info)   

                    if submit_ctr != submit_max_num:
                        # update model:
                        behavior_policies=pull_policies(rollout_worker,policy_ids)
                    
            # elif episode_mode == 'time-step':
            #     # used for off-policy algorithms
            #     episode = step_data_list
            #     transitions = []
            #     for step in range(len(episode)-1):
            #         transition = {
            #             EpisodeKey.CUR_OBS: episode[step][EpisodeKey.CUR_OBS][np.newaxis, ...],
            #             EpisodeKey.ACTION_MASK: episode[step][EpisodeKey.ACTION_MASK][np.newaxis, ...],
            #             EpisodeKey.ACTION: episode[step][EpisodeKey.ACTION][np.newaxis, ...],
            #             EpisodeKey.REWARD: episode[step][EpisodeKey.REWARD][np.newaxis, ...],
            #             EpisodeKey.DONE: episode[step][EpisodeKey.DONE][np.newaxis, ...],
            #             EpisodeKey.NEXT_OBS: episode[step + 1][EpisodeKey.CUR_OBS][np.newaxis, ...],
            #             EpisodeKey.NEXT_ACTION_MASK: episode[step + 1][EpisodeKey.ACTION_MASK][np.newaxis, ...]
            #         }
            #         transitions.append(transition)
            #     data_server.save.remote(
            #         default_table_name(
            #             rollout_desc.agent_id,
            #             rollout_desc.policy_id,
            #             rollout_desc.share_policies,
            #         ),
            #         transitions
            #     )
                    
        ##### check if  env ends #####
        if env.is_terminated():
            stats = env.get_episode_stats()
            if record_value:
                result = {
                    "main_agent_id": rollout_desc.agent_id,
                    "policy_ids": policy_ids,
                    "stats": stats,
                    "value": value_list,
                    "pos": pos_list,
                    "assist_info": assist_info,
                }
            else:
                result = {
                    "main_agent_id": rollout_desc.agent_id,
                    "policy_ids": policy_ids,
                    "stats": stats,
                }
            results.append(result)
            
            # reset env
            step_data = env_reset(env,behavior_policies,custom_reset_config)
            flash_manager.reset()
            strategy_manager.reset()
            tactical_reward_computer.reset()
    
    if not eval and sample_length <= 0:            #collect after rollout done
        # used for the academy
        if episode_mode == 'traj':
            submit_traj(data_server,step_data_list,step_data,rollout_desc)
        elif episode_mode == 'time-step':
            submit_batches(data_server, step_data_list, rollout_desc)



    results={"results":results}            
    return results
