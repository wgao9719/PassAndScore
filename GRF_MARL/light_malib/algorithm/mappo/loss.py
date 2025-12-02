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

from typing import Union
import torch
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.logger import Logger
from light_malib.registry import registry
import numpy as np

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return (e**2) / 2


def to_value(tensor: torch.Tensor):
    return tensor.detach().cpu().item()


def basic_stats(name, tensor: torch.Tensor):
    stats = {}
    stats["{}_max".format(name)] = to_value(tensor.max())
    stats["{}_min".format(name)] = to_value(tensor.min())
    stats["{}_mean".format(name)] = to_value(tensor.mean())
    stats["{}_std".format(name)] = to_value(tensor.std())
    return stats


@registry.registered(registry.LOSS)
class MAPPOLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super(MAPPOLoss, self).__init__()

        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        if self._use_huber_loss:
            self.huber_delta = 10.0
        self._use_max_grad_norm = True

    def reset(self, policy, config):
        """
        reset should always be called for each training task.
        """
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()
        
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

        self.sub_algorithm_name = policy.custom_config.get("sub_algorithm_name","MAPPO")   
        assert self.sub_algorithm_name in ["MAPPO","CoPPO","HAPPO","A2PO"]
        
        if self.sub_algorithm_name in ["IPPO","MAPPO"]:
            self._use_seq=False
            self._use_two_stage=False
            self._use_co_ma_ratio=False
            self._clip_before_prod=False
            self._clip_others=False
        elif self.sub_algorithm_name=="CoPPO":
            self._use_seq=False
            self._use_two_stage=False
            self._use_co_ma_ratio=True
            self._clip_before_prod=True
            self._clip_others=True
            self._other_clip_param=policy.custom_config["other_clip_param"]
            self._num_agents=policy.custom_config["num_agents"]
        elif self.sub_algorithm_name=="HAPPO":
            self._use_seq=True
            self._use_two_stage=False
            self._use_co_ma_ratio=True
            self._clip_before_prod=True
            self._clip_others=False
            self._num_agents=policy.custom_config["num_agents"]
            self._seq_strategy=policy.custom_config.get("seq_strategy","random")
            # TODO(jh): check default
            self._one_agent_per_update=False
            self._use_agent_block=policy.custom_config.get("use_agent_block",False)
            if self._use_agent_block:
                self._block_num=policy.custom_config["block_num"]
            self._use_cum_sequence=True
            self._agent_seq=[]
        elif self.sub_algorithm_name=="A2PO":
            self._use_seq=True
            self._use_two_stage=True
            self._use_co_ma_ratio=True
            self._clip_before_prod=False
            self._clip_others=True
            self._other_clip_param=policy.custom_config["other_clip_param"]
            self._num_agents=policy.custom_config["num_agents"]
            self._seq_strategy=policy.custom_config.get("seq_strategy","semi_greedy")
            # TODO(jh): check default
            self._one_agent_per_update=False
            self._use_agent_block=policy.custom_config.get("use_agent_block",False)
            if self._use_agent_block:
                self._block_num=policy.custom_config["block_num"]
            self._use_cum_sequence=True
            self._agent_seq=[]
        else:
            raise NotImplementedError     
            
    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""
        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))
        
        # TODO(jh): update actor and critic simutaneously
        param_groups=[]
        
        if len(list(self._policy.actor.parameters()))>0:
            param_groups.append({'params': self.policy.actor.parameters(), 'lr': self._params["actor_lr"]})
        
        if len(list(self._policy.critic.parameters()))>0:
            param_groups.append({'params': self.policy.critic.parameters(), 'lr': self._params["critic_lr"]})
        
        if self._policy.share_backbone and len(list(self._policy.backbone.parameters()))>0:
            param_groups.append({'params': self.policy.backbone.parameters(), 'lr': self._params["backbone_lr"]})
            
        self.optimizer=optim_cls(
            param_groups,
            eps=self._params["opti_eps"],
            weight_decay=self._params["weight_decay"]
        )
        
        self.optimizer.zero_grad()
        
        # Discriminator optimizer for Phase 1 strategy conditioning
        self.discriminator_optimizer = None
        if self._policy.discriminator is not None:
            discriminator_lr = self._params.get("discriminator_lr", self._params["actor_lr"])
            self.discriminator_optimizer = optim_cls(
                self._policy.discriminator.parameters(),
                lr=discriminator_lr,
                eps=self._params["opti_eps"],
                weight_decay=self._params["weight_decay"]
            )
            self.discriminator_optimizer.zero_grad()
        
        # Supervisor optimizer for Phase 2 training
        self.supervisor_optimizer = None
        if self._policy.supervisor is not None:
            supervisor_lr = self._params.get("supervisor_lr", self._params["actor_lr"])
            self.supervisor_optimizer = optim_cls(
                self._policy.supervisor.parameters(),
                lr=supervisor_lr,
                eps=self._params["opti_eps"],
                weight_decay=self._params["weight_decay"]
            )
            self.supervisor_optimizer.zero_grad()
        
        self.n_opt_steps=0
        self.grad_accum_step=self._params.get("grad_accum_step",1)
        
    def loss_compute(self, sample):
        self.n_opt_steps+=1
        
        policy = self._policy
        policy.train()
        
        # Check training phase
        training_phase = policy.custom_config.get("training_phase", "normal")
        
        if training_phase == "phase2":
            # Phase 2: Train supervisor only, players are frozen
            return self.loss_compute_supervisor(sample)
        elif self._use_seq:
            return self.loss_compute_sequential(sample)
        else:
            return self.loss_compute_simultaneous(sample)
            
    def _select_data_from_agent_ids(
        self,
        x: Union[np.ndarray, torch.Tensor],
        agent_ids: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        '''
        we assume x is the shape [#batch_size*#agents,...]
        '''
        if agent_ids is None:
            return x        
        
        if not isinstance(x,(np.ndarray,torch.Tensor)):
            return x
        
        x = x.reshape(-1, self._num_agents, *x.shape[1:])[:, agent_ids]
        x = x.reshape(-1,*x.shape[2:])
        return x

    def loss_compute_simultaneous(
        self, 
        sample,
        agent_ids:np.ndarray=None,
        update_actor:bool=True
    ):
        # agent_ids not None means block update
        if agent_ids is not None:
            assert len(agent_ids.shape)==1
        
        (
            share_obs_batch,
            obs_batch,
            flash_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            available_actions_batch,
            actor_rnn_states_batch,
            critic_rnn_states_batch,
            dones_batch,
            adv_targ,
            delta,
        ) = (
            sample[EpisodeKey.CUR_STATE],
            sample[EpisodeKey.CUR_OBS],
            sample.get(EpisodeKey.FLASH_TOKEN, None),
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.RETURN],
            sample.get(EpisodeKey.ACTIVE_MASK, None),
            sample[EpisodeKey.ACTION_LOG_PROB],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTOR_RNN_STATE],
            sample[EpisodeKey.CRITIC_RNN_STATE],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.ADVANTAGE],
            sample["delta"],
        )
        
        # Strategy conditioning data (Phase 1)
        strategy_code_batch = sample.get(EpisodeKey.STRATEGY_CODE, None)

        if update_actor:
            action_kwargs = {
                EpisodeKey.CUR_STATE: share_obs_batch,
                EpisodeKey.CUR_OBS: obs_batch,
                EpisodeKey.ACTION: actions_batch,
                EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states_batch,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states_batch,
                EpisodeKey.DONE: dones_batch,
                EpisodeKey.ACTION_MASK: available_actions_batch,
            }
            if flash_batch is not None:
                action_kwargs[EpisodeKey.FLASH_TOKEN] = flash_batch
            if strategy_code_batch is not None:
                action_kwargs[EpisodeKey.STRATEGY_CODE] = strategy_code_batch
            ret = self._policy.compute_action(
                **action_kwargs,
                inference=False,
                explore=False
            )
            
            values=ret[EpisodeKey.STATE_VALUE]
            action_log_probs=ret[EpisodeKey.ACTION_LOG_PROB]
            dist_entropy=ret[EpisodeKey.ACTION_ENTROPY]     
            
             # ============================== Policy Loss ================================
            imp_weights = torch.exp(
                action_log_probs - old_action_log_probs_batch
            ).view(-1,1)
            approx_kl = (
                (old_action_log_probs_batch - action_log_probs).mean().item()
            )
        
            # CoPPO, HAPPO, A2PO
            if self._use_co_ma_ratio:
                each_agent_imp_weights = imp_weights.reshape(
                    -1, self._num_agents, 1
                )
                # NOTE(jh): important to detach, so gradients won't flow back from other agents' policy update
                each_agent_imp_weights = each_agent_imp_weights.detach()
                
                mask_self = torch.eye(self._num_agents,device=each_agent_imp_weights.device,dtype=torch.bool)

                if self.sub_algorithm_name != 'CoPPO':
                    mask_self = mask_self[agent_ids]
                
                # (#selected_agents,#agents,1)
                mask_self = mask_self.unsqueeze(-1)
                
                # (#batch,1,#agents,1)
                each_agent_imp_weights = each_agent_imp_weights.unsqueeze(1)
                # (#batch,#selected_agents,#agents,1)
                if agent_ids is None:
                    repeats=self._num_agents
                else:
                    repeats=len(agent_ids)
                each_agent_imp_weights = each_agent_imp_weights.repeat_interleave(repeats,dim=1)
                each_agent_imp_weights[..., mask_self] = 1.0
                
                # (#batch,#selected_agents,1)
                other_agents_prod_imp_weights = each_agent_imp_weights.prod(dim=2)
                
                # CoPPO, A2PO
                if self._clip_others:
                    other_agents_prod_imp_weights = torch.clamp(
                        other_agents_prod_imp_weights,
                        1.0-self._other_clip_param,
                        1.0+self._other_clip_param
                    )

                other_agents_prod_imp_weights = other_agents_prod_imp_weights.reshape(-1, 1)
                
            imp_weights = self._select_data_from_agent_ids(imp_weights, agent_ids)
            adv_targ = self._select_data_from_agent_ids(adv_targ, agent_ids)
            active_masks_batch = self._select_data_from_agent_ids(active_masks_batch,agent_ids)
            dist_entropy = self._select_data_from_agent_ids(dist_entropy, agent_ids)
            
            # CoPPO, A2PO
            if self._use_co_ma_ratio and not self._clip_before_prod:
                imp_weights = imp_weights * other_agents_prod_imp_weights
        
            surr1 = imp_weights * adv_targ
            surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
            )
            
            # HAPPO
            if self._use_co_ma_ratio and self._clip_before_prod:
                surr1 = surr1 * other_agents_prod_imp_weights
                surr2 = surr2 * other_agents_prod_imp_weights

            if active_masks_batch is not None:
                surr = torch.min(surr1, surr2)
                policy_action_loss = (
                    -torch.sum(surr, dim=-1, keepdim=True) * active_masks_batch
                ).sum() / (active_masks_batch.sum()+1e-20)
                assert dist_entropy.shape==active_masks_batch.shape
                policy_entropy_loss = - (dist_entropy*active_masks_batch).sum()/(active_masks_batch.sum()+1e-20)
            else:
                surr = torch.min(surr1, surr2)
                policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()
                policy_entropy_loss = - dist_entropy.mean()

            policy_loss = policy_action_loss + policy_entropy_loss * self._policy.custom_config["entropy_coef"]

        else:
            ret = self._policy.value_function(
                **{
                    EpisodeKey.CUR_STATE: share_obs_batch,
                    EpisodeKey.CUR_OBS: obs_batch,
                    EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states_batch,
                    EpisodeKey.DONE: dones_batch
                },
                inference=False
            )
            values=ret[EpisodeKey.STATE_VALUE]
            
            policy_loss = 0
            active_masks_batch = self._select_data_from_agent_ids(active_masks_batch, agent_ids)
        
        # ============================== Value Loss ================================
       
        values = self._select_data_from_agent_ids(values, agent_ids)
        value_preds_batch = self._select_data_from_agent_ids(value_preds_batch, agent_ids)
        return_batch = self._select_data_from_agent_ids(return_batch, agent_ids)
       
        value_loss = self._calc_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        # ============================== Total Loss ================================        
        total_loss = policy_loss + value_loss * self._policy.custom_config.get("value_loss_coef",1.0)
        
        total_loss = total_loss/self.grad_accum_step

        # ============================== Optimizer ================================
        total_loss.backward()        
        if self.n_opt_steps%self.grad_accum_step==0:     
            if self._use_max_grad_norm:
                for param_group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.max_grad_norm
                    )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # ============================== Discriminator Update (Phase 1) ================================
        # NOTE: Discriminator is now updated via update_discriminator_with_trajectory() 
        # which is called from the trainer with the FULL trajectory batch.
        # Mini-batched data loses temporal structure needed for discriminator.
        # Stats are added there instead.

        # ============================== Statistics ================================
        if update_actor:
            # TODO(jh): miss active masks?
            stats = dict(
                ratio=float(imp_weights.detach().mean().cpu().numpy()),
                ratio_std=float(imp_weights.detach().std().cpu().numpy()),
                policy_loss=float(policy_loss.detach().cpu().numpy()),
                value_loss=float(value_loss.detach().cpu().numpy()),
                entropy=float(dist_entropy.detach().mean().cpu().numpy()),
                approx_kl=approx_kl,
            )

            stats.update(basic_stats("imp_weights", imp_weights))
            stats.update(basic_stats("advantages", adv_targ))
            stats.update(basic_stats("V", values))
            stats.update(basic_stats("Old_V", value_preds_batch))
            stats.update(basic_stats("delta", delta))

            stats["upper_clip_ratio"] = to_value(
                (imp_weights > (1 + self.clip_param)).float().mean()
            )
            stats["lower_clip_ratio"] = to_value(
                (imp_weights < (1 - self.clip_param)).float().mean()
            )
            stats["clip_ratio"] = stats["upper_clip_ratio"] + stats["lower_clip_ratio"]
            
            # NOTE: Discriminator stats are now added by update_discriminator_with_trajectory()
            # which is called from the trainer with full trajectory data.
        else:
            stats = {}
            
        return stats
    
    def _update_discriminator(self, sample, strategy_code_batch):
        """
        Update the discriminator to predict strategy code from trajectory.
        
        The discriminator learns: q(c | trajectory) 
        This provides the intrinsic reward gradient for the policy.
        
        Returns:
            loss: float, the discriminator loss
            accuracy: float, the classification accuracy
        """
        if self._policy.discriminator is None:
            return None, None
        
        # Get observations and actions from the sample
        # Sample shape is typically [T, batch*agents, dim] for trajectories
        obs_batch = sample[EpisodeKey.CUR_OBS]
        actions_batch = sample[EpisodeKey.ACTION].long()
        
        # Reshape for discriminator: need [batch, window, agents, dim]
        # The sample is from a trajectory, so we can use consecutive steps
        num_agents = self._policy.custom_config.get("num_agents", 4)
        window_size = self._policy.custom_config.get("discriminator_window_size", 16)
        
        # Flatten batch dimension and reshape
        # obs_batch: [T, batch*agents, obs_dim]
        T = obs_batch.shape[0] if len(obs_batch.shape) > 2 else 1
        
        # Debug: log shapes on first call
        if self.n_opt_steps == 0:
            Logger.warning(f"[Discriminator] obs_batch shape: {obs_batch.shape}, T={T}, window_size={window_size}")
            Logger.warning(f"[Discriminator] strategy_code_batch shape: {strategy_code_batch.shape if strategy_code_batch is not None else 'None'}")
        
        if T < window_size:
            # Not enough trajectory steps, skip discriminator update
            if self.n_opt_steps == 0:
                Logger.warning(f"[Discriminator] T={T} < window_size={window_size}, SKIPPING update!")
            return None, None
        
        # For simplicity, use the last window_size steps
        # In practice, you might want to sample multiple windows
        if len(obs_batch.shape) == 3:
            # [T, batch*agents, obs_dim]
            batch_agents = obs_batch.shape[1]
            batch_size = batch_agents // num_agents
            
            # Take last window_size steps
            obs_window = obs_batch[-window_size:]  # [window, batch*agents, obs_dim]
            action_window = actions_batch[-window_size:]  # [window, batch*agents]
            
            # Reshape to [batch, window, agents, dim]
            obs_window = obs_window.permute(1, 0, 2)  # [batch*agents, window, obs_dim]
            obs_window = obs_window.reshape(batch_size, num_agents, window_size, -1)
            obs_window = obs_window.permute(0, 2, 1, 3)  # [batch, window, agents, obs_dim]
            
            action_window = action_window.permute(1, 0)  # [batch*agents, window]
            action_window = action_window.reshape(batch_size, num_agents, window_size)
            action_window = action_window.permute(0, 2, 1)  # [batch, window, agents]
            
            # Strategy code should be same for all agents, take first agent's
            if len(strategy_code_batch.shape) == 2:
                # [T, batch*agents]
                strategy = strategy_code_batch[0].reshape(batch_size, num_agents)[:, 0]
            else:
                strategy = strategy_code_batch.reshape(-1, num_agents)[:, 0]
        else:
            # Fallback for different shapes
            return None, None
        
        # Compute discriminator loss
        loss, accuracy = self._policy.discriminator.compute_loss(
            obs_window, action_window, strategy
        )
        
        # Update discriminator
        loss_scaled = loss / self.grad_accum_step
        loss_scaled.backward()
        
        if self.n_opt_steps % self.grad_accum_step == 0:
            if self._use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self._policy.discriminator.parameters(), self.max_grad_norm
                )
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad()
        
        loss_val = float(loss.detach().cpu().numpy())
        acc_val = float(accuracy.detach().cpu().numpy())
        
        # Log discriminator stats periodically
        if self.n_opt_steps % 50 == 0:
            Logger.info(f"[Discriminator] step={self.n_opt_steps}, loss={loss_val:.4f}, accuracy={acc_val:.4f}")
        
        return loss_val, acc_val
    
    def update_discriminator_with_trajectory(self, full_batch):
        """
        Update discriminator using the FULL trajectory batch (before mini-batching).
        
        This is called once per training epoch from the trainer, because:
        - The discriminator needs temporal structure (window of consecutive steps)
        - Mini-batching flattens/shuffles this structure
        
        Args:
            full_batch: The complete trajectory batch with shape [T, batch, agents, dim]
        
        Returns:
            dict with discriminator_loss and discriminator_accuracy, or empty dict
        """
        # Skip discriminator update in Phase 2 (players/discriminator frozen)
        training_phase = self._policy.custom_config.get("training_phase", "normal")
        if training_phase == "phase2":
            return {}
        
        if self._policy.discriminator is None or self.discriminator_optimizer is None:
            return {}
        
        strategy_code_batch = full_batch.get(EpisodeKey.STRATEGY_CODE, None)
        if strategy_code_batch is None:
            Logger.warning("[Discriminator] No STRATEGY_CODE in batch, skipping update")
            return {}
        
        # Get observations and actions - keep trajectory structure!
        # full_batch shape: [T+1, batch, agents, dim] 
        obs_batch = full_batch[EpisodeKey.CUR_OBS]
        actions_batch = full_batch[EpisodeKey.ACTION]
        
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.from_numpy(obs_batch).float().to(self._policy.device)
        if isinstance(actions_batch, np.ndarray):
            actions_batch = torch.from_numpy(actions_batch).long().to(self._policy.device)
        if isinstance(strategy_code_batch, np.ndarray):
            strategy_code_batch = torch.from_numpy(strategy_code_batch).long().to(self._policy.device)
        
        # Shape: [B, T+1, N, obs_dim] where B=batch, T+1=traj_length+1, N=num_agents
        # This is the standard shape from return_compute
        if len(obs_batch.shape) != 4:
            Logger.warning(f"[Discriminator] Unexpected obs shape: {obs_batch.shape}")
            return {}
        
        B, Tp1, N, obs_dim = obs_batch.shape
        T = Tp1 - 1  # Actual trajectory length (last step is for bootstrapping)
        
        window_size = self._policy.custom_config.get("discriminator_window_size", 32)
        
        # CRITICAL: Use a larger batch size for stable discriminator training!
        # With only B=5 rollout workers, we'd have very noisy gradients.
        # Sample multiple windows from the trajectory to get a healthy batch size.
        disc_batch_size = self._policy.custom_config.get("discriminator_batch_size", 64)
        
        if T < window_size:
            Logger.warning(f"[Discriminator] T={T} < window={window_size}, skipping.")
            return {}
        
        # Get strategy codes per rollout thread (same for all timesteps within a thread)
        # strategy_code_batch shape: [B, T, N] or [B, T, N, 1]
        if len(strategy_code_batch.shape) == 4:
            strategies_per_thread = strategy_code_batch[:, 0, 0, 0]  # [B]
        elif len(strategy_code_batch.shape) == 3:
            strategies_per_thread = strategy_code_batch[:, 0, 0]  # [B]
        else:
            strategies_per_thread = strategy_code_batch.flatten()[:B]
        
        # ====== VECTORIZED WINDOW EXTRACTION (5-10x faster than Python loop) ======
        # Use torch.unfold to extract ALL sliding windows in one operation (zero-copy!)
        
        # obs_batch: [B, T, N, obs_dim] -> we want windows of size W along dim 1
        # unfold creates: [B, num_windows, N, obs_dim, W] then we permute
        num_windows = T - window_size + 1
        
        if num_windows <= 0:
            Logger.warning(f"[Discriminator] Not enough timesteps for windows")
            return {}
        
        # Unfold along time dimension (dim=1)
        # Result: [B, num_windows, N, obs_dim, window_size]
        obs_unfolded = obs_batch[:, :T, :, :].unfold(dimension=1, size=window_size, step=1)
        # Permute to [B, num_windows, window_size, N, obs_dim]
        obs_unfolded = obs_unfolded.permute(0, 1, 4, 2, 3)
        
        # Same for actions: [B, T, N] -> [B, num_windows, window_size, N]
        act_unfolded = actions_batch[:, :T, :].unfold(dimension=1, size=window_size, step=1)
        act_unfolded = act_unfolded.permute(0, 1, 3, 2)
        
        # ====== DONE-MASKING: Create validity mask for windows ======
        # A window starting at t is valid if done[t:t+window_size] are all False
        done_batch = full_batch.get(EpisodeKey.DONE, None)
        
        # Default: all windows are valid
        # valid_mask: [B, num_windows] - True if window is valid
        valid_mask = torch.ones(B, num_windows, dtype=torch.bool, device=obs_batch.device)
        
        if done_batch is not None:
            if isinstance(done_batch, np.ndarray):
                done_batch = torch.from_numpy(done_batch).float().to(obs_batch.device)
            
            # Extract done flags: [B, T]
            if len(done_batch.shape) == 4:
                done_per_step = done_batch[:, :T, 0, 0]
            elif len(done_batch.shape) == 3:
                done_per_step = done_batch[:, :T, 0]
            else:
                done_per_step = done_batch[:, :T]
            
            # Unfold done flags: [B, num_windows, window_size]
            done_unfolded = done_per_step.unfold(dimension=1, size=window_size, step=1)
            
            # Window is valid if NO done=True within it
            # (checking if max < 0.5 means all are False/0)
            valid_mask = done_unfolded.max(dim=2).values < 0.5  # [B, num_windows]
        
        # ====== SAMPLE FROM VALID WINDOWS ======
        # Flatten to get all valid (batch, window_idx) pairs
        # valid_indices: list of (batch_idx, window_idx) tuples
        valid_indices = torch.nonzero(valid_mask)  # [num_valid, 2]
        
        if len(valid_indices) == 0:
            Logger.warning(f"[Discriminator] No valid windows found!")
            return {}
        
        num_valid = len(valid_indices)
        samples_collected = min(disc_batch_size, num_valid)
        
        # Randomly sample from valid indices
        if num_valid >= disc_batch_size:
            # Sample without replacement
            perm = torch.randperm(num_valid, device=obs_batch.device)[:disc_batch_size]
            sampled_indices = valid_indices[perm]
        else:
            # Not enough valid windows, use all of them
            sampled_indices = valid_indices
            Logger.warning(f"[Discriminator] Only {num_valid} valid windows, using all")
        
        # Extract sampled windows using advanced indexing
        batch_indices = sampled_indices[:, 0]  # [samples_collected]
        window_indices = sampled_indices[:, 1]  # [samples_collected]
        
        # Gather windows: [samples_collected, window_size, N, obs_dim]
        obs_windows = obs_unfolded[batch_indices, window_indices]
        action_windows = act_unfolded[batch_indices, window_indices]
        strategy_targets = strategies_per_thread[batch_indices]
        
        # Compute loss on the larger batch
        loss, accuracy = self._policy.discriminator.compute_loss(
            obs_windows, action_windows, strategy_targets
        )
        
        # Backward and update
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        
        if self._use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._policy.discriminator.parameters(), self.max_grad_norm
            )
        self.discriminator_optimizer.step()
        
        loss_val = float(loss.detach().cpu().numpy())
        acc_val = float(accuracy.detach().cpu().numpy())
        
        Logger.info(f"[Discriminator] UPDATED: batch={samples_collected}, loss={loss_val:.4f}, accuracy={acc_val:.4f}")
        
        return {
            "discriminator_loss": loss_val,
            "discriminator_accuracy": acc_val,
        }
    
    def loss_compute_supervisor(self, sample):
        """
        Phase 2: Train supervisor via PPO while players are frozen.
        
        The supervisor selects strategy codes c_t = F_φ(s^global_t).
        Players execute π(a|o, c_t) with frozen weights.
        Supervisor is trained on team reward.
        """
        if self._policy.supervisor is None:
            Logger.warning("Phase 2 training requested but supervisor is None")
            return {}
        
        # Ensure players are frozen
        if not self._policy._players_frozen:
            self._policy.freeze_players()
        
        # Extract supervisor-specific data from sample
        # We need: global observations, supervisor actions, returns, advantages
        obs_batch = sample[EpisodeKey.CUR_OBS]
        supervisor_action_batch = sample.get(EpisodeKey.SUPERVISOR_ACTION, sample.get(EpisodeKey.STRATEGY_CODE))
        old_log_prob_batch = sample.get(EpisodeKey.SUPERVISOR_LOG_PROB)
        value_preds_batch = sample.get(EpisodeKey.SUPERVISOR_VALUE, sample.get(EpisodeKey.STATE_VALUE))
        return_batch = sample[EpisodeKey.RETURN]
        adv_targ = sample[EpisodeKey.ADVANTAGE]
        active_masks_batch = sample.get(EpisodeKey.ACTIVE_MASK, None)
        
        if supervisor_action_batch is None or old_log_prob_batch is None:
            Logger.warning("Missing supervisor data in sample for Phase 2 training")
            return {}
        
        # Convert to tensors
        if isinstance(supervisor_action_batch, np.ndarray):
            supervisor_action_batch = torch.from_numpy(supervisor_action_batch).long().to(self._policy.device)
        if isinstance(old_log_prob_batch, np.ndarray):
            old_log_prob_batch = torch.from_numpy(old_log_prob_batch).float().to(self._policy.device)
        if isinstance(value_preds_batch, np.ndarray):
            value_preds_batch = torch.from_numpy(value_preds_batch).float().to(self._policy.device)
        if isinstance(return_batch, np.ndarray):
            return_batch = torch.from_numpy(return_batch).float().to(self._policy.device)
        if isinstance(adv_targ, np.ndarray):
            adv_targ = torch.from_numpy(adv_targ).float().to(self._policy.device)
        if active_masks_batch is not None and isinstance(active_masks_batch, np.ndarray):
            active_masks_batch = torch.from_numpy(active_masks_batch).float().to(self._policy.device)
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.from_numpy(obs_batch).float().to(self._policy.device)
        
        # Pool observations to get global state for supervisor
        # obs_batch shape: [T, batch*agents, obs_dim] or [batch*agents, obs_dim]
        num_agents = self._policy.custom_config.get("num_agents", 4)
        if obs_batch.dim() == 3:
            # [T, batch*agents, obs_dim] -> pool across agents
            T, batch_agents, obs_dim = obs_batch.shape
            batch_size = batch_agents // num_agents
            obs_pooled = obs_batch.view(T, batch_size, num_agents, obs_dim).mean(dim=2)  # [T, batch, obs_dim]
            obs_pooled = obs_pooled.view(-1, obs_dim)  # [T*batch, obs_dim]
        else:
            # [batch*agents, obs_dim]
            batch_agents, obs_dim = obs_batch.shape
            batch_size = batch_agents // num_agents
            obs_pooled = obs_batch.view(batch_size, num_agents, obs_dim).mean(dim=1)  # [batch, obs_dim]
        
        # Get supervisor action shape right (take first agent's strategy, same for all)
        # supervisor_action_batch may be 1D [batch*agents] or 2D [batch, agents]
        expected_batch = obs_pooled.shape[0]
        if supervisor_action_batch.dim() > 1:
            supervisor_action_batch = supervisor_action_batch.reshape(-1, num_agents)[:, 0]
        elif supervisor_action_batch.shape[0] != expected_batch:
            # 1D but per-agent: reshape to [batch, agents] and take first agent
            supervisor_action_batch = supervisor_action_batch.reshape(-1, num_agents)[:, 0]
        
        if old_log_prob_batch.dim() > 1:
            old_log_prob_batch = old_log_prob_batch.reshape(-1, num_agents)[:, 0]
        elif old_log_prob_batch.shape[0] != expected_batch:
            old_log_prob_batch = old_log_prob_batch.reshape(-1, num_agents)[:, 0]
        
        # Forward pass through supervisor
        _, new_log_prob, values, entropy = self._policy.supervisor(
            obs_pooled,
            explore=False,
            strategy=supervisor_action_batch,
        )
        
        # Compute PPO loss for supervisor
        log_prob = new_log_prob.view(-1, 1)
        old_log_prob = old_log_prob_batch.view(-1, 1)
        
        # Importance weights
        imp_weights = torch.exp(log_prob - old_log_prob)
        
        # Aggregate advantages across agents (same strategy for all)
        if adv_targ.dim() > 1 and adv_targ.shape[-1] > 1:
            adv_targ_sup = adv_targ.view(-1, num_agents, 1).mean(dim=1)  # Average across agents
        elif adv_targ.numel() != expected_batch:
            # 1D per-agent data: reshape and average
            adv_targ_sup = adv_targ.view(-1, num_agents).mean(dim=1, keepdim=True)
        else:
            adv_targ_sup = adv_targ.view(-1, 1)
        
        # Clipped surrogate objective
        surr1 = imp_weights * adv_targ_sup
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_sup
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        if entropy is not None:
            entropy_loss = -entropy.mean()
        else:
            entropy_loss = torch.tensor(0.0)
        
        # Value loss
        if return_batch.dim() > 1 and return_batch.shape[-1] > 1:
            return_sup = return_batch.view(-1, num_agents, 1).mean(dim=1)
        elif return_batch.numel() != expected_batch:
            return_sup = return_batch.view(-1, num_agents).mean(dim=1, keepdim=True)
        else:
            return_sup = return_batch.view(-1, 1)
        
        if value_preds_batch.dim() > 1 and value_preds_batch.shape[-1] > 1:
            value_preds_sup = value_preds_batch.view(-1, num_agents, 1).mean(dim=1)
        elif value_preds_batch.numel() != expected_batch:
            value_preds_sup = value_preds_batch.view(-1, num_agents).mean(dim=1, keepdim=True)
        else:
            value_preds_sup = value_preds_batch.view(-1, 1)
        
        value_loss = self._calc_value_loss(values, value_preds_sup, return_sup)
        
        # Total loss
        entropy_coef = self._policy.custom_config.get("entropy_coef", 0.01)
        value_loss_coef = self._policy.custom_config.get("value_loss_coef", 1.0)
        
        total_loss = policy_loss + entropy_loss * entropy_coef + value_loss * value_loss_coef
        total_loss = total_loss / self.grad_accum_step
        
        # Backward and update (supervisor only - critic provides baselines via inference)
        total_loss.backward()
        
        if self.n_opt_steps % self.grad_accum_step == 0:
            if self._use_max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self._policy.supervisor.parameters(), self.max_grad_norm
                )
            self.supervisor_optimizer.step()
            self.supervisor_optimizer.zero_grad()
        
        # Compute approx_kl for consistency with trainer expectations
        approx_kl = (old_log_prob - log_prob).mean().item()
        
        # Statistics
        stats = {
            "supervisor_policy_loss": float(policy_loss.detach().cpu().numpy()),
            "supervisor_value_loss": float(value_loss.detach().cpu().numpy()),
            "supervisor_entropy": float(entropy.mean().detach().cpu().numpy()) if entropy is not None else 0.0,
            "supervisor_ratio": float(imp_weights.mean().detach().cpu().numpy()),
            "approx_kl": approx_kl,  # Required by trainer for KL early stopping
        }
        
        return stats
    
    def loss_compute_sequential(self, sample):
        '''
        NOTE(jh): sharing policy is actually not suggested in sequentially-updating agorithm.
        the reason is the update of one agent will also affect others' policies that is not carefully analized.
        but as an approximation used in practice, it might be acceptable. so we don't restrict it.
        '''
        (
            value_preds_batch,
            adv_targ,
        ) = (
            sample[EpisodeKey.STATE_VALUE],
            sample[EpisodeKey.ADVANTAGE],
        )
        
        if not self._one_agent_per_update:
            self._agent_seq=self._get_agent_sequence(adv_targ, value_preds_batch)
        elif self._one_agent_per_update and len(self._agent_seq) == 0:
            self._agent_seq = self._get_agent_sequence(adv_targ, value_preds_batch)
            
        stats = {}
        for a_ids in self._agent_seq:
            if self._use_two_stage:
                self.loss_compute_simultaneous(sample, a_ids, update_actor=False)
            _stats = self.loss_compute_simultaneous(sample, a_ids)
            for k, v in _stats.items():
                if k in stats:
                    stats[k] += v
                else:
                    stats[k] = v
            if self._one_agent_per_update:
                self._agent_seq.pop(0)
                return stats
        for k, v in stats.items():
            stats[k] = v / len(self._agent_seq)
        return stats

    def _get_agent_sequence(self, adv_targ, value_preds_batch):
        # size (bsz, num_agents, ...)
        if self._seq_strategy == "random":
            seq = np.random.permutation(self._num_agents)
        elif self._seq_strategy in ["semi_greedy","greedy"]:
            adv_targ = adv_targ.reshape(-1, self._num_agents, *adv_targ.shape[1:])
            value_preds_batch = value_preds_batch.reshape(
                -1, self._num_agents, *value_preds_batch.shape[1:]
            )
            score = np.abs(
                adv_targ.cpu().numpy() / (value_preds_batch.cpu().numpy() + 1e-8)
            )
            score = np.mean(score, axis=0)
            score = np.sum(score, axis=score.shape[1:])
            id_scores = [(_i, _s) for (_i, _s) in zip(range(self._num_agents), score)]
            id_scores = sorted(id_scores, key=lambda x: x[1], reverse=True)
            if self._seq_strategy=="semi_greedy":
                # print("semi")
                seq = []
                a_i = 0
                while a_i < self._num_agents:
                    seq.append(id_scores[0][0])
                    id_scores.pop(0)
                    a_i += 1
                    if len(id_scores) > 0:
                        next_i = np.random.choice(len(id_scores))
                        seq.append(id_scores[next_i][0])
                        id_scores.pop(next_i)
                        a_i += 1
                seq = np.array(seq)
            else:
                seq = np.array([_i for (_i, _s) in id_scores])
        else:
            raise NotImplementedError("you can only select random, semi_greedy or greedy as your seq_strategy now.")
        if self._use_agent_block:
            _seq = np.array_split(seq, self._block_num)
        else:
            _seq = seq.reshape(-1, 1)

        if self._use_cum_sequence:
            seq = []
            for s_i in range(len(_seq)):
                seq.append(np.concatenate(_seq[: s_i + 1]))
        else:
            seq = _seq
        return seq

    def _calc_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch=None
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if active_masks_batch is not None:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / (active_masks_batch.sum()+1e-20)
        else:
            value_loss = value_loss.mean()

        return value_loss

    def zero_grad(self):
        pass

    def step(self):
        pass
