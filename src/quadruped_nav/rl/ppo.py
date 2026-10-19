"""Proximal Policy Optimization (clipped objective, GAE, adaptive learning rate).

The structure follows rsl_rl / legged_gym, which is the standard recipe for
learning legged locomotion with thousands of parallel simulated robots.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class Rollout:
    """Tensors of shape [steps, robots, ...] collected with the current policy."""
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_values: torch.Tensor     # value of the observation after the last step


def gae(rollout, gamma, lam):
    """Generalized advantage estimation; returns (advantages, value targets)."""
    advantages = torch.zeros_like(rollout.rewards)
    running = torch.zeros_like(rollout.last_values)
    next_values = rollout.last_values
    for t in reversed(range(len(rollout.rewards))):
        not_done = (~rollout.dones[t]).float()
        delta = rollout.rewards[t] + gamma * next_values * not_done - rollout.values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
        next_values = rollout.values[t]
    return advantages, advantages + rollout.values


class PPO:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.learning_rate = cfg.learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def update(self, rollout):
        cfg, model = self.cfg, self.model
        advantages, returns = gae(rollout, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: x.flatten(0, 1)
        actor_obs, critic_obs, actions = flat(rollout.actor_obs), flat(rollout.critic_obs), flat(rollout.actions)
        old_log_probs, old_values, returns, advantages = flat(rollout.log_probs), flat(rollout.values), flat(returns), flat(advantages)
        mean_advantage = float(advantages.abs().mean())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch = len(actions); size = batch // cfg.minibatches
        stats = dict(policy_loss=0., value_loss=0., entropy=0., kl=0., clip_fraction=0.)
        count = 0
        for _ in range(cfg.epochs):
            for ids in torch.randperm(batch, device=actions.device).split(size):
                dist = model.distribution(actor_obs[ids])
                log_probs = dist.log_prob(actions[ids]).sum(-1)
                ratio = torch.exp(log_probs - old_log_probs[ids])
                surrogate = torch.minimum(ratio * advantages[ids], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[ids])
                values = model.value(critic_obs[ids])
                clipped_values = old_values[ids] + (values - old_values[ids]).clamp(-cfg.clip, cfg.clip)
                value_loss = torch.maximum((values - returns[ids]).square(), (clipped_values - returns[ids]).square()).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = -surrogate.mean() + cfg.value_coef * value_loss - cfg.entropy * entropy
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm); self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean()
                    # Adaptive learning rate: keep the policy change per minibatch near desired_kl.
                    if kl > 2 * cfg.desired_kl: self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl < cfg.desired_kl / 2: self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups: group['lr'] = self.learning_rate
                    stats['policy_loss'] += float(-surrogate.mean()); stats['value_loss'] += float(value_loss)
                    stats['entropy'] += float(entropy); stats['kl'] += float(kl)
                    stats['clip_fraction'] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                count += 1
        stats = {k: v / count for k, v in stats.items()}
        explained = 1 - (returns - old_values).var() / returns.var().clamp_min(1e-8)
        stats.update(learning_rate=self.learning_rate, explained_variance=float(explained),
                     action_std=float(model.log_std.detach().clamp(-4, 1).exp().mean()), mean_advantage_abs=mean_advantage)
        return stats
