"""Actor-critic networks, observation normalization and portable policy bundles."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .config import OBSERVATION_SPEC


class RunningNorm(nn.Module):
    """Running mean/variance normalizer, updated with every collected observation."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]); n = x.shape[0]
        delta = x.mean(0) - self.mean; total = self.count + n
        self.var.copy_((self.var * self.count + x.var(0, unbiased=False) * n + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total); self.count.copy_(total)

    def forward(self, x):
        # Epsilon on the std (rsl_rl): near-constant features cannot explode when they change.
        return ((x - self.mean) / (torch.sqrt(self.var) + 1e-2)).clamp(-10, 10)


def mlp(size, hidden, out, last_gain):
    layers = []
    for width in hidden:
        layers += [nn.Linear(size, width), nn.ELU()]; size = width
    layers.append(nn.Linear(size, out))
    model = nn.Sequential(*layers)
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2 ** .5); nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model[-1].weight, gain=last_gain)
    return model


class ActorCritic(nn.Module):
    """Gaussian policy with a state-independent standard deviation, and a value function.

    The actor sees the (possibly noisy) actor observation; the critic sees clean
    observations plus privileged simulator values (asymmetric actor-critic).
    """

    def __init__(self, hidden=(512, 256, 128), init_std=1.):
        super().__init__()
        actor_dim, critic_dim = OBSERVATION_SPEC['actor_dim'], OBSERVATION_SPEC['critic_dim']
        self.actor = mlp(actor_dim, hidden, 12, .01)
        self.critic = mlp(critic_dim, hidden, 1, 1.)
        self.log_std = nn.Parameter(torch.full((12,), float(torch.log(torch.tensor(init_std)))))
        self.actor_norm, self.critic_norm = RunningNorm(actor_dim), RunningNorm(critic_dim)

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(self.actor_norm(obs)), self.log_std.clamp(-4, 1).exp())

    def value(self, critic_obs): return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

    def act(self, obs, deterministic=True):
        """Joint-target action for observation obs (the mean action when deterministic)."""
        return self.actor(self.actor_norm(obs)) if deterministic else self.distribution(obs).sample()


class InferenceActor(nn.Module):
    """Deterministic actor with its normalizer, for TorchScript export."""

    def __init__(self, model):
        super().__init__(); self.norm = model.actor_norm; self.actor = model.actor

    def forward(self, obs): return self.actor(self.norm(obs))


def load_controller(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('observation_spec') != OBSERVATION_SPEC:
        raise ValueError('Checkpoint observation contract does not match this code')
    model = ActorCritic(checkpoint['config']['hidden']).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    return model, checkpoint


def export_bundle(checkpoint_path, destination):
    """Copy a checkpoint, trace a TorchScript actor and write a manifest, then zip."""
    import shutil
    model, checkpoint = load_controller(checkpoint_path)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination / 'controller.pt')
    actor = torch.jit.trace(InferenceActor(model).eval(), torch.zeros(1, OBSERVATION_SPEC['actor_dim']))
    actor.save(str(destination / 'actor.ts'))
    manifest = dict(format='quadruped-controller/v3', observation_spec=OBSERVATION_SPEC, config=checkpoint['config'],
                    steps=checkpoint['steps'], update=checkpoint['update'], source_revision=checkpoint.get('source_revision'),
                    versions=checkpoint.get('versions'),
                    trained_performance='See evaluation.jsonl of the run; evaluate held-out scenes before use')
    manifest['sha256'] = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in ['controller.pt', 'actor.ts']}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return Path(shutil.make_archive(str(destination), 'zip', root_dir=destination))
