"""Actor-critic, fixed observation normalization, and portable policy bundles."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .config import OBSERVATION_SPEC


class RunningNorm(nn.Module):
    def __init__(self,size):
        super().__init__();self.register_buffer('mean',torch.zeros(size));self.register_buffer('var',torch.ones(size));self.register_buffer('count',torch.tensor(1e-4))
    @torch.no_grad()
    def update(self,x):
        x=x.reshape(-1,x.shape[-1]);n=x.shape[0];delta=x.mean(0)-self.mean;total=self.count+n
        self.var.copy_((self.var*self.count+x.var(0,unbiased=False)*n+delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*n/total);self.count.copy_(total)
    def forward(self,x):return ((x-self.mean)/torch.sqrt(self.var+1e-5)).clamp(-10,10)


def mlp(size,hidden,out):
    layers=[]
    for width in hidden:layers.extend([nn.Linear(size,width),nn.ELU()]);size=width
    layers.append(nn.Linear(size,out))
    model=nn.Sequential(*layers)
    for layer in model:
        if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,gain=1.414);nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model[-1].weight,gain=.01 if out==12 else 1.)
    return model


class ActorCritic(nn.Module):
    def __init__(self,hidden=(256,128,64)):
        super().__init__();self.actor=mlp(873,hidden,12);self.critic=mlp(876,hidden,1)
        self.log_std=nn.Parameter(torch.full((12,),-.5))
        self.actor_norm=RunningNorm(873);self.critic_norm=RunningNorm(876)
    def distribution(self,obs):return torch.distributions.Normal(self.actor(self.actor_norm(obs)),self.log_std.clamp(-5,1).exp())
    def value(self,obs):return self.critic(self.critic_norm(obs)).squeeze(-1)
    def act(self,obs,deterministic=True):
        dist=self.distribution(obs);return torch.tanh(dist.mean if deterministic else dist.sample())


def mirror_action(action):
    order=torch.tensor([3,4,5,0,1,2,9,10,11,6,7,8],device=action.device)
    signs=action.new_tensor([-1,1,1]*4)
    return action[...,order]*signs


def mirror_observation(obs):
    x=obs.clone()
    x[...,:3]=obs[...,:3]*obs.new_tensor([-1,1,-1])
    x[...,3:6]=obs[...,3:6]*obs.new_tensor([1,-1,1])
    for start in (6,18,30):x[...,start:start+12]=mirror_action(obs[...,start:start+12])
    x[...,43]=-obs[...,43];x[...,44]=-obs[...,44]
    x[...,48:873]=obs[...,48:873].reshape(*obs.shape[:-1],33,25).flip(-1).flatten(-2)
    return x


class InferenceActor(nn.Module):
    def __init__(self,model):super().__init__();self.norm=model.actor_norm;self.actor=model.actor
    def forward(self,obs):return torch.tanh(self.actor(self.norm(obs)))


def load_controller(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('observation_spec')!=OBSERVATION_SPEC:
        raise ValueError('Checkpoint observation/joint contract does not match this code')
    model=ActorCritic(checkpoint['config']['hidden']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint


def export_bundle(checkpoint_path,destination):
    import shutil
    model,checkpoint=load_controller(checkpoint_path)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(checkpoint_path,destination/'controller.pt')
    actor=torch.jit.trace(InferenceActor(model).eval(),torch.zeros(1,873))
    actor.save(str(destination/'actor.ts'))
    manifest=dict(format='quadruped-controller/v1',observation_spec=OBSERVATION_SPEC,
                  config=checkpoint['config'],steps=checkpoint['steps'],update=checkpoint['update'],
                  source_revision=checkpoint.get('source_revision'),versions=checkpoint.get('versions'),
                  trained_performance='Not certified by export; evaluate held-out episodes before use')
    manifest['sha256']={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ['controller.pt','actor.ts']}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=shutil.make_archive(str(destination),'zip',root_dir=destination)
    return Path(archive)
