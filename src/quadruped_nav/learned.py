"""DAgger pile scoring with L0/L1/L2/L3 memory ablations and optional REINFORCE."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .abstract import AbstractEnv,Terrain,Limits
from .planners import OraclePlanner
from .terrain import generate_terrain


class PileNetwork(nn.Module):
    def __init__(self,variant='L0'):
        super().__init__()
        if variant not in ('L0','L1','L2','L3'):raise ValueError('Unknown memory variant')
        self.variant=variant;self.recurrent=variant in ('L1','L3')
        self.encoder=nn.Sequential(nn.Linear(9 if variant in ('L2','L3') else 8,64),nn.ELU(),nn.Linear(64,64),nn.ELU())
        if self.recurrent:self.gru=nn.GRUCell(130,64)
        self.scorer=nn.Sequential(nn.Linear(194+(64 if self.recurrent else 0),64),nn.ELU(),nn.Linear(64,1))

    def forward(self,features,goal,mask,hidden=None):
        embedding=self.encoder(features)
        context=torch.cat([embedding.mean(0),embedding.max(0).values,goal])
        parts=[embedding,context.expand(len(embedding),-1)]
        if self.recurrent:
            if hidden is None:hidden=features.new_zeros(1,64)
            hidden=self.gru(context[None],hidden);parts.append(hidden.expand(len(embedding),-1))
        logits=self.scorer(torch.cat(parts,-1)).squeeze(-1).masked_fill(~mask,-1e9)
        return logits,hidden


class LearnedPlanner:
    def __init__(self,model,device='cpu'):
        self.model=model.to(device);self.device=device;self.reset()
    def reset(self):self.hidden=None;self.visits=Counter()
    def inputs(self,obs):
        self.visits[obs.current_pile_id]+=1
        features=[]
        for p in obs.visible_piles:
            row=[*(v/2 for v in p.relative_top_center_m),*p.top_size_xy_m,*p.top_normal_local]
            if self.model.variant in ('L2','L3'):row.append(float(np.log1p(self.visits[p.id])))
            features.append(row)
        return (torch.tensor(features,dtype=torch.float32,device=self.device),
                torch.tensor(obs.goal_relative_xy_m,dtype=torch.float32,device=self.device)/10,
                torch.tensor(obs.action_mask,dtype=torch.bool,device=self.device))
    def select_subgoal(self,obs):
        inputs=self.inputs(obs)
        if not any(obs.action_mask):return None
        with torch.no_grad():logits,self.hidden=self.model(*inputs,self.hidden)
        return obs.visible_piles[int(logits.argmax())].id


def load_planner(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!='pile-planner/v1':raise ValueError('Unknown planner checkpoint format')
    model=PileNetwork(checkpoint['variant']);model.load_state_dict(checkpoint['model']);model.eval()
    return LearnedPlanner(model)


def train_planner(variant='L0',rounds=20,episodes=8,seed=0,output='runs/planner',rl_rounds=0):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=random.Random(seed)
    model=PileNetwork(variant);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    dataset=[];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'planner.pt').exists():raise FileExistsError('Choose a fresh planner output directory')
    for iteration in range(rounds):
        for episode in range(episodes):
            family=rng.choice(['open','holes','moat','dead_end'])
            scene=generate_terrain(family,row=rng.randrange(4),seed=rng.randrange(100000))
            env=AbstractEnv(Terrain.from_scene(scene));oracle=OraclePlanner(env.geometry);learner=LearnedPlanner(model)
            sequence=[];obs=env.observe();beta=max(.05,.7*(.85**iteration))
            while env.status=='running':
                target=oracle.select_subgoal(obs)
                if target is None:break
                ids=[p.id for p in obs.visible_piles];label=ids.index(target)
                inputs=learner.inputs(obs)
                with torch.no_grad():logits,learner.hidden=model(*inputs,learner.hidden)
                choice=target if rng.random()<beta else ids[int(logits.argmax())]
                sequence.append((*[x.cpu() for x in inputs],label))
                obs,_=env.step(choice)
            if sequence:dataset.append(sequence)
        # Aggregate past student-state data, bounded by whole episodes; no split
        # across train/test layouts and no hidden terrain in network features.
        dataset=dataset[-256:];losses=[]
        if not dataset:raise RuntimeError('No oracle-labelled training episodes; inspect terrain/connectivity')
        for sequence in rng.sample(dataset,min(len(dataset),32)):
            hidden=None;loss=0.;count=0
            for i,(features,goal,mask,label) in enumerate(sequence):
                logits,hidden=model(features,goal,mask,hidden)
                loss=loss+nn.functional.cross_entropy(logits[None],torch.tensor([label]));count+=1
                if count==16 or i==len(sequence)-1:
                    optimizer.zero_grad();(loss/count).backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    losses.append(float(loss.detach())/count);loss=0.;count=0
                    if hidden is not None:hidden=hidden.detach()
        log=dict(round=iteration+1,method='dagger',episodes=len(dataset),loss=sum(losses)/max(1,len(losses)))
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        print(json.dumps(log),flush=True)
    # Optional on-policy fine-tuning. Privileged oracle DISTANCE shapes training
    # only; it is never added to the learned observation. No revisit penalty.
    for iteration in range(rl_rounds):
        env=AbstractEnv(Terrain.from_scene(generate_terrain(rng.choice(['open','moat','dead_end']),row=0,seed=rng.randrange(100000))))
        oracle=OraclePlanner(env.geometry);learner=LearnedPlanner(model);obs=env.observe();logps=[];rewards=[]
        def remaining(current):
            route=oracle.route(current)
            if not route:return None
            nodes=env.geometry.by_id
            return sum(np.hypot(nodes[b].x-nodes[a].x,nodes[b].y-nodes[a].y) for a,b in zip(route,route[1:]))
        before=remaining(env.current_id)
        if before is None:continue
        while env.status=='running':
            inputs=learner.inputs(obs)
            if not any(obs.action_mask):break
            logits,learner.hidden=model(*inputs,learner.hidden)
            distribution=torch.distributions.Categorical(logits=logits);index=distribution.sample();logps.append(distribution.log_prob(index))
            obs,_=env.step(obs.visible_piles[int(index)].id);after=remaining(env.current_id)
            rewards.append((before-after if after is not None else -1)-.02+(5 if env.status=='success' else 0))
            before=after if after is not None else before
        if logps:
            returns=[];value=0.
            for reward in rewards[::-1]:value=reward+.99*value;returns.append(value)
            ret=torch.tensor(returns[::-1],dtype=torch.float32);loss=-(torch.stack(logps)*ret).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(round=iteration+1,method='reinforce',loss=float(loss.detach()),status=env.status))+'\n')
    path=out/'planner.pt'
    torch.save(dict(format='pile-planner/v1',variant=variant,model=model.state_dict(),seed=seed,dagger_rounds=rounds,rl_rounds=rl_rounds),path)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=['L0','L1','L2','L3'],default='L0')
    p.add_argument('--rounds',type=int,default=20);p.add_argument('--episodes',type=int,default=8);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--rl-rounds',type=int,default=0);p.add_argument('--output',default='runs/planner')
    a=p.parse_args()
    if min(a.rounds,a.episodes)<1 or a.rl_rounds<0:p.error('Positive round/episode counts required')
    print(train_planner(a.variant,a.rounds,a.episodes,a.seed,a.output,a.rl_rounds))

if __name__=='__main__':main()
