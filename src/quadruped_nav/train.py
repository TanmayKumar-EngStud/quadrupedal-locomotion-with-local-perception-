"""PPO actor-critic training. Run a smoke test before allocating long GPU jobs."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import subprocess
import time
import mujoco
import torch
from .rl.config import TrainConfig, EnvConfig, OBSERVATION_SPEC
from .rl.env import GoalEnv
from .rl.policy import ActorCritic, mirror_action, mirror_observation, export_bundle, load_controller


def advantages(rewards,values,next_values,terminated,done,gamma,lam):
    result=torch.zeros_like(rewards);tail=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*(~terminated[t])*next_values[t]-values[t]
        tail=delta+gamma*lam*(~done[t])*tail;result[t]=tail
    return result,result+values


def train(cfg,resume=None,warm_start=None):
    cfg.env.validate();torch.manual_seed(cfg.env.seed)
    if cfg.env.device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    device=cfg.env.device
    model=ActorCritic(cfg.hidden).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,eps=1e-5)
    first,steps=0,0
    if resume or warm_start:
        loaded,checkpoint=load_controller(resume or warm_start,device)
        model.load_state_dict(loaded.state_dict())
        if resume:
            old=checkpoint['config']
            if old['hidden']!=cfg.hidden or any(old['env'][k]!=asdict(cfg.env)[k] for k in ['backend','num_envs','action_scale','kp','kd','sim_dt','control_dt']):
                raise ValueError('Resume must preserve task/model/control settings; use --warm-start for a new task')
            optimizer.load_state_dict(checkpoint['optimizer']);first=checkpoint['update'];steps=checkpoint['steps']
            cfg.env.row=old['env']['row'];cfg.env.family=old['env']['family'];cfg.env.seed=old['env']['seed']
            if old.get('terrain_families',[])!=cfg.terrain_families:raise ValueError('Resume must preserve terrain-family schedule')
            torch.set_rng_state(checkpoint['rng'].cpu())
    out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not resume:raise FileExistsError('Run directory already has a checkpoint; use --resume or a fresh output')
    (out/'config.json').write_text(json.dumps(cfg.dict(),indent=2)+'\n')
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:revision='unknown'
    if not resume and cfg.env.curriculum=='uniform':cfg.env.row=int(torch.randint(0,20,()).item())
    env=GoalEnv(cfg.env);obs,critic=env.observations()
    started=time.monotonic();window_arrivals=window_falls=window_timeouts=0
    for update in range(first,cfg.updates):
        env.update_index=update
        # Freeze moments for the whole rollout and all associated PPO epochs.
        actor_obs=[];critic_obs=[];raw_actions=[];logps=[];values=[];rewards=[];next_values=[];terms=[];dones=[]
        reward_totals={k:0. for k in cfg.env.rewards}
        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                dist=model.distribution(obs);raw=dist.sample();logp=dist.log_prob(raw).sum(-1);value=model.value(critic)
                nxt,nxtc,reward,done,info=env.step(torch.tanh(raw))
                bootstrap=model.value(nxtc)
                terminal=model.value(info['terminal_critic'])
                bootstrap=torch.where(info['truncated'],terminal,bootstrap)
            actor_obs.append(obs);critic_obs.append(critic);raw_actions.append(raw);logps.append(logp);values.append(value)
            rewards.append(reward);next_values.append(bootstrap);terms.append(info['terminated']);dones.append(done)
            window_arrivals+=int(info['arrived'].sum());window_falls+=int(info['fall'].sum());window_timeouts+=int(info['target_timeout'].sum())
            for k,v in info['reward_terms'].items():reward_totals[k]+=float(v.mean())
            obs,critic=nxt,nxtc
        arrays=[torch.stack(x) for x in [actor_obs,critic_obs,raw_actions,logps,values,rewards,next_values,terms,dones]]
        a,c,raw,oldlog,val,rew,nv,terminal,done=arrays
        adv,returns=advantages(rew,val,nv,terminal,done,cfg.gamma,cfg.gae_lambda)
        a=a.flatten(0,1);c=c.flatten(0,1);raw=raw.flatten(0,1);oldlog=oldlog.flatten();returns=returns.flatten()
        adv=adv.flatten();adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        losses=[];last_kl=0.
        for _ in range(cfg.epochs):
            for ids in torch.randperm(len(a),device=device).split(cfg.minibatch_size):
                dist=model.distribution(a[ids]);logp=dist.log_prob(raw[ids]).sum(-1)
                logratio=logp-oldlog[ids];ratio=logratio.exp()
                policy=-torch.minimum(ratio*adv[ids],ratio.clamp(1-cfg.clip,1+cfg.clip)*adv[ids]).mean()
                value_loss=(model.value(c[ids])-returns[ids]).square().mean()
                entropy=dist.entropy().sum(-1).mean()
                symmetry=(model.act(mirror_observation(a[ids]))-mirror_action(model.act(a[ids])).detach()).square().mean()
                loss=policy+cfg.value_coef*value_loss-cfg.entropy*entropy+cfg.symmetry_coef*symmetry
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss; stop and inspect rollout/rewards')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm);optimizer.step()
                last_kl=float(((ratio-1)-logratio).mean().detach());losses.append(float(loss.detach()))
            if last_kl>cfg.target_kl:break
        with torch.no_grad():model.actor_norm.update(a);model.critic_norm.update(c)
        steps+=cfg.rollout_steps*cfg.env.num_envs
        env.backend.check()
        record=dict(update=update+1,steps=steps,row=cfg.env.row,mean_reward=float(rew.mean()),loss=sum(losses)/len(losses),kl=last_kl,
                    samples_per_second=(steps-checkpoint['steps'] if resume else steps)/max(time.monotonic()-started,1e-6),
                    arrivals=window_arrivals,falls=window_falls,target_timeouts=window_timeouts,reward_terms=reward_totals)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps({k:record[k] for k in ['update','steps','row','mean_reward','samples_per_second','arrivals','falls']}),flush=True)
        if (update+1)%cfg.terrain_every==0 and update+1<cfg.updates:
            row=cfg.env.row
            if cfg.env.curriculum=='uniform':cfg.env.row=int(torch.randint(0,20,()).item())
            elif cfg.env.curriculum=='smooth':
                attempts=max(1,window_arrivals+window_falls+window_timeouts)
                rate=window_arrivals/attempts
                cfg.env.row=min(19,row+1) if rate>.7 else max(0,row-1) if rate<.2 else row
            # Refresh layout even if row does not change. This resets rollout state,
            # not across a GAE trajectory; each rollout has already been consumed.
            if cfg.terrain_families:
                cfg.env.family=cfg.terrain_families[((update+1)//cfg.terrain_every)%len(cfg.terrain_families)]
            cfg.env.seed+=1;env=GoalEnv(cfg.env);obs,critic=env.observations()
            window_arrivals=window_falls=window_timeouts=0
        if (update+1)%cfg.checkpoint_every==0 or update+1==cfg.updates:
            payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),config=cfg.dict(),
                         observation_spec=OBSERVATION_SPEC,update=update+1,steps=steps,rng=torch.get_rng_state(),source_revision=revision,
                         versions=dict(python=platform.python_version(),torch=str(torch.__version__),mujoco=mujoco.__version__))
            temporary=out/'last.tmp';torch.save(payload,temporary);temporary.replace(out/'last.pt')
            if update+1==cfg.updates or (update+1)%(cfg.checkpoint_every*10)==0:torch.save(payload,out/f'update_{update+1:06d}.pt')
    return out/'last.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/smoke.json');p.add_argument('--output');p.add_argument('--resume');p.add_argument('--warm-start')
    p.add_argument('--export',help='Export this checkpoint without training');p.add_argument('--bundle',default='artifacts/controller')
    a=p.parse_args()
    if a.export:print(export_bundle(a.export,a.bundle));return
    if a.resume and a.warm_start:p.error('Choose resume or warm-start')
    cfg=TrainConfig.load(a.config)
    if a.output:cfg.output=a.output
    checkpoint=train(cfg,a.resume,a.warm_start)
    print(f'Checkpoint: {checkpoint}\nRun quad-train --export {checkpoint} --bundle {a.bundle}')

if __name__=='__main__':main()
