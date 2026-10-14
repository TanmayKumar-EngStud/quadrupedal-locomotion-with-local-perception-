"""Measure short-goal success and save empirical transition probabilities."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from .rl.config import EnvConfig
from .rl.env import GoalEnv
from .rl.policy import load_controller
from .terrain import generate_terrain


def calibrate(checkpoint,output,trials=20,rows=(0,5,10,15,19)):
    model,saved=load_controller(checkpoint);torch.set_num_threads(2);records=[]
    for family in ['open','rough','slanted']:
        for row in rows:
            for trial in range(trials):
                cfg=replace(EnvConfig(**saved['config']['env']),num_envs=1,backend='native',device='cpu',seed=300000+trial,
                            noise=0.,episode_seconds=4.,final_probability=1.)
                scene=generate_terrain(family,row,cfg.seed);env=GoalEnv(cfg,scene)
                q=env.backend.tensor('qpos')[0].clone()
                lo,hi=[(.25,1.),(1.,1.5),(1.5,2.01)][trial%3]
                distances=torch.linalg.vector_norm(env.piles[:,:2]-q[:2],dim=-1)
                candidates=torch.where((distances>=lo)&(distances<hi))[0]
                if not len(candidates):continue
                target=env.piles[int(candidates[trial%len(candidates)])].clone()
                heading=float(torch.atan2(target[1]-q[1],target[0]-q[0]))
                env.set_target(target[None,:2],heading,True)
                length=float(torch.linalg.vector_norm(target[:2]-q[:2]));success=False
                for _ in range(200):
                    with torch.no_grad():action=model.act(env.observations(False)[0])
                    _,_,_,done,info=env.step(action,auto_reset=False,chain=False)
                    if bool(info['arrived'][0]):success=True;break
                    if bool(done[0]):break
                records.append(dict(family=family,row=row,trial=trial,length_m=length,gap_m=scene['parameters']['gap'],
                                    height_delta_m=float(target[2]-q[2]+.3),success=success))
    bins=[]
    for family in ['open','rough','slanted']:
        for row in rows:
            for lo,hi in [(0.,1.),(1.,1.5),(1.5,2.1)]:
                selected=[r for r in records if r['family']==family and r['row']==row and lo<=r['length_m']<hi]
                if selected:bins.append(dict(family=family,row=row,length_min=lo,length_max=hi,n=len(selected),successes=sum(r['success'] for r in selected),probability=sum(r['success'] for r in selected)/len(selected)))
    result=dict(format='hop-calibration/v1',checkpoint=str(Path(checkpoint).resolve()),records=records,bins=bins,
                note='Empirical family/row/length bins; no extrapolation to unseen configurations. Not a stability-at-goal certificate.')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');return path


class CalibratedEnv:
    """Wrapper replacing deterministic abstract arrivals with measured odds."""
    def __init__(self,env,table,family,row,seed=0):
        import random
        self.env=env;self.table=json.loads(Path(table).read_text());self.family=family;self.row=row;self.rng=random.Random(seed)
    def __getattr__(self,name):return getattr(self.env,name)
    def reset(self):return self.env.reset()
    def step(self,target):
        if target is None:return self.env.step(target)
        import math
        start=self.env.geometry.by_id[self.env.current_id];end=self.env.geometry.by_id.get(target)
        if end is None:return self.env.step(target)
        length=math.hypot(end.x-start.x,end.y-start.y)
        choices=[b for b in self.table['bins'] if b['family']==self.family and b['row']==self.row and b['length_min']<=length<b['length_max']]
        if not choices:raise ValueError('No measured transition bin; collect calibration rather than extrapolating')
        before=self.env.state();obs,done=self.env.step(target)
        if self.env.status not in ('timeout','no_candidate','planner_no_route') and self.rng.random()>choices[0]['probability']:
            self.env.current_id=before['pile_id'];self.env.distance_m=before['distance_m'];self.env.status='fall'
            self.env.states[-1]=self.env.state();self.env.events[-1]['outcome']='fall';return self.env.observe(),True
        return obs,done


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--trials',type=int,default=20);p.add_argument('--rows',nargs='+',type=int,default=[0,5,10,15,19]);p.add_argument('--output',default='outputs/calibration/hops.json');a=p.parse_args()
    if a.trials<1:p.error('At least one trial required')
    print(calibrate(a.checkpoint,a.output,a.trials,a.rows))
if __name__=='__main__':main()
