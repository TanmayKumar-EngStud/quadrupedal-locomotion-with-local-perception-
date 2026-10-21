"""Execute a downloaded controller in local MuJoCo with a high-level planner."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .abstract import Terrain,Limits,Geometry,observe_pose
from .planners import GreedyPlanner,OraclePlanner,OptimisticPlanner
from .rl.config import EnvConfig
from .rl.env import GoalEnv,yaw_of
from .rl.policy import load_controller
from .terrain import TERRAINS,generate_terrain
from .scene import load_scene
from .markers import describe_scan,draw_scan


def make_planner(name,terrain,limits,checkpoint=None):
    if name=='oracle':return OraclePlanner(Geometry(terrain,limits))
    if name=='greedy':return GreedyPlanner()
    if name=='optimistic':return OptimisticPlanner(max_gap=limits.max_gap_m)
    if name=='learned':
        from .learned import load_planner
        if not checkpoint:raise ValueError('--planner-checkpoint is required')
        return load_planner(checkpoint)
    raise ValueError('Unknown planner')


def draw_subgoals(scene,current,visited,radius):
    """Viewer markers: the current subgoal (red sphere inside its arrival circle) and earlier subgoals (small blue spheres)."""
    import mujoco
    scene.ngeom=0
    def add(kind,size,pos,rgba):
        if scene.ngeom>=scene.maxgeom:return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],kind,np.array(size,dtype=np.float64),np.array(pos,dtype=np.float64),
                            np.eye(3).ravel(),np.array(rgba,dtype=np.float32));scene.ngeom+=1
    for x,y,z in visited[-40:]:add(mujoco.mjtGeom.mjGEOM_SPHERE,[.035,0,0],[x,y,z+.035],[.15,.45,.95,.9])
    if current is not None:
        x,y,z=current
        add(mujoco.mjtGeom.mjGEOM_CYLINDER,[radius,.002,0],[x,y,z+.004],[.95,.3,.2,.25])
        add(mujoco.mjtGeom.mjGEOM_SPHERE,[.06,0,0],[x,y,z+.06],[.95,.2,.15,1])


def evaluate(checkpoint,scene,planner_name='optimistic',seconds=60,seed=0,planner_checkpoint=None,viewer=False,ground='boxes',on_step=None,show_scan=True):
    """One planner episode in native MuJoCo. on_step(env), if given, is called after every control step (e.g. to render).

    With the viewer, show_scan draws the terrain samples the controller senses (see markers.py)."""
    torch.set_num_threads(2);torch.manual_seed(seed)
    policy,saved=load_controller(checkpoint)
    cfg=replace(EnvConfig(**saved['config']['env']),num_envs=1,backend='native',device='cpu',seed=seed,noise=0.,episode_seconds=seconds)
    env=GoalEnv(cfg,scene,ground=ground);terrain=Terrain.from_scene(scene);limits=Limits()
    planner=make_planner(planner_name,terrain,limits,planner_checkpoint)
    goal=env.piles[[p['id'] for p in env.supports].index('P2')]
    current=None;decisions=[];trajectory=[];status='timeout';distance=0.;hold=0.;visited=[]
    handle=None
    if viewer:
        import mujoco.viewer
        handle=mujoco.viewer.launch_passive(env.model,env.backend.data[0])
        handle.cam.lookat[:]=[3,0,0];handle.cam.distance=12;handle.cam.azimuth=130;handle.cam.elevation=-50
        print(describe_scan(env),flush=True)
    try:
        for step in range(math.ceil(seconds/cfg.control_dt)):
            started=time.monotonic();q=env.backend.tensor('qpos')[0].clone();yaw=float(yaw_of(q[None,3:7])[0])
            if trajectory:distance+=float(torch.linalg.vector_norm(q[:2]-torch.tensor(trajectory[-1]['position'][:2])))
            trajectory.append(dict(time_s=step*cfg.control_dt,position=q[:3].tolist(),yaw=yaw))
            goal_distance=float(torch.linalg.vector_norm(q[:2]-goal[:2]))
            if goal_distance<cfg.arrival_radius and abs(float(q[2]-goal[2])-.3)<.15:
                hold+=cfg.control_dt
                if hold>=.5:status='success';break
            else:hold=0.
            reached=current is not None and float(torch.linalg.vector_norm(q[:2]-env.target[0,:2]))<cfg.arrival_radius
            timed_out=float(env.target_elapsed[0])>=float(env.target_budget[0])
            if current is None or (reached and not bool(env.final[0])) or timed_out:
                obs=observe_pose(terrain,limits,q[:3],yaw,len(decisions),step*cfg.control_dt)
                target=planner.select_subgoal(obs)
                if goal_distance<cfg.arrival_radius:target=terrain.goal_id
                if target is None:status='no_candidate';break
                valid={p.id for p,ok in zip(obs.visible_piles,obs.action_mask) if ok}
                if target not in valid and goal_distance>=cfg.arrival_radius:raise RuntimeError('Planner returned a masked target')
                pile=next(p for p in terrain.piles if p.id==target)
                heading=math.atan2(pile.y-float(q[1]),pile.x-float(q[0])) if goal_distance>=cfg.arrival_radius else yaw
                if current is not None:visited.append(previous_xyz)
                env.set_target([[pile.x,pile.y]],heading,target==terrain.goal_id);current=target;previous_xyz=(pile.x,pile.y,pile.z)
                decisions.append(dict(time_s=step*cfg.control_dt,target=target,x=pile.x,y=pile.y,reason='timeout' if timed_out else 'arrival' if reached else 'initial'))
                if handle:
                    hop=math.hypot(pile.x-float(q[0]),pile.y-float(q[1]))
                    print(f"{step*cfg.control_dt:6.1f} s  subgoal {len(decisions):2d}: pile at ({pile.x:5.2f}, {pile.y:5.2f}), {hop:.2f} m away"
                          f"{' (goal)' if target==terrain.goal_id else ''}  [{decisions[-1]['reason']}]",flush=True)
            with torch.no_grad():action=policy.act(env.observations(noise=False)[0])
            _,_,_,done,info=env.step(action,auto_reset=False,chain=False)
            if on_step:on_step(env)
            if handle:
                if not handle.is_running():status='viewer_closed';break
                with handle.lock():
                    draw_subgoals(handle.user_scn,previous_xyz,visited,cfg.arrival_radius)
                    if show_scan:draw_scan(handle.user_scn,env)
                handle.sync();time.sleep(max(0,cfg.control_dt-(time.monotonic()-started)))
            if bool(info['fall'][0]):status='fall';break
            if bool(done[0]):break
        env.backend.check()
    finally:
        if handle:handle.close();print(f'Episode ended: {status} after {float(env.elapsed[0]):.1f} s, {len(decisions)} subgoals',flush=True)
    return dict(planner=planner_name,status=status,success=status=='success',seed=seed,scene=scene,
                time_s=float(env.elapsed[0]),distance_m=distance,decisions=decisions,trajectory=trajectory,
                checkpoint=str(Path(checkpoint).resolve()),physics='native-mujoco',ground=ground,model_config=vars(cfg))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--planner',choices=['greedy','oracle','optimistic','learned'],default='optimistic')
    p.add_argument('--planner-checkpoint');p.add_argument('--family',choices=TERRAINS,default='open');p.add_argument('--row',type=int,default=0)
    p.add_argument('--seed',type=int,default=10000);p.add_argument('--scene');p.add_argument('--seconds',type=float,default=60);p.add_argument('--viewer',action='store_true')
    p.add_argument('--ground',choices=['boxes','heightfield'],default='boxes',help='scene supports as boxes, or as the heightfield the controller senses')
    p.add_argument('--no-scan',action='store_true',help='viewer: hide the terrain samples the controller senses')
    p.add_argument('--output',default='outputs/evaluation/episode.json');a=p.parse_args()
    if a.seconds<=0:p.error('Duration must be positive')
    scene=load_scene(a.scene) if a.scene else generate_terrain(a.family,a.row,a.seed)
    result=evaluate(a.checkpoint,scene,a.planner,a.seconds,a.seed,a.planner_checkpoint,a.viewer,a.ground,show_scan=not a.no_scan)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['planner','status','time_s','distance_m']}));print(path)

if __name__=='__main__':main()
