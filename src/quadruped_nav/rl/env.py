"""Goal-reaching Go1 task shared by native MuJoCo and batched MuJoCo Warp."""
from __future__ import annotations
import math
from pathlib import Path
import tempfile
import numpy as np
import mujoco
import torch
import torch.nn.functional as F
from .config import EnvConfig, OBSERVATION_SPEC
from .backend import NativeBackend, WarpBackend
from ..mjcf import build_mjcf
from ..terrain import generate_terrain, surface_normal


def yaw_of(q):
    w,x,y,z=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def inverse_rotate(q,v):
    # MuJoCo quaternions use w,x,y,z.
    u=-q[:,1:]; t=2*torch.cross(u,v,dim=-1)
    return v+q[:,:1]*t+torch.cross(u,t,dim=-1)


class HeightMap:
    """Perfect precomputed elevation sensing (Lab 1 style), 2.5 cm raster."""
    def __init__(self,scene,device):
        self.resolution=.025
        records=scene['stones']+scene['platforms']
        self.xmin=min(p['x']-p['size'] for p in records)-1
        self.ymin=min(p['y']-p['size'] for p in records)-1
        xmax=max(p['x']+p['size'] for p in records)+1
        ymax=max(p['y']+p['size'] for p in records)+1
        nx,ny=math.ceil((xmax-self.xmin)/self.resolution)+1,math.ceil((ymax-self.ymin)/self.resolution)+1
        self.floor=-scene['parameters']['pit_depth']
        heights=np.full((ny,nx),self.floor,dtype=np.float32)
        for p in records+scene.get('obstacles',[]):
            sx,sy=p.get('size',p.get('size_x')),p.get('size',p.get('size_y'))
            r,ph=p.get('roll',0),p.get('pitch',0)
            # R=Ry(pitch)Rx(roll); no yaw in this terrain contract.
            R=np.array([[math.cos(ph),math.sin(ph)*math.sin(r),math.sin(ph)*math.cos(r)],
                        [0,math.cos(r),-math.sin(r)],[-math.sin(ph),math.cos(ph)*math.sin(r),math.cos(ph)*math.cos(r)]])
            radius=max(sx,sy)
            ix0=max(0,int((p['x']-radius-self.xmin)/self.resolution));ix1=min(nx,int((p['x']+radius-self.xmin)/self.resolution)+2)
            iy0=max(0,int((p['y']-radius-self.ymin)/self.resolution));iy1=min(ny,int((p['y']+radius-self.ymin)/self.resolution)+2)
            x,y=np.meshgrid(self.xmin+np.arange(ix0,ix1)*self.resolution,self.ymin+np.arange(iy0,iy1)*self.resolution)
            dx,dy=x-p['x'],y-p['y'];normal=R[:,2]
            dz=-(normal[0]*dx+normal[1]*dy)/normal[2]
            localx=R[0,0]*dx+R[1,0]*dy+R[2,0]*dz
            localy=R[0,1]*dx+R[1,1]*dy+R[2,1]*dz
            valid=(abs(localx)<=sx/2)&(abs(localy)<=sy/2)
            patch=heights[iy0:iy1,ix0:ix1]
            np.maximum(patch,np.where(valid,p['z_top']+dz,self.floor),out=patch)
        self.values=torch.tensor(heights,device=device)

    def sample(self,xy):
        ix=torch.round((xy[...,0]-self.xmin)/self.resolution).long()
        iy=torch.round((xy[...,1]-self.ymin)/self.resolution).long()
        valid=(ix>=0)&(ix<self.values.shape[1])&(iy>=0)&(iy<self.values.shape[0])
        result=self.values[iy.clamp(0,self.values.shape[0]-1),ix.clamp(0,self.values.shape[1]-1)]
        return torch.where(valid,result,torch.full_like(result,self.floor))


class GoalEnv:
    def __init__(self,cfg:EnvConfig,scene=None):
        cfg.validate();self.cfg=cfg;self.device=cfg.device;self.n=cfg.num_envs
        self.generator=torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.scene=scene if scene is not None else generate_terrain(cfg.family,cfg.row,cfg.seed)
        with tempfile.TemporaryDirectory() as temp:
            self.model=mujoco.MjModel.from_xml_path(str(build_mjcf(self.scene,Path(temp)/'scene.xml')))
        # Warp MULTICCD requires zero collision margins. Apply the same geometry
        # contract to native evaluation; keep multi-contact collision enabled.
        self.model.geom_margin[:]=0
        self.model.pair_margin[:]=0
        self.model.opt.timestep=cfg.sim_dt
        self.model.opt.iterations=100;self.model.opt.ls_iterations=50
        self.model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
        self.model.actuator_gainprm[:,0]=cfg.kp
        self.model.actuator_biasprm[:,1]=-cfg.kp;self.model.actuator_biasprm[:,2]=-cfg.kd
        self.model.dof_damping[6:]=.2
        self.backend=(NativeBackend if cfg.backend=='native' else WarpBackend)(self.model,self.n,self.device,nconmax=cfg.nconmax,njmax=cfg.njmax)
        self.home=torch.tensor(self.model.key_qpos[0],dtype=torch.float32,device=self.device)
        self.joint_low=torch.tensor(self.model.actuator_ctrlrange[:,0],dtype=torch.float32,device=self.device)
        self.joint_high=torch.tensor(self.model.actuator_ctrlrange[:,1],dtype=torch.float32,device=self.device)
        self.heightmap=HeightMap(self.scene,self.device)
        self.supports=self.scene['stones']+self.scene['platforms']
        self.piles=torch.tensor([[p['x'],p['y'],p['z_top']] for p in self.supports],device=self.device,dtype=torch.float32)
        self.foot_ids=[self.model.site(n).id for n in ['FR','FL','RR','RL']]
        self.trunk_id=self.model.body('trunk').id
        x,y=torch.meshgrid(torch.linspace(-.8,.8,33,device=self.device),torch.linspace(-.6,.6,25,device=self.device),indexing='ij')
        self.scan=torch.stack([x.flatten(),y.flatten()],-1)
        self.target=torch.zeros((self.n,3),device=self.device)
        self.target_heading=torch.zeros(self.n,device=self.device)
        self.final=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.arrival_awarded=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.previous_action=torch.zeros((self.n,12),device=self.device)
        self.previous_feet=torch.zeros((self.n,4,3),device=self.device)
        self.air_time=torch.zeros((self.n,4),device=self.device)
        self.elapsed=torch.zeros(self.n,device=self.device)
        self.target_elapsed=torch.zeros(self.n,device=self.device)
        self.update_index=0
        self.reset(torch.arange(self.n,device=self.device))

    def rand(self,shape): return torch.rand(shape,device=self.device,generator=self.generator)

    def reset(self,ids):
        if not len(ids): return
        q=self.home.repeat(len(ids),1)
        # Starting platform is large enough for the full robot, unlike small piles.
        q[:,:2]+=(self.rand((len(ids),2))-.5)*.10
        angle=(self.rand((len(ids),))-.5)*.8
        q[:,3]=torch.cos(angle/2);q[:,4:6]=0;q[:,6]=torch.sin(angle/2)
        self.backend.reset(ids,q)
        self.elapsed[ids]=0;self.target_elapsed[ids]=0;self.previous_action[ids]=0;self.air_time[ids]=0
        self.previous_feet[ids]=self.backend.tensor('site_xpos')[ids][:,self.foot_ids]
        self.sample_targets(ids)

    def sample_targets(self,ids):
        if not len(ids): return
        pos=self.backend.tensor('qpos')[ids,:3]
        delta=self.piles[None,:,:2]-pos[:,None,:2]
        dist=torch.linalg.vector_norm(delta,dim=-1)
        eligible=(dist>=1.)&(dist<=2.)&(abs(self.piles[None,:,2]-pos[:,None,2]+.3)<.15)
        # Fallback to nearer observed support where a nominal target is absent.
        fallback=(dist>.25)&(dist<=2.)
        eligible=torch.where(eligible.any(-1,keepdim=True),eligible,fallback)
        scores=self.rand(eligible.shape).masked_fill(~eligible,-1)
        selected=scores.argmax(-1)
        self.target[ids]=self.piles[selected]
        none=~eligible.any(-1)
        self.target[ids[none]]=pos[none]
        self.target_heading[ids]=torch.atan2(self.target[ids,1]-pos[:,1],self.target[ids,0]-pos[:,0])
        self.final[ids]=self.rand((len(ids),))<self.cfg.final_probability
        self.final[ids[none]]=True
        self.target_elapsed[ids]=0;self.arrival_awarded[ids]=False

    def set_target(self,xy,heading,final=False,ids=None):
        ids=torch.arange(self.n,device=self.device) if ids is None else ids
        self.target[ids,:2]=torch.as_tensor(xy,device=self.device)
        self.target[ids,2]=self.heightmap.sample(self.target[ids,:2])
        self.target_heading[ids]=heading;self.final[ids]=final
        self.target_elapsed[ids]=0;self.arrival_awarded[ids]=False

    def observations(self,noise=True):
        q=self.backend.tensor('qpos');v=self.backend.tensor('qvel');quat=q[:,3:7]
        yaw=yaw_of(quat);c,s=torch.cos(yaw),torch.sin(yaw)
        delta=self.target[:,:2]-q[:,:2]
        target_local=torch.stack([c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]],-1)
        error=self.target_heading-yaw
        points=torch.stack([c[:,None]*self.scan[None,:,0]-s[:,None]*self.scan[None,:,1],s[:,None]*self.scan[None,:,0]+c[:,None]*self.scan[None,:,1]],-1)+q[:,None,:2]
        heights=(q[:,2,None]-self.heightmap.sample(points)-.30).clamp(-1,1)
        gravity=inverse_rotate(quat,torch.tensor([0.,0.,-1.],device=self.device).expand(self.n,3))
        # MuJoCo free-joint angular qvel is body-local; linear qvel is world.
        actor=torch.cat([v[:,3:6],gravity,q[:,7:]-self.home[7:],v[:,6:]*.05,self.previous_action,
                         target_local,torch.stack([torch.sin(error),torch.cos(error)],-1),
                         (1-self.target_elapsed/self.cfg.subgoal_seconds).clamp(0,1)[:,None],self.final.float()[:,None],heights],-1)
        if noise and self.cfg.noise:
            actor=actor.clone();actor[:,:30]+=(self.rand((self.n,30))-.5)*2*self.cfg.noise
        actor=actor.clamp(-10,10)
        critic=torch.cat([actor,inverse_rotate(quat,v[:,:3])],-1)
        assert actor.shape[-1]==OBSERVATION_SPEC['actor_dim']
        return actor,critic

    def step(self,action,auto_reset=True,chain=True):
        cfg=self.cfg;dt=cfg.control_dt
        action=action.clamp(-1,1)
        before=self.backend.tensor('qpos').clone();old_v=self.backend.tensor('qvel').clone()
        old_distance=torch.linalg.vector_norm(self.target[:,:2]-before[:,:2],dim=-1)
        targets=(self.home[7:]+cfg.action_scale*action).clamp(self.joint_low,self.joint_high)
        self.backend.step(targets,round(dt/cfg.sim_dt))
        q=self.backend.tensor('qpos').clone();v=self.backend.tensor('qvel').clone()
        self.elapsed+=dt;self.target_elapsed+=dt
        finite=torch.isfinite(q).all(-1)&torch.isfinite(v).all(-1)
        if not bool(finite.all()):raise FloatingPointError('Nonfinite physics state; stop rather than train on corrupted rollouts')
        q=torch.nan_to_num(q);v=torch.nan_to_num(v)
        distance=torch.linalg.vector_norm(self.target[:,:2]-q[:,:2],dim=-1)
        yaw=yaw_of(q[:,3:7]);error=torch.atan2(torch.sin(self.target_heading-yaw),torch.cos(self.target_heading-yaw))
        gravity=inverse_rotate(q[:,3:7],torch.tensor([0.,0.,-1.],device=self.device).expand(self.n,3))
        feet=self.backend.tensor('site_xpos')[:,self.foot_ids].clone()
        foot_h=feet[:,:,2]-self.heightmap.sample(feet[:,:,:2])
        contact=foot_h<.045
        foot_vel=(feet-self.previous_feet)/dt
        self.air_time+=dt
        # Relative to actual local terrain; over a hole the base may be above
        # pit height, so also terminate below the support scene's nominal floor.
        base_h=q[:,2]-self.heightmap.sample(q[:,:2])
        fall=(gravity[:,2]>-.35)|(base_h<.14)|(q[:,2]<-.3)|(~finite)
        arrived=(distance<cfg.arrival_radius)&(~fall)&(base_h>.18)&(base_h<.5)
        newly_arrived=arrived&(~self.arrival_awarded)
        self.arrival_awarded|=arrived
        target_timeout=self.target_elapsed>=cfg.subgoal_seconds
        timeout=self.elapsed>=cfg.episode_seconds
        final_complete=self.final&self.arrival_awarded&target_timeout
        terminated=fall|final_complete
        truncated=timeout&(~terminated)
        direction=(self.target[:,:2]-q[:,:2])/distance.clamp_min(.01)[:,None]
        desired=direction*torch.minimum(distance,torch.full_like(distance,.6))[:,None]
        final_hold=self.final&arrived
        desired[final_hold]=0
        torques=self.backend.tensor('qfrc_actuator')[:,6:]
        softlo=self.home[7:]+.95*(self.joint_low-self.home[7:]);softhi=self.home[7:]+.95*(self.joint_high-self.home[7:])
        values=dict(
            position=1/(1+distance.square())*(self.target_elapsed>cfg.subgoal_seconds-1),
            heading=1/(1+error.square())*(distance<.5),
            progress=(old_distance-distance)/dt*(self.update_index<150),
            tracking_lin_vel=torch.exp(-((v[:,:2]-desired).square().sum(-1))/.25),
            tracking_ang_vel=torch.exp(-(v[:,5]-error.clamp(-1,1)).square()/.25),
            orientation=gravity[:,:2].square().sum(-1),lin_vel_z=v[:,2].square(),ang_vel_xy=v[:,3:5].square().sum(-1),
            pose=torch.exp(-(q[:,7:]-self.home[7:]).square().mean(-1)),
            feet_clearance=(foot_h.clamp(0,.1)*(~contact)).mean(-1)*(distance>.25),
            feet_slip=(foot_vel[:,:,:2].square().sum(-1)*contact).mean(-1),
            feet_air_time=((self.air_time-.08).clamp(0,.4)*contact).mean(-1)*(distance>.25),
            action_rate=(action-self.previous_action).square().sum(-1),torques=torques.square().sum(-1),
            energy=abs(torques*v[:,6:]).sum(-1),dof_acc=((v[:,6:]-old_v[:,6:])/dt).square().sum(-1),
            dof_pos_limits=((softlo-q[:,7:]).clamp_min(0)+(q[:,7:]-softhi).clamp_min(0)).sum(-1),
            stall=((distance>.5)&(torch.linalg.vector_norm(v[:,:2],dim=-1)<.05)).float(),
            stand_still=(v[:,:3].square().sum(-1)+.05*v[:,6:].square().sum(-1))*final_hold,
            arrival=newly_arrived.float()/dt,fall=fall.float()/dt)
        weighted={k:torch.nan_to_num(values[k])*weight*dt for k,weight in cfg.rewards.items()}
        reward=sum(weighted.values())
        self.previous_action=action.clone();self.previous_feet=feet;self.air_time[contact]=0
        # Value bootstrap for time limits MUST see the terminal state, before reset.
        terminal_actor,terminal_critic=self.observations(noise=False)
        info=dict(terminated=terminated.clone(),truncated=truncated.clone(),terminal_critic=terminal_critic,
                  arrived=newly_arrived.clone(),fall=fall.clone(),final_complete=final_complete.clone(),target_timeout=(target_timeout&~arrived).clone(),
                  reward_terms=weighted,distance=distance.clone())
        if chain:
            switch=(arrived&~self.final)|target_timeout
            self.sample_targets(torch.where(switch&~terminated&~truncated)[0])
        if auto_reset:self.reset(torch.where(terminated|truncated)[0])
        return *self.observations(),reward,terminated|truncated,info
