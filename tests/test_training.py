import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import mujoco
torch=pytest.importorskip('torch')
from quadruped_nav.terrain import generate_terrain,TERRAINS
from quadruped_nav.mjcf import build_mjcf
from quadruped_nav.rl.config import EnvConfig,TrainConfig,OBSERVATION_SPEC
from quadruped_nav.rl.env import GoalEnv,HeightMap
from quadruped_nav.rl.policy import ActorCritic,mirror_action,mirror_observation,load_controller,export_bundle
from quadruped_nav.train import train,advantages
from quadruped_nav.abstract import AbstractEnv,Terrain,Limits
from quadruped_nav.planners import OptimisticPlanner
from quadruped_nav.scene import generate_scene
from quadruped_nav.experiment import run_episode
from quadruped_nav.learned import PileNetwork,LearnedPlanner


@pytest.fixture(scope='module',autouse=True)
def threads():torch.set_num_threads(2)

@pytest.mark.parametrize('family',TERRAINS)
def test_new_terrains_compile_and_are_reproducible(family,tmp_path):
    scene=generate_terrain(family,7,42)
    assert scene==generate_terrain(family,7,42)
    assert scene!=generate_terrain(family,7,43)
    model=mujoco.MjModel.from_xml_path(str(build_mjcf(scene,tmp_path/'terrain.xml')))
    assert model.nu==12 and len(scene['stones'])>0
    terrain=Terrain.from_scene(scene)
    assert len(terrain.piles)==len(scene['stones'])+2


def test_observations_actions_and_rewards_have_frozen_contract():
    env=GoalEnv(EnvConfig(num_envs=2,noise=0))
    assert np.all(env.model.geom_margin==0) and np.all(env.model.pair_margin==0)
    obs,critic=env.observations(False)
    assert obs.shape==(2,873) and critic.shape==(2,876)
    assert torch.isfinite(obs).all()
    nextobs,nextcritic,reward,done,info=env.step(torch.zeros(2,12))
    assert torch.isfinite(reward).all()
    assert torch.allclose(reward,sum(info['reward_terms'].values()))
    assert info['terminal_critic'].shape==(2,876)
    assert nextobs.shape==obs.shape
    assert float(abs(env.backend.data[0].ctrl-env.home[7:].numpy()).max())<1e-6


def test_scan_matches_mujoco_vertical_rays_on_slanted_piles(tmp_path):
    scene=generate_terrain('slanted',19,42);height=HeightMap(scene,'cpu')
    model=mujoco.MjModel.from_xml_path(str(build_mjcf(scene,tmp_path/'slanted.xml')));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for p in scene['stones'][::max(1,len(scene['stones'])//12)]:
        point=np.array([p['x']+.015,p['y']+.01,3.])
        geom=np.zeros(1,dtype=np.int32)
        distance=mujoco.mj_ray(model,data,point,np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,geom)
        analytic=float(height.sample(torch.tensor(point[None,:2],dtype=torch.float32))[0])
        assert abs((3-distance)-analytic)<.02


def test_time_limit_bootstrap_differs_from_true_terminal():
    reward=torch.tensor([[1.,1.]])
    val=torch.zeros_like(reward);nxt=torch.tensor([[10.,10.]])
    term=torch.tensor([[False,True]]);done=torch.ones_like(term)
    adv,_=advantages(reward,val,nxt,term,done,.9,.95)
    assert adv.tolist()==[[10.,1.]]


def test_symmetry_is_an_involution():
    x=torch.randn(5,873);a=torch.randn(5,12)
    assert torch.equal(mirror_observation(mirror_observation(x)),x)
    assert torch.equal(mirror_action(mirror_action(a)),a)
    assert mirror_observation(x)[0,43]==-x[0,43]


def test_training_export_reload_and_resume(tmp_path):
    cfg=TrainConfig(env=EnvConfig(num_envs=2,noise=0,curriculum='none'),updates=1,rollout_steps=4,epochs=1,minibatch_size=8,hidden=[32],checkpoint_every=1,output=str(tmp_path/'run'))
    checkpoint=train(cfg)
    model,loaded=load_controller(checkpoint)
    assert loaded['steps']==8
    archive=export_bundle(checkpoint,tmp_path/'bundle')
    assert archive.exists()
    actor=torch.jit.load(str(tmp_path/'bundle/actor.ts'))
    x=torch.randn(2,873)
    assert torch.allclose(actor(x),model.act(x),atol=1e-6)
    cfg.updates=2;train(cfg,resume=checkpoint)
    _,resumed=load_controller(checkpoint);assert resumed['steps']==16
    corrupt=dict(resumed);corrupt['observation_spec']={};torch.save(corrupt,tmp_path/'bad.pt')
    with pytest.raises(ValueError):load_controller(tmp_path/'bad.pt')


@pytest.mark.parametrize('variant',['L0','L1','L2','L3'])
def test_learned_planner_masks_and_memory_reset(variant):
    env=AbstractEnv(Terrain.from_scene(generate_scene('open')))
    planner=LearnedPlanner(PileNetwork(variant));obs=env.observe()
    choice=planner.select_subgoal(obs)
    assert choice in {p.id for p,ok in zip(obs.visible_piles,obs.action_mask) if ok}
    assert planner.visits
    planner.reset();assert not planner.visits and planner.hidden is None


def test_optimistic_escapes_and_resets():
    planner=OptimisticPlanner()
    env=AbstractEnv(Terrain.from_scene(generate_scene('dead_end')))
    result=run_episode(env,planner,'optimistic')
    assert result['metrics']['success'] and planner.memory
    again=run_episode(env,planner,'optimistic')
    assert result==again


def test_fall_is_terminal_and_reset_has_no_cross_episode_actions():
    env=GoalEnv(EnvConfig(num_envs=1,noise=0))
    q=env.home[None].clone();q[:,0]=-10;q[:,2]=-.5;env.backend.reset(torch.tensor([0]),q)
    obs,_,_,done,info=env.step(torch.ones(1,12)*.1)
    assert done[0] and info['terminated'][0] and info['fall'][0]
    assert torch.equal(obs[:,30:42],torch.zeros(1,12))


def test_cpu_config_does_not_silently_claim_cuda():
    with pytest.raises(ValueError):GoalEnv(EnvConfig(num_envs=1,backend='native',device='cuda:0'))


def test_resume_restores_rotating_terrain_schedule(tmp_path):
    cfg=TrainConfig(env=EnvConfig(num_envs=1,noise=0,curriculum='none'),updates=2,rollout_steps=2,epochs=1,minibatch_size=2,hidden=[16],checkpoint_every=1,terrain_every=1,terrain_families=['flat','open'],output=str(tmp_path/'rotating'))
    checkpoint=train(cfg)
    fresh=replace(cfg,env=EnvConfig(num_envs=1,noise=0,curriculum='none'),updates=3)
    train(fresh,resume=checkpoint)
    _,result=load_controller(checkpoint)
    assert result['update']==3 and result['config']['terrain_families']==['flat','open']


def test_calibrated_failure_does_not_award_goal(tmp_path):
    from quadruped_nav.calibrate import CalibratedEnv
    from quadruped_nav.abstract import Pile
    path=tmp_path/'calibration.json';path.write_text(json.dumps({'bins':[dict(family='open',row=0,length_min=0,length_max=2,probability=0)]}))
    terrain=Terrain((Pile('a',0,0,0,.5),Pile('b',.6,0,0,.5)),'a','b')
    env=CalibratedEnv(AbstractEnv(terrain),path,'open',0)
    _,done=env.step('b')
    assert done and env.status=='fall' and env.current_id=='a'
    assert env.states[-1]['status']=='fall' and env.distance_m==0


def test_pocket_escape_requires_entry_exit_and_goal_success():
    from quadruped_nav.benchmark import pocket_metrics
    scene=generate_terrain('dead_end',0,0,pocket_depth=2)
    path=[(0,0),(2,0),(0,0),(0,2),(7,2)]
    assert pocket_metrics(scene,path,True)['pocket_escaped']
    assert not pocket_metrics(scene,path,False)['pocket_escaped']
    assert not pocket_metrics(scene,[(0,2),(7,2)],True)['pocket_entered']
