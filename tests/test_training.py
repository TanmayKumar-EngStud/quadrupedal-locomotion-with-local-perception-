import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco
torch = pytest.importorskip('torch')
from quadruped_nav.terrain import generate_terrain, TERRAINS
from quadruped_nav.mjcf import build_mjcf
from quadruped_nav.rl.config import EnvConfig, TrainConfig, OBSERVATION_SPEC, default_rewards
from quadruped_nav.rl.env import GoalEnv, robots_per_type
from quadruped_nav.rl.terrain_grid import TerrainGrid, HeightGrid, TERRAIN_TYPES, pile_pit_depth, pit_depth
from quadruped_nav.rl.robot import build_training_model
from quadruped_nav.rl.policy import load_controller, export_bundle
from quadruped_nav.rl.ppo import Rollout, gae
from quadruped_nav.rl import rewards
from quadruped_nav.train import train
from quadruped_nav.abstract import AbstractEnv, Terrain
from quadruped_nav.planners import OptimisticPlanner
from quadruped_nav.scene import generate_scene
from quadruped_nav.experiment import run_episode
from quadruped_nav.learned import PileNetwork, LearnedPlanner


@pytest.fixture(scope='module', autouse=True)
def threads(): torch.set_num_threads(2)


def small_config(**kwargs):
    return EnvConfig(**{**dict(num_envs=4, noise=0., terrain_levels=2, max_start_level=1, tile_size=6.,
                               terrain_resolution=.1), **kwargs})


@pytest.mark.parametrize('family', TERRAINS)
def test_project_scenes_compile_and_are_reproducible(family, tmp_path):
    scene = generate_terrain(family, 7, 42)
    assert scene == generate_terrain(family, 7, 42)
    assert scene != generate_terrain(family, 7, 43)
    model = mujoco.MjModel.from_xml_path(str(build_mjcf(scene, tmp_path / 'terrain.xml')))
    assert model.nu == 12 and len(scene['stones']) > 0
    assert len(Terrain.from_scene(scene).piles) == len(scene['stones']) + 2


def test_terrain_grid_is_reproducible_and_tiles_join_without_cliffs():
    a, b = TerrainGrid(resolution=.1, seed=3), TerrainGrid(resolution=.1, seed=3)
    assert np.array_equal(a.heights, b.heights)
    assert not np.array_equal(a.heights, TerrainGrid(resolution=.1, seed=4).heights)
    n = round(a.tile_size / a.resolution); border = round(a.border / a.resolution)
    # First row of the rough, slope and stair tiles: only rough noise, no cliffs.
    edges = a.heights[[border + c * n for c in range(4)], border:-border]
    assert abs(edges).max() < .11
    assert a.heights.shape == (2 * border + len(TERRAIN_TYPES) * n + 1, 2 * border + a.levels * n + 1)


def test_height_samples_match_mujoco_rays_on_the_heightfield():
    grid = TerrainGrid(types=('rough', 'slope'), levels=2, tile_size=6., resolution=.1)
    model = build_training_model(grid); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    sampler = grid.height_grid('cpu'); rng = np.random.default_rng(0)
    groups = np.zeros(6, dtype=np.uint8); groups[0] = 1
    for x, y in rng.uniform([grid.border, grid.border], [grid.size_x - grid.border, grid.size_y - grid.border], (40, 2)):
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.]), np.array([0., 0., -1.]), groups, 1, -1, geom)
        assert abs((5 - distance) - float(sampler.sample(torch.tensor([[x, y]], dtype=torch.float32))[0])) < .02


def test_scene_height_raster_matches_mujoco_rays_on_slanted_piles(tmp_path):
    scene = generate_terrain('slanted', 19, 42); height = HeightGrid.from_scene(scene)
    model = mujoco.MjModel.from_xml_path(str(build_mjcf(scene, tmp_path / 'slanted.xml'))); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for p in scene['stones'][::max(1, len(scene['stones']) // 12)]:
        point = np.array([p['x'] + .015, p['y'] + .01, 3.]); geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, geom)
        assert abs((3 - distance) - float(height.sample(torch.tensor(point[None, :2], dtype=torch.float32))[0])) < .03


def test_pile_tiles_leave_v1_tiles_unchanged_and_hold_3cm_gaps():
    v1 = TerrainGrid(TERRAIN_TYPES, levels=2, tile_size=4., seed=5)
    v2 = TerrainGrid(TERRAIN_TYPES + ('piles',), levels=2, tile_size=4., seed=5)
    assert v2.factor == 2 and v2.height_resolution == .025
    v1_tiles, v2_tiles = {t[0]: t[3] for t in v1.tiles()}, {t[0]: t[3] for t in v2.tiles()}
    for name, heights in v1_tiles.items():
        if not name.endswith('_6'):                     # gaps borders the pile tile now
            assert np.array_equal(heights, v2_tiles[name])
    pile = v2_tiles['tile_0_7']
    assert pile.shape == (161, 161)                     # 4 m at 2.5 cm
    # Level 0: 0.50 m piles with 3 cm gaps (pitch 0.53 m); each gap crossed by a row holds a pit node.
    row = pile[np.argmax((pile[:50] > -.01).sum(1)), :-1]   # a row across piles, away from the spawn platform
    edges = np.flatnonzero((row[1:] < -pile_pit_depth(0) / 2) & (row[:-1] > -.01))
    assert len(edges) >= 4 and np.allclose(np.diff(edges) * .025, .53, atol=.026)


def test_fine_heightfield_matches_height_samples_at_grid_nodes():
    grid = TerrainGrid(('gaps', 'piles'), levels=2, tile_size=4., seed=2)
    model = build_training_model(grid); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    sampler, rng = grid.height_grid('cpu'), np.random.default_rng(1)
    groups = np.zeros(6, dtype=np.uint8); groups[0] = 1
    nodes = rng.integers([grid.border / .025 + 1] * 2, [(grid.size_y - grid.border) / .025 - 1, (grid.size_x - grid.border) / .025 - 1], (60, 2))
    for iy, ix in nodes:
        # Coarse tiles only define their own (even) nodes; fine tiles every node.
        x, y = ix * .025, iy * .025
        if y < grid.border + grid.tile_size: x, y = 2 * (ix // 2) * .025, 2 * (iy // 2) * .025
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.]), np.array([0., 0., -1.]), groups, 1, -1, geom)
        assert abs((5 - distance) - float(sampler.sample(torch.tensor([[x, y]], dtype=torch.float32))[0])) < 2e-3


def test_terrain_weights_split_the_robots():
    assert robots_per_type(4096, [1, 1, 1, 1, 1, 1.2, 1, 4.8]).tolist() == [342, 341, 341, 341, 341, 410, 341, 1639]
    env = GoalEnv(small_config(num_envs=5, terrain_types=['rough', 'piles'], terrain_weights=[1, 4], tile_size=4.))
    assert env.column.tolist() == [0, 1, 1, 1, 1]
    with pytest.raises(ValueError): small_config(terrain_weights=[1]).validate()


def test_heightfield_scene_is_the_sensed_height_map(tmp_path):
    scene = generate_terrain('rough', 9, 4)
    boxes = GoalEnv(small_config(num_envs=1), scene)
    field = GoalEnv(small_config(num_envs=1), scene, ground='heightfield')
    assert torch.equal(boxes.terrain.values, field.terrain.values)
    assert not any(field.model.geom(g).name.startswith('T') for g in range(field.model.ngeom))
    data = field.backend.data[0]
    groups = np.zeros(6, dtype=np.uint8); groups[0] = 1
    for p in scene['stones'][::max(1, len(scene['stones']) // 15)] + scene['platforms']:
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(field.model, data, np.array([p['x'], p['y'], 3.]), np.array([0., 0., -1.]), groups, 1, -1, geom)
        assert abs((3 - distance) - p['z_top']) < 1e-3
    obs, *_ = field.step(torch.zeros(1, 12))
    assert torch.isfinite(obs).all()


def test_terrain_noise_is_reproducible_and_keeps_edges_and_spawn_platforms():
    kinds = ('slope', 'stairs_up', 'stairs_down', 'obstacles', 'stepping_stones', 'gaps', 'piles')
    regular = TerrainGrid(kinds, levels=3, tile_size=4., seed=2)
    noisy = TerrainGrid(kinds, levels=3, tile_size=4., seed=2, noise=1.)
    assert np.array_equal(noisy.heights, TerrainGrid(kinds, levels=3, tile_size=4., seed=2, noise=1.).heights)
    assert np.array_equal(regular.heights, TerrainGrid(kinds, levels=3, tile_size=4., seed=2, noise=0.).heights)
    regular_tiles, noisy_tiles = {t[0]: t[3] for t in regular.tiles()}, {t[0]: t[3] for t in noisy.tiles()}
    for column, kind in enumerate(kinds):
        for level in range(3):
            name = f'tile_{level}_{column}'
            if level or kind not in ('slope', 'stairs_up', 'stairs_down'):   # level-0 slopes have no level change to blur
                assert not np.array_equal(regular_tiles[name], noisy_tiles[name]), name
    assert (noisy.spawn_height == 0).all() == (regular.spawn_height == 0).all()
    for column, kind in enumerate(kinds[:3]):            # slopes and stairs still end at height 0: no cliffs
        tile = noisy_tiles[f'tile_2_{column}']
        assert abs(tile[0]).max() < 1e-6 and abs(tile[:, 0]).max() < 1e-6, kind


def test_noisy_stones_stay_above_their_pits_and_stairs_stay_walkable():
    grid = TerrainGrid(('stepping_stones', 'piles', 'stairs_up'), levels=10, tile_size=4., seed=3, noise=1.)
    for name, _, _, tile in grid.tiles():
        level, column = map(int, name.split('_')[1:])
        kind, d = grid.columns[column][0], level / 9
        inner = tile[:-1, :-1]                           # the last row and column belong to the neighbouring tile
        if kind == 'stairs_up':
            risers = np.unique(np.round(abs(np.diff(inner, axis=0)), 4))
            assert risers.max() <= .2 + 1e-6
            continue
        pit = pit_depth(d) if kind == 'stepping_stones' else pile_pit_depth(d)
        tops = inner[inner > -pit + 1e-6]
        assert tops.min() > -pit / 2, (name, tops.min())   # every stone node is far above the pit floor


def test_tile_variants_and_robots_drawing_them():
    grid = TerrainGrid(('piles',), levels=2, tile_size=4., seed=1, variants=3, noise=.5)
    assert len(grid.columns) == 3 and grid.spawn_height.shape == (2, 3)
    tiles = {t[0]: t[3] for t in grid.tiles()}
    assert not np.array_equal(tiles['tile_1_0'], tiles['tile_1_1'])
    env = GoalEnv(small_config(num_envs=16, terrain_types=['rough', 'piles'], tile_size=4., terrain_variants=2,
                               terrain_noise=.5))
    assert set(env.variant.tolist()) == {0, 1}
    tile = env._tile(torch.arange(16))
    assert int(tile.max()) < len(env.grid.columns)
    q = env.backend.tensor('qpos')[:, :2]
    tile_y = env.tile_y[tile]
    assert ((q[:, 1] > tile_y) & (q[:, 1] < tile_y + 4.)).all()   # every robot stands on its own tile


def test_time_budget_stops_training_and_validation_tiles_are_unseen(tmp_path):
    cfg = TrainConfig(env=small_config(num_envs=2), updates=50, rollout_steps=2, epochs=1, minibatches=1, hidden=[16],
                      checkpoint_every=10, evaluation_every=10, max_hours=1e-9, validation_seed=100,
                      output=str(tmp_path / 'run'))
    checkpoint = train(cfg)
    _, saved = load_controller(checkpoint)
    assert saved['update'] == 1 and (tmp_path / 'run/update_000001.pt').exists()
    assert len((tmp_path / 'run/evaluation.jsonl').read_text().splitlines()) == 1


def test_observation_contract_and_reward_bookkeeping():
    env = GoalEnv(small_config())
    obs, critic = env.observations(False)
    assert obs.shape == (4, OBSERVATION_SPEC['actor_dim']) and critic.shape == (4, OBSERVATION_SPEC['critic_dim'])
    assert OBSERVATION_SPEC['actor_dim'] - 4 * len(OBSERVATION_SPEC['foot_scan']) - len(OBSERVATION_SPEC['body_scan']) == 55
    assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
    _, _, reward, done, info = env.step(torch.zeros(4, 12))
    dense = sum(v for k, v in info['reward_terms'].items() if k not in rewards.EVENTS)
    events = sum(v for k, v in info['reward_terms'].items() if k in rewards.EVENTS)
    assert torch.allclose(reward, dense + events)
    assert info['terminal_critic'].shape == critic.shape and not done.any()


def test_standing_still_earns_less_than_walking_to_the_target():
    command = torch.tensor([[.8, 0., 0.]])
    def state(velocity):
        return SimpleNamespace(lin_vel_body=torch.tensor([[velocity, 0., 0.]]), lin_vel_world=torch.tensor([[velocity, 0., 0.]]),
                               ang_vel_body=torch.zeros(1, 3), command=command, target_direction=torch.tensor([[1., 0.]]),
                               max_speed=torch.tensor([.8]), moving=torch.tensor([True]))
    still, walking = state(0.), state(.8)
    total = lambda s: sum(default_rewards()[n] * getattr(rewards, n)(s) for n in ('tracking_lin_vel', 'tracking_ang_vel', 'goal_progress'))
    assert float(total(walking)) > 3 * float(total(still))
    assert float(rewards.goal_progress(state(-.5))) < 0   # walking away is penalised


def test_arrival_is_paid_once_then_holds_then_chains_a_new_target():
    env = GoalEnv(small_config(num_envs=1, hold_seconds=[.1, .1]))
    position = env.backend.tensor('qpos')[0, :2].clone()
    env.set_target(position[None] + torch.tensor([[.1, 0.]]))
    _, _, reward, _, info = env.step(torch.zeros(1, 12))
    assert info['arrived'][0] and float(info['reward_terms']['arrival'][0]) == default_rewards()['arrival']
    assert float(env._command(*env._target_geometry(env._kinematics())[::2]).abs().sum()) == 0
    _, _, _, _, info = env.step(torch.zeros(1, 12))
    assert not info['arrived'][0]
    for _ in range(10): env.step(torch.zeros(1, 12))
    assert not env.reached[0] and float(env.start_distance[0]) > .5   # a new target was assigned


def test_fall_is_terminal_resets_and_demotes():
    env = GoalEnv(small_config(num_envs=1))
    env.level[:] = 1
    q = env.home[None].clone(); q[:, :2] = env.backend.tensor('qpos')[0, :2]; q[:, 2] = -.5
    env.backend.reset(torch.tensor([0]), q)
    obs, _, reward, done, info = env.step(torch.ones(1, 12))
    assert done[0] and info['terminated'][0] and info['fall'][0]
    assert float(reward[0]) <= default_rewards()['fall'] + 1
    assert int(env.level[0]) == 0 and torch.equal(env.previous_action, torch.zeros(1, 12))


def test_curriculum_promotes_after_clean_episodes():
    env = GoalEnv(small_config(num_envs=2))
    env.level[:] = 0; env.episode_arrivals[:] = torch.tensor([3., 0.]); env.episode_misses[:] = torch.tensor([0., 2.])
    env._update_levels(torch.arange(2), torch.tensor([False, False]))
    assert env.level.tolist() == [1, 0]


def test_generalized_advantage_estimation_respects_episode_ends():
    rollout = Rollout(actor_obs=None, critic_obs=None, actions=None, log_probs=None,
                      values=torch.zeros(2, 2), rewards=torch.ones(2, 2), dones=torch.tensor([[False, True], [False, False]]),
                      last_values=torch.tensor([10., 10.]))
    advantages, _ = gae(rollout, .9, 1.)
    assert torch.allclose(advantages[:, 0], torch.tensor([1 + .9 * (1 + .9 * 10), 1 + .9 * 10]))
    assert float(advantages[0, 1]) == 1.   # no bootstrap across the episode boundary


def test_training_export_reload_and_resume(tmp_path):
    cfg = TrainConfig(env=small_config(num_envs=2), updates=1, rollout_steps=4, epochs=1, minibatches=2, hidden=[32],
                      checkpoint_every=1, evaluation_every=1, output=str(tmp_path / 'run'))
    checkpoint = train(cfg)
    model, loaded = load_controller(checkpoint)
    assert loaded['steps'] == 8
    assert (tmp_path / 'run/evaluation.jsonl').exists() and json.loads((tmp_path / 'run/metrics.jsonl').read_text())['update'] == 1
    archive = export_bundle(checkpoint, tmp_path / 'bundle'); assert archive.exists()
    actor = torch.jit.load(str(tmp_path / 'bundle/actor.ts'))
    x = torch.randn(2, OBSERVATION_SPEC['actor_dim'])
    assert torch.allclose(actor(x), model.act(x), atol=1e-6)
    cfg.updates = 2; train(cfg, resume=checkpoint)
    _, resumed = load_controller(checkpoint); assert resumed['steps'] == 16
    corrupt = dict(resumed); corrupt['observation_spec'] = {}; torch.save(corrupt, tmp_path / 'bad.pt')
    with pytest.raises(ValueError): load_controller(tmp_path / 'bad.pt')


def test_planner_execution_on_a_project_scene(tmp_path):
    from quadruped_nav.evaluate import evaluate
    cfg = TrainConfig(env=small_config(num_envs=1), updates=1, rollout_steps=2, epochs=1, minibatches=1, hidden=[16],
                      checkpoint_every=1, evaluation_every=0, output=str(tmp_path / 'run'))
    result = evaluate(train(cfg), generate_terrain('open', 0, 1), 'optimistic', seconds=.5)
    assert result['status'] in ('timeout', 'fall') and result['decisions']


def test_cpu_config_does_not_silently_claim_cuda():
    with pytest.raises(ValueError): GoalEnv(EnvConfig(num_envs=1, backend='native', device='cuda:0'))


@pytest.mark.parametrize('variant', ['L0', 'L1', 'L2', 'L3'])
def test_learned_planner_masks_and_memory_reset(variant):
    env = AbstractEnv(Terrain.from_scene(generate_scene('open')))
    planner = LearnedPlanner(PileNetwork(variant)); obs = env.observe()
    choice = planner.select_subgoal(obs)
    assert choice in {p.id for p, ok in zip(obs.visible_piles, obs.action_mask) if ok}
    assert planner.visits
    planner.reset(); assert not planner.visits and planner.hidden is None


def test_optimistic_escapes_and_resets():
    planner = OptimisticPlanner()
    env = AbstractEnv(Terrain.from_scene(generate_scene('dead_end')))
    result = run_episode(env, planner, 'optimistic')
    assert result['metrics']['success'] and planner.memory
    assert result == run_episode(env, planner, 'optimistic')


def test_calibrated_failure_does_not_award_goal(tmp_path):
    from quadruped_nav.calibrate import CalibratedEnv
    from quadruped_nav.abstract import Pile
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({'bins': [dict(family='open', row=0, length_min=0, length_max=2, probability=0)]}))
    terrain = Terrain((Pile('a', 0, 0, 0, .5), Pile('b', .6, 0, 0, .5)), 'a', 'b')
    env = CalibratedEnv(AbstractEnv(terrain), path, 'open', 0)
    _, done = env.step('b')
    assert done and env.status == 'fall' and env.current_id == 'a'
    assert env.states[-1]['status'] == 'fall' and env.distance_m == 0


def test_pocket_escape_requires_entry_exit_and_goal_success():
    from quadruped_nav.benchmark import pocket_metrics
    scene = generate_terrain('dead_end', 0, 0, pocket_depth=2)
    path = [(0, 0), (2, 0), (0, 0), (0, 2), (7, 2)]
    assert pocket_metrics(scene, path, True)['pocket_escaped']
    assert not pocket_metrics(scene, path, False)['pocket_escaped']
    assert not pocket_metrics(scene, [(0, 2), (7, 2)], True)['pocket_entered']
