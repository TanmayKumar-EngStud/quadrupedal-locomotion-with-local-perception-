import json
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco
torch = pytest.importorskip('torch')
from quadruped_nav.terrain import generate_terrain, TERRAINS
from quadruped_nav.mjcf import build_mjcf
from quadruped_nav.rl.config import EnvConfig, TrainConfig, OBSERVATION_SPEC, observation_spec, default_rewards, default_reward_params
from quadruped_nav.rl.env import GoalEnv, robots_per_type, FOOT_RADIUS
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


def test_mix_family_and_pile_irregularities():
    from quadruped_nav.terrain import MIX_FAMILIES
    mix = generate_terrain('mix', 9, 3)
    sections = {f: (a, b) for f, a, b in mix['metadata']['mix_sections']}
    assert sorted(sections) == sorted(MIX_FAMILIES)                                   # every family once
    inside = lambda f: [s for s in mix['stones'] if sections[f][0] <= s['x'] < sections[f][1]]
    t9 = 9 / 19
    assert all(abs(s['y']) <= .55 - .3 * t9 + 1e-9 for s in inside('bridge'))         # a narrow strip
    assert len({round(s['z_top'], 4) for s in inside('rough')}) > 5                   # random heights
    assert any(abs(s['roll']) > .01 for s in inside('slanted'))                       # tilted tops
    assert max(s['z_top'] for s in inside('steps')) > .03                             # a raised terrace
    assert len(inside('holes')) < len(inside('rough'))                                # missing piles
    wall = mix['obstacles'][0]
    assert not any(abs(s['x'] - wall['x']) < .3 and abs(s['y'] - wall['y']) < 1. for s in mix['stones'])   # slalom wall
    moat_band = [s for s in inside('moat') if .35 < s['x'] - sections['moat'][0] < 1.15]
    assert moat_band and all(abs(s['y'] - mix['metadata']['bridge_y']) <= .6 + 1e-9 for s in moat_band)  # one crossing
    assert generate_terrain('open', 5, 1, irregularity=0.) == generate_terrain('open', 5, 1)
    first, again = (generate_terrain('open', 5, 1, irregularity=1., irregularity_seed=8) for _ in range(2))
    assert first == again and first != generate_terrain('open', 5, 1, irregularity=1., irregularity_seed=9)
    assert generate_terrain('open', 5, 1, irregularity=1.) != generate_terrain('open', 5, 1, irregularity=1.)   # new layout every run
    regular = generate_terrain('open', 5, 1)
    pitch = regular['parameters']['stone_size'] + regular['parameters']['gap']
    for s, b in zip(first['stones'], regular['stones']):
        assert abs(s['x'] - b['x']) <= (pitch - s['size']) / 2 + 1e-9 and abs(s['y'] - b['y']) <= (pitch - s['size']) / 2 + 1e-9
        assert max(abs(s['roll']), abs(s['pitch'])) <= math.radians(15) + 1e-9 and s['size'] >= .12
    Terrain.from_scene(first); HeightGrid.from_scene(first)                            # planner and height map accept it


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
    sample = lambda points: sampler.sample(torch.tensor(points, dtype=torch.float32)).numpy()
    checked = 0
    for iy, ix in nodes:
        # Coarse tiles only define their own (even) nodes; fine tiles every node.
        x, y, step = ix * .025, iy * .025, .025
        if y < grid.border + grid.tile_size: x, y, step = 2 * (ix // 2) * .025, 2 * (iy // 2) * .025, .05
        # A ray exactly on a grid vertex or tile seam can slip to the 0.5 m pit floor depending on the
        # platform's rounding. So only flat neighbourhoods are checked, with the ray inside the cell.
        neighbourhood = sample([[x + dx * step, y + dy * step] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        if np.ptp(neighbourhood) > 1e-6: continue
        x, y = x + .3 * step, y + .4 * step
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.]), np.array([0., 0., -1.]), groups, 1, -1, geom)
        assert abs((5 - distance) - float(sample([[x, y]])[0])) < 2e-3
        checked += 1
    assert checked >= 20


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


def test_noise_ramp_grows_the_irregularities_with_the_level():
    kinds = ('stepping_stones', 'piles')
    ramp = TerrainGrid(kinds, levels=5, tile_size=4., seed=4, noise=1., noise_ramp=4)
    assert [ramp.level_noise(level) for level in range(5)] == [0., .25, .5, .75, 1.]
    tiles = lambda grid: {t[0]: t[3] for t in grid.tiles()}
    ramped = tiles(ramp)
    for noise, level in ((0., 0), (.5, 2), (1., 4)):     # each level matches a grid with that level's noise everywhere
        same = tiles(TerrainGrid(kinds, levels=5, tile_size=4., seed=4, noise=noise))
        for column in range(2):              # the last row and column belong to the neighbouring tiles
            assert np.array_equal(ramped[f'tile_{level}_{column}'][:-1, :-1], same[f'tile_{level}_{column}'][:-1, :-1])
    env = GoalEnv(small_config(terrain_types=['piles'], tile_size=4., terrain_noise=1., terrain_noise_ramp=4))
    assert env.grid.level_noise(1) == .25


def test_time_budget_stops_training_and_validation_tiles_are_unseen(tmp_path):
    cfg = TrainConfig(env=small_config(num_envs=2), updates=50, rollout_steps=2, epochs=1, minibatches=1, hidden=[16],
                      checkpoint_every=10, evaluation_every=10, max_hours=1e-9, validation_seed=100,
                      output=str(tmp_path / 'run'))
    checkpoint = train(cfg)
    _, saved = load_controller(checkpoint)
    assert saved['update'] == 1 and (tmp_path / 'run/update_000001.pt').exists()
    assert len((tmp_path / 'run/evaluation.jsonl').read_text().splitlines()) == 1


def test_validation_size_follows_the_config(tmp_path):
    cfg = TrainConfig(env=small_config(num_envs=2), updates=1, rollout_steps=2, epochs=1, minibatches=1, hidden=[16],
                      checkpoint_every=1, evaluation_every=1, evaluation_trials=2, evaluation_levels=[0, 1],
                      output=str(tmp_path / 'run'))
    train(cfg)
    record = json.loads((tmp_path / 'run/evaluation.jsonl').read_text().splitlines()[0])
    assert record['trials_per_cell'] == 2 and record['levels'] == [0, 1]
    assert record['overall']['trials'] == len(cfg.env.terrain_types) * 2 * 2


def test_arrival_speed_and_feet_in_gap_penalties():
    state = SimpleNamespace(speed_xy=torch.tensor([.4, 1., 1.5, 1.5]), newly_arrived=torch.tensor([True, True, True, False]),
                            foot_depth=torch.tensor([[0., .01, .05, .5]] * 4))
    params = default_reward_params()
    assert torch.allclose(rewards._reward_arrival_speed(state, params), torch.tensor([0., .25, 1., 0.]))   # only on arrival, above 0.5 m/s
    assert torch.allclose(rewards._reward_feet_in_gap(state, params), torch.full((4,), .02 + .07))          # 3 cm deadband, 10 cm cap
    weights = {**default_rewards(), 'arrival_speed': -10., 'feet_in_gap': -20.}
    env = GoalEnv(small_config(num_envs=2, terrain_types=['rough'], rewards=weights))
    for _ in range(5):
        _, _, _, _, info = env.step(torch.zeros(2, 12))
    assert float(info['reward_terms']['feet_in_gap'].abs().max()) < 1e-6    # standing on level ground costs nothing


def test_feet_edge_penalty_marks_supported_feet_near_a_drop():
    env = GoalEnv(small_config(num_envs=1, terrain_types=['rough']))
    step = np.zeros((101, 101), dtype=np.float32); step[:, 50:] = -.5         # a 0.5 m drop at x = 0.5 m
    env.terrain = HeightGrid(step, 0., 0., .01, outside=0.)
    feet = torch.tensor([[[.30, .5, FOOT_RADIUS], [.47, .5, FOOT_RADIUS],    # far from the edge; 3 cm from it
                          [.52, .5, FOOT_RADIUS], [.47, .5, .20]]])          # centre over the drop; in the air
    drop = env._edge_drop(SimpleNamespace(feet=feet))
    assert torch.allclose(drop, torch.tensor([[0., .5, .5, 0.]]), atol=1e-4)
    assert float(rewards._reward_feet_edge(SimpleNamespace(edge_drop=drop), default_reward_params())) == 2.
    env = GoalEnv(small_config(num_envs=2, terrain_types=['rough'], rewards={'feet_edge': -1.}))
    for _ in range(5):
        _, _, _, _, info = env.step(torch.zeros(2, 12))
    assert float(info['reward_terms']['feet_edge'].abs().max()) == 0.       # level ground has no edges


def test_feet_clearance_pays_the_swing_height_on_touchdown():
    state = SimpleNamespace(touchdown=torch.tensor([[True, True, False, True]]), moving=torch.tensor([True]),
                            air_time_on_touchdown=torch.tensor([[.3, .05, .3, .3]]), swing_peak=torch.tensor([[.05, .2, .2, .2]]))
    # 5 cm of a 10 cm target; a swing too short to count; no touchdown; capped at 1
    assert float(rewards._reward_feet_clearance(state, default_reward_params())) == pytest.approx(1.5)
    env = GoalEnv(small_config(num_envs=2, terrain_types=['rough'], rewards={'feet_clearance': .5}))
    paid, generator = 0., torch.Generator().manual_seed(0)
    for _ in range(60):
        _, _, _, _, info = env.step(torch.randn(2, 12, generator=generator))
        paid += float(info['reward_terms']['feet_clearance'].sum())
    assert paid > 0 and (env.swing_peak >= 0).all()                  # stepping feet earn it; peaks reset on touchdown


def test_heading_alignment_rewards_walking_forwards():
    state = SimpleNamespace(heading_cos=torch.tensor([1., -1., 0., 1., 1.]), speed_xy=torch.tensor([.8, .8, .8, .1, .8]),
                            moving=torch.tensor([True, True, True, True, True]), target_far=torch.tensor([True, True, True, True, False]))
    # forwards, backwards, sideways; too slow to count; close to the target (stepping back is free there)
    assert torch.allclose(rewards._reward_heading_alignment(state, default_reward_params()), torch.tensor([1., -1., 0., 0., 0.]))
    env = GoalEnv(small_config(num_envs=2, terrain_types=['rough'], rewards={'heading_alignment': .5}))
    for _ in range(5):
        _, _, _, _, info = env.step(torch.zeros(2, 12))
    assert torch.isfinite(info['reward_terms']['heading_alignment']).all()


def test_run_configs_change_only_the_reward_values_they_list():
    cfg = EnvConfig(rewards={'feet_in_gap': -20.}, reward_params={'gap_deadband': .05})
    assert cfg.rewards == {**default_rewards(), 'feet_in_gap': -20.}
    assert cfg.reward_params == {**default_reward_params(), 'gap_deadband': .05}
    assert list(cfg.rewards) == list(default_rewards())          # terms are added in reward_config() order
    assert set(rewards.TERMS) == set(default_rewards())          # one _reward_<name>() per weight
    state = SimpleNamespace(foot_depth=torch.tensor([[.04, 0., 0., 0.]]))
    assert float(rewards._reward_feet_in_gap(state, cfg.reward_params)) == 0.   # 4 cm is inside the new 5 cm deadband
    with pytest.raises(ValueError): EnvConfig(reward_params={'tracking_sigmaa': .3}).validate()


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
    total = lambda s: sum(default_rewards()[n] * rewards.TERMS[n](s, default_reward_params())
                          for n in ('tracking_lin_vel', 'tracking_ang_vel', 'goal_progress'))
    assert float(total(walking)) > 3 * float(total(still))
    assert float(rewards._reward_goal_progress(state(-.5), default_reward_params())) < 0   # walking away is penalised


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


def test_relaxed_curriculum_promotes_at_60_percent_and_demotes_only_without_progress():
    env = GoalEnv(small_config(num_envs=6, terrain_levels=3, curriculum_rule='relaxed'))
    env.level[:] = 1
    env.episode_arrivals[:] = torch.tensor([3., 2., 2., 0., 0., 1.])
    env.episode_misses[:] = torch.tensor([2., 0., 2., 1., 1., 3.])
    env.episode_progress[:] = torch.tensor([1., 1., 1., .1, .5, 1.])
    fell = torch.tensor([False, True, False, True, True, False])
    env._update_levels(torch.arange(6), fell)
    # 3 of 5 targets: up; 2 then a fall (2 of 3): up; 2 of 4: stay; no target and 10% closed: down;
    # no target but half the distance closed: stay despite the fall; 1 of 4: stay
    assert env.level.tolist() == [2, 2, 1, 0, 1, 1]
    strict = GoalEnv(small_config(num_envs=1, terrain_levels=3)); strict.level[:] = 1
    strict.episode_arrivals[:] = 2.; strict._update_levels(torch.arange(1), torch.tensor([True]))
    assert strict.level.tolist() == [0]                 # the strict rule demotes on any fall
    env.reset(torch.arange(6)); assert float(env.episode_progress.abs().max()) == 0.


def test_feet49_scan_layout():
    spec = observation_spec('feet49')
    assert spec['actor_dim'] == 55 + 200 and len(spec['body_scan']) == 4 and [len(f) for f in spec['foot_scan']] == [49] * 4
    assert observation_spec() == OBSERVATION_SPEC and 'scan_layout' not in OBSERVATION_SPEC   # v1-v3 checkpoints still load
    env = GoalEnv(small_config(num_envs=1, scan_layout='feet49'))
    obs, critic = env.observations(False)
    assert obs.shape == (1, 255) and critic.shape == (1, 261) and torch.isfinite(obs).all()
    k = env._kinematics(); _, feet = env._scan_points(k)
    c, s = torch.cos(-k.yaw[0]), torch.sin(-k.yaw[0])
    d = feet[0] - k.pos[0, :2]
    local = torch.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], -1)    # (4, 49, 2), base frame
    boxes = [(local[i, :, 0].min(), local[i, :, 0].max(), local[i, :, 1].min(), local[i, :, 1].max()) for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):            # in the standing pose no two squares overlap
            assert boxes[i][1] < boxes[j][0] or boxes[j][1] < boxes[i][0] or boxes[i][3] < boxes[j][2] or boxes[j][3] < boxes[i][2]
    foot = k.feet[0, :, :2] - k.pos[0, :2]
    foot = torch.stack([c * foot[:, 0] - s * foot[:, 1], s * foot[:, 0] + c * foot[:, 1]], -1)
    assert torch.allclose(torch.linalg.vector_norm(local - foot[:, None], dim=-1).amin(-1), torch.zeros(4), atol=1e-4)   # one sample under each foot
    body, feet = env.scan_points()
    assert body.shape == (1, 4, 3) and feet.shape == (1, 4, 49, 3)


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
    other = replace(cfg, env=replace(cfg.env, scan_layout='feet49'), updates=1, output=str(tmp_path / 'feet49'))
    with pytest.raises(ValueError): train(other, resume=checkpoint)          # a v1-layout controller cannot be resumed
    feet49 = train(other)
    model, loaded = load_controller(feet49)
    assert loaded['observation_spec']['actor_dim'] == 255 and loaded['config']['env']['scan_layout'] == 'feet49'
    export_bundle(feet49, tmp_path / 'feet49_bundle')
    x = torch.randn(2, 255)
    assert torch.allclose(torch.jit.load(str(tmp_path / 'feet49_bundle/actor.ts'))(x), model.act(x), atol=1e-6)


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


def test_hops_judged_from_the_pile_centre_and_no_bounce_back():
    from quadruped_nav.abstract import Limits, observe_pose
    from quadruped_nav.evaluate import without_backtrack
    terrain = Terrain.from_scene(generate_terrain('open', 5, 10000)); limits = Limits()
    piles = {(round(p.x, 2), round(p.y, 2)): p.id for p in terrain.piles}
    seam = (1.17, -0.29, .3)                               # on the gap between two rows of piles
    allowed = lambda obs: {p.id for p, ok in zip(obs.visible_piles, obs.action_mask) if ok}
    ahead = piles[1.83, -0.46]
    assert ahead not in allowed(observe_pose(terrain, limits, seam, 0.))        # one line of feet along the gap
    centred = observe_pose(terrain, limits, seam, 0., snap=True)
    assert ahead in allowed(centred) and centred.current_pile_id not in allowed(centred)
    back = piles[0.91, -0.46]
    assert back in allowed(centred) and back not in allowed(without_backtrack(centred, back))
    only = replace(centred, action_mask=tuple(p.id == back for p in centred.visible_piles))
    assert allowed(without_backtrack(only, back)) == {back}   # still allowed when nothing else is


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
