import json
import math
import re
import sys

import pytest

from quadruped_nav.abstract import AbstractEnv, Limits, Pile, Terrain, straight_approach
from quadruped_nav.experiment import export_replay, run_comparison, run_episode
from quadruped_nav.planners import GreedyPlanner, OraclePlanner
from quadruped_nav.scene import generate_scene


def line_terrain():
    return Terrain(tuple(Pile(str(i), i*.6, 0, 0, .5) for i in range(6)), "0", "5")


def test_reset_seeded_run_and_local_information_boundary():
    terrain = line_terrain()
    changed_hidden = Terrain(terrain.piles + (Pile("secret", 20, 20, 0, 1),), "0", "5")
    env = AbstractEnv(terrain)
    observation = env.observe()
    assert observation == AbstractEnv(changed_hidden).observe()
    assert "5" not in [p.id for p in observation.visible_piles]
    assert all(math.hypot(*p.relative_top_center_m[:2]) <= 2 for p in observation.visible_piles)
    assert len(observation.visible_piles) == len(observation.action_mask)
    assert GreedyPlanner().select_subgoal(observation) == "3"
    env.step("3")
    assert env.reset() == observation
    assert env.events == [] and len(env.states) == 1
    a = run_episode(env, GreedyPlanner(), "greedy")
    b = run_episode(AbstractEnv(terrain), GreedyPlanner(), "greedy")
    assert a == b


def test_winding_support_does_not_allow_straight_gap_crossing():
    a, b = Pile("a", 0, 0, 0, .5), Pile("b", 1.2, 0, 0, .5)
    detour = [Pile(f"via{i}", x, .6, 0, .5) for i, x in enumerate([0, .6, 1.2])]
    assert not straight_approach(a, b, [a, b, *detour], Limits())
    assert straight_approach(a, b, [a, b, Pile("middle", .6, 0, 0, .5)], Limits())


def test_height_step_is_checked_between_supports():
    a, b = Pile("a", 0, 0, 0, .5), Pile("b", 1.2, 0, .1, .5)
    assert straight_approach(a, b, [a, Pile("step", .6, 0, .05, .5), b], Limits())
    assert not straight_approach(a, b, [a, Pile("step", .6, 0, .09, .5), b], Limits())


def test_invalid_or_hidden_target_does_not_change_state():
    env = AbstractEnv(line_terrain())
    before = env.state()
    for target in ("unknown", "5", "0"):
        with pytest.raises(ValueError):
            env.step(target)
        assert env.state() == before and env.events == []


def test_goal_success_and_post_terminal_guard():
    env = AbstractEnv(line_terrain())
    env.step("3")
    _, done = env.step("5")
    assert done and env.status == "success"
    assert env.distance_m == pytest.approx(3)
    assert env.time_s == pytest.approx(5)
    with pytest.raises(RuntimeError):
        env.step("3")


def test_episode_timeout_records_last_confirmed_support():
    env = AbstractEnv(line_terrain(), Limits(episode_budget_s=1))
    env.step("3")
    assert env.status == "timeout" and env.time_s == 1
    assert env.current_id == "0" and env.distance_m == 0
    assert env.events[-1]["target_pile_id"] == "3"


def test_subgoal_time_budget_affects_local_eligibility():
    env = AbstractEnv(line_terrain(), Limits(subgoal_budget_s=1))
    assert [p.id for p, valid in zip(env.observe().visible_piles, env.observe().action_mask) if valid] == ["1"]


def test_disconnected_terrain_has_explicit_failure():
    env = AbstractEnv(Terrain((Pile("start", 0, 0, 0, .5), Pile("goal", 4, 0, 0, .5)), "start", "goal"))
    assert OraclePlanner(env.geometry).route("start") == []
    result = run_episode(env, GreedyPlanner(), "greedy")
    assert result["metrics"]["status"] == "no_candidate"
    assert result["metrics"]["path_length_ratio"] is None


def test_world_to_robot_conversion_after_turn():
    terrain = Terrain((Pile("a", 0, 0, 0, .5), Pile("b", 0, .6, 0, .5),
                       Pile("c", 0, 1.2, 0, .5)), "a", "c")
    env = AbstractEnv(terrain)
    env.step("b")
    obs = env.observe()
    assert obs.yaw_rad == pytest.approx(math.pi/2)
    assert obs.goal_relative_xy_m == pytest.approx((.6, 0))
    visible = {p.id: p for p in obs.visible_piles}
    assert visible["c"].relative_top_center_m == pytest.approx((.6, 0, -.3))


@pytest.mark.parametrize("family", ["open", "moat", "dead_end"])
def test_comparisons_use_same_limits_and_valid_targets(family):
    result = run_comparison(generate_scene(family))
    oracle, greedy = result["episodes"]["oracle"], result["episodes"]["greedy"]
    assert oracle["metrics"]["success"]
    if family == "open":
        assert greedy["metrics"]["success"]
        assert greedy["metrics"]["path_length_ratio"] == pytest.approx(1)
    else:
        assert greedy["metrics"]["status"] == "timeout"
        assert greedy["metrics"]["loop_piles"] > 0
        assert greedy["metrics"]["stuck"]
        assert greedy["metrics"]["path_length_ratio"] is None
    for episode in (oracle, greedy):
        for event in episode["events"]:
            obs = event["observation"]
            valid = {p["id"] for p, valid in zip(obs["visible_piles"], obs["action_mask"]) if valid}
            assert event["target_pile_id"] in valid


def test_abstract_entrypoint_does_not_import_mujoco():
    # Run in a clean interpreter: existing physics tests import MuJoCo themselves.
    import subprocess
    subprocess.run([sys.executable, "-c", "import sys; import quadruped_nav.experiment; assert 'mujoco' not in sys.modules"], check=True)


def test_export_is_replayable_and_escapes_html(tmp_path):
    scene = generate_scene("open")
    scene["family"] = "</script><script>alert(1)</script>"
    result = run_comparison(scene)
    html, log = export_replay([result], tmp_path)
    bundle = json.loads(log.read_text())
    embedded = re.search(r'<script id="episode-data" type="application/json">(.*?)</script>', html.read_text(), re.S).group(1)
    assert json.loads(embedded) == bundle
    assert scene["family"] not in html.read_text()
    assert "__EPISODE_DATA__" not in html.read_text()


@pytest.mark.parametrize("limits", [{"max_gap_m": -1}, {"speed_m_s": 0}, {"episode_budget_s": float('nan')}])
def test_invalid_limits_rejected(limits):
    with pytest.raises(ValueError):
        Limits(**limits)


def test_adapter_keeps_scene_unchanged_and_uses_opaque_ids():
    scene = generate_scene()
    before = json.dumps(scene, sort_keys=True)
    terrain = Terrain.from_scene(scene)
    assert all(p.id.startswith("pile_") and "S00" not in p.id for p in terrain.piles)
    assert len(terrain.piles) == len(scene["stones"]) + len(scene["platforms"])
    assert json.dumps(scene, sort_keys=True) == before
    with pytest.raises(ValueError):
        Terrain.from_scene(scene, "missing")
