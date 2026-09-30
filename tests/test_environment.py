import math
import numpy as np
import pytest
import mujoco

from quadruped_nav.scene import generate_scene, difficulty, geometric_route, save_scene, load_scene
from quadruped_nav.mjcf import build_mjcf
from quadruped_nav.simulation import Simulation


@pytest.mark.parametrize("family", ["open", "moat", "dead_end"])
@pytest.mark.parametrize("level", [0, 5, 9])
def test_scene_has_routes_and_real_gaps(family, level, tmp_path):
    scene = generate_scene(family, level)
    assert len(scene["platforms"]) == 4
    assert difficulty(level) == pytest.approx({0:(.46,.04),5:(.26,.09),9:(.10,.13)}[level])
    assert len({n["id"] for n in scene["stones"]+scene["platforms"]}) == len(scene["stones"])+4
    for goal in ("P1", "P2", "P3"):
        route = geometric_route(scene, goal)
        assert route and route[0]["id"] == "P0" and route[-1]["id"] == goal
        if family != "open" and goal == "P2":
            assert max(n["y"] for n in route) >= 2.0
            length = sum(math.hypot(a["x"]-b["x"], a["y"]-b["y"]) for a,b in zip(route,route[1:]))
            assert length > route[-1]["x"]-route[0]["x"]+1.0
    model = mujoco.MjModel.from_xml_path(str(build_mjcf(scene, tmp_path/f"{family}.xml")))
    assert model.nu == 12 and model.nq == 19
    assert model.geom("pit_floor").pos[2] == -1.2
    for landmark in scene["landmarks"]:
        geom = model.geom(landmark["id"])
        assert geom.contype == 0 and geom.conaffinity == 0
    for stone in scene["stones"]:
        geom = model.geom(stone["id"])
        assert geom.pos[2]+geom.size[2] == pytest.approx(stone["z_top"])
    # All support surfaces are disjoint (no hidden slabs filling the pits).
    nodes = scene["stones"]+scene["platforms"]
    xy = np.array([[n["x"],n["y"]] for n in nodes])
    sizes = np.array([n["size"] for n in nodes])
    separation = np.abs(xy[:,None]-xy[None,:]) - (sizes[:,None,None]+sizes[None,:,None])/2
    overlap = np.all(separation < -1e-6, axis=-1)
    np.fill_diagonal(overlap,False)
    assert not overlap.any()


def test_deterministic_json_and_random_removal(tmp_path):
    a = generate_scene("open",5,42,.2)
    b = generate_scene("open",5,42,.2)
    c = generate_scene("open",5,43,.2)
    assert a == b and a["stones"] != c["stones"]
    save_scene(a,tmp_path/"scene.json")
    assert load_scene(tmp_path/"scene.json") == a


def test_go1_stands_and_falls_in_moat(tmp_path):
    scene = generate_scene("moat")
    sim = Simulation(build_mjcf(scene,tmp_path/"scene.xml"))
    sim.step(2500)
    assert sim.status()["finite"]
    assert .2 < sim.data.qpos[2] < .4
    assert abs(sim.data.qpos[0]-sim.model.key_qpos[0,0]) < .15
    assert not sim.status()["foot_in_pit"]
    assert sim.data.ncon >= 4
    sim.reset()
    # Default moat is centred at x=3.5 and x=4.0, spanning 3.27..4.23 m.
    sim.data.qpos[0:3] = [3.75,0,.3]
    mujoco.mj_forward(sim.model,sim.data)
    sim.step(1500)
    assert sim.data.qpos[2] < -.6
    assert sim.status()["foot_in_pit"]


@pytest.mark.parametrize("level", [-1,10])
def test_invalid_difficulty(level):
    with pytest.raises(ValueError): generate_scene(level=level)
