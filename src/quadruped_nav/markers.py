"""Viewer markers for the terrain samples the controller senses (watch.py and play.py).

Every sample of the policy's terrain scan is drawn as a small sphere at its world
x, y and at the height the policy senses there. On the pile scenes with box
ground the sensed pits are 0.5 m deep (HeightGrid.from_scene), so samples over a
pit float inside the 1.2 m pit: that is what the controller "sees".
"""
from __future__ import annotations
import mujoco
import numpy as np

RADIUS = .012
BODY_COLOUR = (1., 1., 1., .9)                                      # samples anchored at the base
FOOT_COLOURS = ((.95, .25, .2, .9), (.2, .8, .3, .9), (.25, .45, .95, .9), (.95, .8, .15, .9))   # FR, FL, RR, RL
FOOT_NAMES = ('FR red', 'FL green', 'RR blue', 'RL yellow')


def describe_scan(env):
    """One line naming the scan layout, its sample counts and the marker colours."""
    body, feet = len(env.body_scan), env.foot_scan.shape[1]
    return (f'terrain scan "{env.cfg.scan_layout}": {body} body samples (white) + 4 x {feet} foot samples '
            f'({", ".join(FOOT_NAMES)})')


def draw_scan(scene, env, robot=0):
    """Append one sphere per terrain sample of the robot's observation to the viewer's user scene."""
    body, feet = env.scan_points()
    groups = [(body[robot], BODY_COLOUR)] + [(feet[robot, i], colour) for i, colour in enumerate(FOOT_COLOURS)]
    size, frame = np.array([RADIUS, 0., 0.]), np.eye(3).ravel()
    for points, colour in groups:
        rgba = np.array(colour, dtype=np.float32)
        for x, y, z in points.tolist():
            if scene.ngeom >= scene.maxgeom: return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, size,
                                np.array([x, y, z + RADIUS]), frame, rgba)
            scene.ngeom += 1
