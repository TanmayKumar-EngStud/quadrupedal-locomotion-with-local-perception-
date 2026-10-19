"""Go1 model construction and the physics settings shared by training and evaluation.

configure_physics() is applied to every compiled model, whether it is the
training heightfield world or a project scene built from boxes, so a policy
always sees the same robot: same gains, solver, and contact set.
"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GO1_XML = ROOT / 'assets/unitree_go1/go1.xml'
FEET = ('FR', 'FL', 'RR', 'RL')            # also the joint/leg order of the 12 actuators
CALVES = tuple(f'{leg}_calf' for leg in FEET)


def build_training_model(grid) -> mujoco.MjModel:
    """Go1 on the training terrain of a TerrainGrid: one heightfield per tile plus flat borders."""
    root = ET.parse(GO1_XML).getroot()
    root.find('compiler').set('meshdir', str(ROOT / 'assets/unitree_go1/assets'))
    asset, world = root.find('asset'), root.find('worldbody')
    ET.SubElement(asset, 'texture', name='grid_tex', type='2d', builtin='checker', rgb1='.55 .58 .62',
                  rgb2='.45 .48 .52', width='512', height='512', mark='edge', markrgb='.3 .3 .3')
    ET.SubElement(asset, 'material', name='grid_mat', texture='grid_tex', texrepeat='16 16', texuniform='false')
    ET.SubElement(world, 'light', name='sun', pos='0 0 10', dir='.3 .3 -1', directional='true', diffuse='.8 .8 .8')
    tiles = list(grid.tiles())
    for name, x, y, heights in tiles:
        low, span = float(heights.min()), max(float(heights.max() - heights.min()), 1e-3)
        ET.SubElement(asset, 'hfield', name=name, nrow=str(heights.shape[0]), ncol=str(heights.shape[1]),
                      size=f'{grid.tile_size / 2} {grid.tile_size / 2} {span:.6f} .5')
        ET.SubElement(world, 'geom', name=name, type='hfield', hfield=name, material='grid_mat',
                      pos=f'{x:.6f} {y:.6f} {low:.6f}', friction='1 .005 .0001')
    for i, (x, y, hx, hy) in enumerate(grid.border_boxes()):
        ET.SubElement(world, 'geom', name=f'border{i}', type='box', size=f'{hx} {hy} .5', pos=f'{x} {y} -.5',
                      material='grid_mat', friction='1 .005 .0001')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    for index, (_, _, _, heights) in enumerate(tiles):
        low, span = float(heights.min()), max(float(heights.max() - heights.min()), 1e-3)
        start = model.hfield_adr[index]
        model.hfield_data[start:start + heights.size] = ((heights - low) / span).ravel()
    return model


def colliding_robot_geoms(model, contact_set='feet_trunk'):
    """Robot geoms that may touch terrain.

    'feet_trunk': feet plus the trunk box and head plate (fast; legs are kept off
    the terrain by the knee-clearance penalty). 'legs': also shins and thighs.
    """
    ids = []
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if model.geom_bodyid[g] == 0 or model.geom_group[g] != 3:
            continue
        if model.geom(g).name in FEET:
            ids.append(g)
        elif body == 'trunk' and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            ids.append(g)   # main trunk box and the thin head plate
        elif contact_set == 'legs' and body.endswith(('_calf', '_thigh')):
            ids.append(g)
    return ids


def configure_physics(model, cfg):
    """Apply solver, actuator and contact settings from an EnvConfig in place."""
    model.opt.timestep = cfg.sim_dt
    model.opt.iterations = cfg.solver_iterations
    model.opt.ls_iterations = cfg.solver_ls_iterations
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.impratio = 1.
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    # MuJoCo Warp convex collisions require zero margins; use them everywhere.
    model.geom_margin[:] = 0
    model.pair_margin[:] = 0
    # PD joint control: position actuators with stiffness kp and joint damping.
    model.actuator_gainprm[:, 0] = cfg.kp
    model.actuator_biasprm[:, 1] = -cfg.kp
    model.actuator_biasprm[:, 2] = 0.
    model.dof_damping[6:] = cfg.joint_damping
    # Contacts: robot geoms touch the world only (no self-collision), and only
    # the geoms that can realistically hit the ground take part.
    robot = set(colliding_robot_geoms(model, cfg.contact_set))
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            model.geom_contype[g], model.geom_conaffinity[g] = 0, 1
        elif g in robot:
            model.geom_contype[g], model.geom_conaffinity[g] = 1, 0
        else:
            model.geom_contype[g], model.geom_conaffinity[g] = 0, 0
    for name in FEET:
        g = model.geom(name).id
        model.geom_condim[g] = 3
        model.geom_friction[g] = (1., .005, .0001)
    return model


def home_pose(model):
    """Standing configuration from the model keyframe (12 joint angles)."""
    return np.asarray(model.key_qpos[0], dtype=np.float32)
