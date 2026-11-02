"""Physics inspection only: position servos hold Go1's standing pose, no learned gait."""
import mujoco
import numpy as np


class Simulation:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def step(self, steps=1):
        self.data.ctrl[:] = self.model.key_ctrl[0]
        mujoco.mj_step(self.model, self.data, nstep=steps)

    def status(self):
        pit = self.model.geom("pit_floor").id
        feet = {self.model.geom(n).id for n in ("FR", "FL", "RR", "RL")}
        foot_pit = any((int(c.geom1) in feet and c.geom2 == pit) or (int(c.geom2) in feet and c.geom1 == pit) for c in self.data.contact)
        return {"time": float(self.data.time), "base_height": float(self.data.qpos[2]),
                "finite": bool(np.isfinite(self.data.qpos).all()), "foot_in_pit": bool(foot_pit)}


def set_camera(camera, scene, mode="overview"):
    start = next(p for p in scene["platforms"] if p["id"] == "P0")
    far = max(p["x"] for p in scene["platforms"])
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if mode == "robot":
        camera.lookat[:] = [start["x"]+0.1, 0, 0.12]
        camera.distance, camera.azimuth, camera.elevation = 2.1, 140, -22
    else:
        camera.lookat[:] = [(start["x"]+far)/2, 0, -0.15]
        camera.distance = 13.4 if mode == "overview" else 13.0
        camera.azimuth, camera.elevation = (130,-48) if mode == "overview" else (90,-89.8)


def render_views(sim, scene, output):
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(sim.model, height=1000, width=1600) as renderer:
        for mode in ("overview", "robot", "top"):
            set_camera(camera, scene, mode)
            renderer.update_scene(sim.data, camera=camera, scene_option=option)
            image = Image.fromarray(renderer.render())
            image.save(output / f"{scene['family']}_{mode}.png")
    # A presentation sheet made from actual MuJoCo renders.
    sheet = Image.new("RGB", (1600,1190), "#0c1420")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=34)
    small = ImageFont.load_default(size=20)
    draw.text((40,28), "QUADRUPEDAL LOCOMOTION WITH LOCAL PERCEPTION", font=font, fill="#e5f0f7")
    draw.text((40,79), f"UNITREE Go1  /  {scene['family'].upper()} FIELD  /  LEVEL {scene['level']}  /  SEED {scene['seed']}", font=small, fill="#75d5c8")
    with Image.open(output/f"{scene['family']}_overview.png") as im:
        sheet.paste(im.resize((1040,650)), (20,132))
    with Image.open(output/f"{scene['family']}_robot.png") as im:
        sheet.paste(im.resize((520,325)), (1060,132))
    with Image.open(output/f"{scene['family']}_top.png") as im:
        sheet.paste(im.resize((520,325)), (1060,457))
    lines = ["SCENE FOUNDATION", "Deep physical gaps  |  Three goal platforms  |  Visual-only coloured landmarks", "P0: start   /   P1: red box   /   P2: blue cone   /   P3: yellow cone", "Moat crossing: follow the side bridge. The direct route has no supporting stones." if scene['family']=='moat' else "Deterministic scene JSON drives the geometry, platform IDs and landmarks.", "Go1 is standing under joint position control. Navigation and PPO are not trained yet."]
    for i,line in enumerate(lines):
        draw.text((40,825+i*55),line,font=small,fill="#75d5c8" if i==0 else "#cfdae6")
    sheet.save(output/f"{scene['family']}_presentation.png")
