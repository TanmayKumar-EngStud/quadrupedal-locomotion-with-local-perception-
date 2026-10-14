"""Compile the scene contract into portable MJCF with the Menagerie Go1 model."""
from pathlib import Path
import math
import os
import xml.etree.ElementTree as ET
from .terrain import surface_normal

ROOT = Path(__file__).resolve().parents[2]
COLOURS = {"red": "0.93 0.25 0.18 1", "blue": "0.08 0.48 0.95 1", "yellow": "1 0.70 0.10 1"}


def build_mjcf(scene, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = ET.parse(ROOT / "assets/unitree_go1/go1.xml").getroot()
    root.set("model", f"Go1 — {scene['family']} field")
    root.find("compiler").set("meshdir", os.path.relpath(ROOT / "assets/unitree_go1/assets", output.parent))
    option = root.find("option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000", azimuth="135", elevation="-40")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.35 0.38 0.44", diffuse="0.7 0.7 0.7", specular="0.25 0.25 0.25")
    ET.SubElement(visual, "rgba", haze="0.08 0.11 0.17 1")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.12 0.17 0.24", rgb2="0.035 0.05 0.075", width="512", height="3072")
    ET.SubElement(asset, "texture", name="pit_tex", type="2d", builtin="checker", rgb1="0.065 0.085 0.11", rgb2="0.08 0.105 0.13", width="512", height="512")
    ET.SubElement(asset, "material", name="pit_mat", texture="pit_tex", texrepeat="20 20", reflectance="0.03")
    # A cone mesh is convex and visual-only. MuJoCo has no primitive cone type.
    verts = [(0,0,0.36)] + [(0.12*math.cos(a*2*math.pi/24), 0.12*math.sin(a*2*math.pi/24), 0) for a in range(24)] + [(0,0,0)]
    faces = [(0,1+i,1+(i+1)%24) for i in range(24)] + [(25,1+(i+1)%24,1+i) for i in range(24)]
    ET.SubElement(asset, "mesh", name="landmark_cone", vertex=" ".join(str(v) for p in verts for v in p), face=" ".join(str(v) for f in faces for v in f))
    world = root.find("worldbody")
    ET.SubElement(world, "light", name="key", pos="1 -4 8", dir="0.3 0.3 -1", directional="true", diffuse="0.85 0.82 0.75", castshadow="true")
    ET.SubElement(world, "light", name="fill", pos="7 4 6", dir="-0.3 -0.3 -1", directional="true", diffuse="0.4 0.48 0.6", castshadow="false")
    depth = scene["parameters"]["pit_depth"]
    ET.SubElement(world, "geom", name="pit_floor", type="plane", pos=f"0 0 {-depth}", size="20 20 0.1", material="pit_mat", group="0")
    for node in scene["stones"] + scene["platforms"]:
        platform = node["id"].startswith("P")
        colour = "0.24 0.65 0.58 1" if node["id"] == "P0" else "0.28 0.45 0.56 1" if platform else "0.48 0.55 0.61 1"
        height = depth + node["z_top"]
        roll, pitch = node.get("roll",0.), node.get("pitch",0.)
        friction = f"{node.get('friction',.9)} 0.02 0.002"
        if roll or pitch:
            normal = surface_normal(roll,pitch)
            centre = [node['x']-.04*normal[0],node['y']-.04*normal[1],node['z_top']-.04*normal[2]]
            ET.SubElement(world,"geom",name=node['id'],type="box",pos=" ".join(map(str,centre)),
                          size=f"{node['size']/2} {node['size']/2} .04",euler=f"{roll} {pitch} 0",rgba=colour,friction=friction,group="0")
        else:
            ET.SubElement(world, "geom", name=node["id"], type="box", pos=f"{node['x']} {node['y']} {node['z_top']-height/2}", size=f"{node['size']/2} {node['size']/2} {height/2}", rgba=colour, friction=friction, group="0")
        if platform:
            # Thin non-colliding target ring, entirely inside the platform.
            ET.SubElement(world, "site", name=f"label_{node['id']}", type="sphere", pos=f"{node['x']} {node['y']} 0.012", size="0.025", rgba="0.7 1 0.93 1", group="4")
    for obstacle in scene.get("obstacles",[]):
        height = depth+obstacle['z_top']
        ET.SubElement(world,"geom",name=obstacle['id'],type="box",pos=f"{obstacle['x']} {obstacle['y']} {obstacle['z_top']-height/2}",size=f"{obstacle['size_x']/2} {obstacle['size_y']/2} {height/2}",rgba=".65 .32 .18 1",group="0")
    for landmark in scene["landmarks"]:
        x,y,z = landmark["x"],landmark["y"],landmark["z_base"]
        ET.SubElement(world, "geom", name=f"{landmark['id']}_plinth", type="cylinder", pos=f"{x} {y} {z-0.04}", size="0.18 0.04", rgba="0.17 0.23 0.29 1", contype="0", conaffinity="0", group="1")
        attr = dict(name=landmark["id"], rgba=COLOURS[landmark["colour"]], contype="0", conaffinity="0", group="1")
        if landmark["type"] == "cone":
            ET.SubElement(world,"geom",type="mesh",mesh="landmark_cone",pos=f"{x} {y} {z}",**attr)
        else:
            ET.SubElement(world,"geom",type="box",size="0.13 0.13 0.16",pos=f"{x} {y} {z+0.16}",**attr)
    start = next(p for p in scene["platforms"] if p["id"] == scene["start"]["platform"])
    angle = scene["start"]["heading"]
    key = root.find("keyframe/key")
    key.set("qpos", f"{start['x']} {start['y']} {start['z_top']+0.30} {math.cos(angle/2)} 0 0 {math.sin(angle/2)} " + "0 0.9 -1.8 "*4)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)
    return output
