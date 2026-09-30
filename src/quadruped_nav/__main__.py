import argparse
import json
from pathlib import Path
import time

from .scene import generate_scene, save_scene, load_scene, geometric_route, FAMILIES
from .mjcf import build_mjcf, ROOT
from .simulation import Simulation, render_views, set_camera


def main():
    parser = argparse.ArgumentParser(description="Build and inspect a MuJoCo Go1 navigation scene.")
    parser.add_argument("command", choices=("build", "check", "render", "view"))
    parser.add_argument("--family", choices=FAMILIES, default="moat")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--removal", type=float, default=0.0)
    parser.add_argument("--scene", type=Path, help="Use an existing JSON layout instead of generating one")
    parser.add_argument("--output", type=Path, default=ROOT/"scenes")
    args = parser.parse_args()
    scene = load_scene(args.scene) if args.scene else generate_scene(args.family,args.level,args.seed,args.removal)
    stem = f"{scene['family']}_level{scene['level']}_seed{scene['seed']}"
    save_scene(scene,args.output/f"{stem}.json")
    path = build_mjcf(scene,args.output/f"{stem}.xml")
    print(f"Scene: {path}",flush=True)
    if args.command == "build":
        return
    sim = Simulation(path)
    if args.command == "check":
        sim.step(2500)
        result = sim.status()
        result["routes"] = {p["id"]: len(geometric_route(scene,p["id"])) for p in scene["platforms"] if p["role"] == "goal"}
        print(json.dumps(result,indent=2))
        if not result["finite"] or result["base_height"] < 0.15 or not all(result["routes"].values()):
            raise SystemExit("Scene verification failed")
    elif args.command == "render":
        sim.step(1000)
        render_views(sim,scene,ROOT/"outputs")
        print(f"Rendered views in {ROOT/'outputs'}")
    else:
        import mujoco.viewer
        state = {"paused":False,"camera":"overview","reset":False}
        def key(code):
            if code == 32: state["paused"] = not state["paused"]
            elif code in (49,50,51): state["camera"] = {49:"overview",50:"robot",51:"top"}[code]
            elif code in (82,114): state["reset"] = True
        print("1 overview | 2 Go1 close-up | 3 top | Space pause | R reset. Mouse: orbit, pan, zoom.",flush=True)
        with mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key,show_left_ui=False,show_right_ui=False) as viewer:
            last_camera = None
            viewer.opt.geomgroup[3] = 0
            while viewer.is_running():
                t = time.monotonic()
                with viewer.lock():
                    if state["reset"]:
                        sim.reset(); state["reset"] = False
                    if state["camera"] != last_camera:
                        set_camera(viewer.cam,scene,state["camera"])
                        last_camera = state["camera"]
                    if not state["paused"]: sim.step(10)
                viewer.sync()
                time.sleep(max(0,0.02-(time.monotonic()-t)))


if __name__ == "__main__":
    main()
