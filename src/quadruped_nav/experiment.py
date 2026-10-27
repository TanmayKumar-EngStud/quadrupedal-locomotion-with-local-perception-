"""Run paired abstract episodes and export a self-contained, offline replay."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import webbrowser

from .abstract import AbstractEnv, Limits, Terrain
from .planners import GreedyPlanner, OraclePlanner, OptimisticPlanner, Planner
from .scene import FAMILIES, generate_scene, load_scene


def run_episode(env: AbstractEnv, planner: Planner, name: str, oracle_distance_m: float | None = None):
    if hasattr(planner,'reset'):planner.reset()
    observation = env.reset()
    while env.status == "running":
        observation, _ = env.step(planner.select_subgoal(observation))
    selections = [event["target_pile_id"] for event in env.events if event["target_pile_id"] is not None]
    counts = Counter(selections)
    repeat_count = len(selections) - len(counts)
    loops = sum(count >= 3 for count in counts.values())
    return {
        "planner": name, "states": env.states, "events": env.events,
        "metrics": {
            "status": env.status, "success": env.status == "success",
            "simulated_time_s": env.time_s, "distance_m": env.distance_m,
            "decisions": env.decisions, "repeat_subgoal_fraction": repeat_count / len(selections) if selections else 0.0,
            "loop_piles": loops, "stuck": env.status == "timeout" and loops > 0,
            "oracle_distance_m": oracle_distance_m,
            "path_length_ratio": env.distance_m / oracle_distance_m
            if env.status == "success" and oracle_distance_m and oracle_distance_m > 0 else None,
        },
    }


def run_comparison(scene: dict, limits: Limits | None = None, target: str = "P2"):
    limits = limits or Limits()
    terrain = Terrain.from_scene(scene, target)
    oracle_env = AbstractEnv(terrain, limits)
    oracle = OraclePlanner(oracle_env.geometry)
    route = oracle.route(terrain.start_id)
    nodes = oracle_env.geometry.by_id
    reference_distance = sum(math.hypot(nodes[b].x-nodes[a].x, nodes[b].y-nodes[a].y)
                             for a, b in zip(route, route[1:])) if route else None
    oracle_result = run_episode(oracle_env, oracle, "oracle", reference_distance)
    greedy_result = run_episode(AbstractEnv(terrain, limits), GreedyPlanner(), "greedy", reference_distance)
    optimistic_result = run_episode(AbstractEnv(terrain, limits), OptimisticPlanner(max_gap=limits.max_gap_m), "optimistic", reference_distance)
    return {
        "family": scene.get("family", "custom"), "level": scene.get("level"), "seed": scene.get("seed"),
        "target_source_id": target,
        "terrain": asdict(terrain), "limits": asdict(limits),
        "oracle_reference": [asdict(nodes[node]) for node in route],
        "episodes": {"greedy": greedy_result, "oracle": oracle_result, "optimistic":optimistic_result},
    }


def export_replay(comparisons: list[dict], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": "abstract-replay/v1",
        "model": "Uncalibrated deterministic point-body model; horizontal square piles; no Go1 physics.",
        "assumptions": [
            "Perfect odometry and centre-based 2 m visibility without occlusion.",
            "Straight-line support check; configurable gap and height limits; no body or foot clearance model.",
            "Targets closer than 1 m are permitted, including recovery; turning is instantaneous.",
            "Constant speed; subgoals are atomic; unfinished timeout transitions stay at last confirmed support.",
            "Full terrain and oracle are exported for observer inspection, never passed to the greedy planner.",
        ],
        "comparisons": comparisons,
    }
    payload = json.dumps(bundle, allow_nan=False)
    json_path = output / "episodes.json"
    json_path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    # Escape HTML metacharacters so even externally supplied IDs cannot close the data element.
    safe_payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("replay.html").read_text()
    html_path = output / "replay.html"
    html_path.write_text(template.replace("__EPISODE_DATA__", safe_payload))
    return html_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Run oracle/greedy abstract navigation and open the replay.")
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--removal", type=float, default=0.0)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--target", default="P2")
    parser.add_argument("--max-gap", type=float, default=.14, help="Provisional centreline gap limit, metres")
    parser.add_argument("--episode-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("outputs/abstract"))
    parser.add_argument("--open", action="store_true", help="Open the generated replay in the default browser")
    args = parser.parse_args()
    limits = Limits(max_gap_m=args.max_gap, episode_budget_s=args.episode_seconds)
    if args.scene:
        scenes = [load_scene(args.scene)]
    else:
        families = FAMILIES if args.family == "all" else (args.family,)
        if args.removal and families != ("open",):
            parser.error("--removal requires --family open (without --scene)")
        scenes = [generate_scene(family, args.level, args.seed, args.removal) for family in families]
    comparisons = []
    for scene in scenes:
        print(f"Running {scene['family']} …", flush=True)
        comparison = run_comparison(scene, limits, args.target)
        comparisons.append(comparison)
        for name, episode in comparison["episodes"].items():
            metrics = episode["metrics"]
            print(f"  {name}: {metrics['status']}, {metrics['decisions']} decisions, "
                  f"{metrics['distance_m']:.2f} m, {metrics['simulated_time_s']:.2f} simulated seconds", flush=True)
    html, log = export_replay(comparisons, args.output.resolve())
    print(f"Replay: {html}\nEpisode log: {log}", flush=True)
    if args.open:
        webbrowser.open(html.as_uri())


if __name__ == "__main__":
    main()
