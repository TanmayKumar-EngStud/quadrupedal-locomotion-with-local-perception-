"""Classical baselines with deliberately different information access."""
from __future__ import annotations

import heapq
import math
from typing import Protocol

from .abstract import Geometry, Observation


class Planner(Protocol):
    def select_subgoal(self, observation: Observation) -> str | None: ...


class GreedyPlanner:
    """Memoryless and local: never receives the environment or full terrain."""
    def select_subgoal(self, observation: Observation) -> str | None:
        gx, gy = observation.goal_relative_xy_m
        candidates = [(math.hypot(p.relative_top_center_m[0]-gx, p.relative_top_center_m[1]-gy), p.id)
                      for p, allowed in zip(observation.visible_piles, observation.action_mask) if allowed]
        return min(candidates)[1] if candidates else None


class OraclePlanner:
    """Privileged full-map A* over the SAME admissible subgoal transitions.

    Uses distance costs and a Euclidean lower bound. This is a reference for the
    provisional abstract model, not a guarantee of the robot's physical ability.
    """
    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    def route(self, start_id: str) -> list[str]:
        geometry = self.geometry
        goal_id = geometry.terrain.goal_id
        goal = geometry.by_id[goal_id]
        def h(p):
            return math.hypot(p.x-goal.x, p.y-goal.y)
        queue = [(h(geometry.by_id[start_id]), 0.0, start_id)]
        costs, previous = {start_id: 0.0}, {}
        while queue:
            _, cost, current_id = heapq.heappop(queue)
            if cost > costs[current_id] + 1e-9:
                continue
            if current_id == goal_id:
                path = [current_id]
                while current_id in previous:
                    current_id = previous[current_id]
                    path.append(current_id)
                return path[::-1]
            current = geometry.by_id[current_id]
            for target in geometry.neighbours(current_id):
                new_cost = cost + math.hypot(target.x-current.x, target.y-current.y)
                if new_cost < costs.get(target.id, math.inf) - 1e-9:
                    costs[target.id], previous[target.id] = new_cost, current_id
                    heapq.heappush(queue, (new_cost+h(target), new_cost, target.id))
        return []

    def select_subgoal(self, observation: Observation) -> str | None:
        route = self.route(observation.current_pile_id)
        return route[1] if len(route) > 1 else None
