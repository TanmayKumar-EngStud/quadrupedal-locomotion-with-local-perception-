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
        valid={p.id for p,ok in zip(observation.visible_piles,observation.action_mask) if ok}
        if len(route)>1 and route[1] in valid:return route[1]
        # A continuous physics pose may not be centred on its nearest pile.
        choices=[];x,y,_=observation.position_odom_m
        for candidate in valid:
            tail=self.route(candidate)
            if not tail:continue
            nodes=self.geometry.by_id
            cost=math.hypot(nodes[candidate].x-x,nodes[candidate].y-y)
            cost+=sum(math.hypot(nodes[b].x-nodes[a].x,nodes[b].y-nodes[a].y) for a,b in zip(tail,tail[1:]))
            choices.append((cost,candidate))
        return min(choices)[1] if choices else None


class OptimisticPlanner:
    """Local-only occupancy memory; unknown grid cells are provisionally walkable.

    Observed support is dilated by half the allowed gap. A conservative inner
    observation disc marks absent support, avoiding centre-filter boundary leaks.
    This resolution-dependent planner makes no completeness guarantee.
    """
    def __init__(self,resolution=.15,max_gap=.14,max_cells=120000):
        self.resolution=resolution;self.max_gap=max_gap;self.max_cells=max_cells;self.reset()

    def reset(self):self.memory={};self.seen_piles={}

    def cell(self,x,y):return round(x/self.resolution),round(y/self.resolution)

    def select_subgoal(self,observation):
        import numpy as np
        o=observation;c,s=math.cos(o.yaw_rad),math.sin(o.yaw_rad);x,y,z=o.position_odom_m
        visible=[]
        for p in o.visible_piles:
            dx,dy,dz=p.relative_top_center_m
            point=(x+c*dx-s*dy,y+s*dx+c*dy)
            visible.append((p.id,*point,p.top_size_xy_m[0]))
            self.seen_piles[p.id]=point
        # A 0.75 m margin covers half the diagonal of the largest (1 m) platform.
        radius=max(0,o.observation_radius_m-.75);r=self.resolution
        for ix in range(math.floor((x-radius)/r),math.ceil((x+radius)/r)+1):
            for iy in range(math.floor((y-radius)/r),math.ceil((y+radius)/r)+1):
                if math.hypot(ix*r-x,iy*r-y)<=radius:
                    self.memory[ix,iy]=any(abs(ix*r-px)<=size/2+self.max_gap/2 and abs(iy*r-py)<=size/2+self.max_gap/2 for _,px,py,size in visible)
        for _,px,py,size in visible:
            for ix in range(math.ceil((px-size/2)/r),math.floor((px+size/2)/r)+1):
                for iy in range(math.ceil((py-size/2)/r),math.floor((py+size/2)/r)+1):self.memory[ix,iy]=True
        gx,gy=o.goal_relative_xy_m;goal=self.cell(x+c*gx-s*gy,y+s*gx+c*gy)
        here=self.cell(x,y);self.memory[here]=True
        candidates=[(p,*self.seen_piles[p.id]) for p,valid in zip(o.visible_piles,o.action_mask) if valid]
        if not candidates:return None
        cells=[self.cell(px,py) for _,px,py in candidates]
        coords=list(self.memory)+[goal,here];pad=math.ceil(2/r)
        xmin=min(t[0] for t in coords)-pad;xmax=max(t[0] for t in coords)+pad
        ymin=min(t[1] for t in coords)-pad;ymax=max(t[1] for t in coords)+pad
        if (xmax-xmin+1)*(ymax-ymin+1)>self.max_cells:raise RuntimeError('Optimistic map exceeded configured search budget')
        queue=[(0.,goal)];distance={goal:0.};pending=set(cells)
        while queue and pending:
            cost,node=heapq.heappop(queue)
            if cost>distance[node]+1e-9:continue
            pending.discard(node)
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt=(node[0]+dx,node[1]+dy)
                if not xmin<=nxt[0]<=xmax or not ymin<=nxt[1]<=ymax or not self.memory.get(nxt,True):continue
                if dx and dy and (not self.memory.get((node[0]+dx,node[1]),True) or not self.memory.get((node[0],node[1]+dy),True)):continue
                value=cost+math.hypot(dx,dy)*r
                if value<distance.get(nxt,math.inf)-1e-9:
                    distance[nxt]=value;heapq.heappush(queue,(value,nxt))
        scored=[]
        for (p,px,py),cell in zip(candidates,cells):
            remaining=distance.get(cell,math.inf)
            scored.append((remaining+math.hypot(px-x,py-y),remaining,p.id))
        best=min(scored)
        return best[2] if math.isfinite(best[0]) else None
