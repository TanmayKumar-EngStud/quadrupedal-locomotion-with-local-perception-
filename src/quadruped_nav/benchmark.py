"""Paired, saved-layout experiments and uncertainty summaries."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from collections import Counter
from .terrain import generate_terrain
from .abstract import AbstractEnv,Terrain,Limits,Geometry
from .evaluate import make_planner,evaluate
from .experiment import run_episode
from .planners import OraclePlanner


def wilson(success,total,z=1.96):
    if total==0:return [None,None]
    p=success/total;den=1+z*z/total;middle=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [middle-half,middle+half]


def pocket_metrics(scene,points,success):
    """Geometric pocket occupancy, sampled along path segments at <=5 cm."""
    if scene['family']!='dead_end' or 'pocket_depth' not in scene['parameters']:
        return dict(pocket_entered=None,pocket_exited=None,pocket_escaped=None)
    opening=3-scene['parameters']['pocket_depth'];entered=exited=False
    for a,b in zip(points,points[1:]):
        count=max(1,math.ceil(math.hypot(b[0]-a[0],b[1]-a[1])/.05))
        for j in range(count+1):
            x=a[0]+(b[0]-a[0])*j/count;y=a[1]+(b[1]-a[1])*j/count
            inside=opening<x<3 and abs(y)<.75
            if inside:entered=True
            elif entered:exited=True
    return dict(pocket_entered=entered,pocket_exited=exited,pocket_escaped=entered and exited and success)


def create_manifest(path,count=10,split='test',row=0):
    offset={'train':0,'validation':100000,'test':200000}[split]
    if not 1<=count<10000:raise ValueError('Count must be in 1..9999')
    cases=[]
    for fidx,family in enumerate(['open','holes','moat','dead_end','steps','rough','slalom','bridge','slanted']):
        for i in range(count):
            seed=offset+fidx*10000+i;depth=[.5,1,2,4][i%4]
            scene=generate_terrain(family,row,seed,pocket_depth=depth)
            cases.append(dict(id=f'{split}-{family}-{i}',scene=scene,seed=seed,pocket_depth=depth))
    result=dict(format='navigation-benchmark/v1',split=split,row=row,cases=cases)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    return path


def benchmark(manifest,output,checkpoint=None,planner_checkpoint=None,planners=('oracle','greedy','optimistic'),calibration=None):
    manifest_path=Path(manifest);spec=json.loads(manifest_path.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];groups=defaultdict(list)
    for case in spec['cases']:
        scene=case['scene'];terrain=Terrain.from_scene(scene);limits=Limits()
        geo=Geometry(terrain,limits);route=OraclePlanner(geo).route(terrain.start_id)
        reference=sum(math.hypot(geo.by_id[a].x-geo.by_id[b].x,geo.by_id[a].y-geo.by_id[b].y) for a,b in zip(route,route[1:])) if route else None
        for name in planners:
            if checkpoint:
                result=evaluate(checkpoint,scene,name,60,case['seed'],planner_checkpoint)
                metrics={k:result[k] for k in ['status','success','time_s','distance_m']}
                targets=[d['target'] for d in result['decisions']];counts=Counter(targets)
                metrics.update(repeat_subgoal_fraction=(len(targets)-len(counts))/len(targets) if targets else 0.,
                               loop_piles=sum(v>=3 for v in counts.values()),decisions=len(targets),
                               path_length_ratio=metrics['distance_m']/reference if reference and metrics['success'] else None)
                points=[p['position'][:2] for p in result['trajectory']]
            else:
                planner=make_planner(name,terrain,limits,planner_checkpoint)
                env=AbstractEnv(terrain,limits)
                if calibration:
                    from .calibrate import CalibratedEnv
                    env=CalibratedEnv(env,calibration,scene['family'],scene['level'],case['seed'])
                result=run_episode(env,planner,name,reference);metrics=result['metrics']
                points=[(p['x'],p['y']) for p in result['states']]
            metrics.update(pocket_metrics(scene,points,metrics['success']))
            row=dict(case=case['id'],family=scene['family'],planner=name,pocket_depth=case['pocket_depth'],metrics=metrics)
            rows.append(row);groups[scene['family'],name].append(metrics)
            (output/f"{case['id']}-{name}.json").write_text(json.dumps(result,indent=2)+'\n')
            print(case['id'],name,metrics['status'],flush=True)
    summary=[]
    for (family,name),values in groups.items():
        successes=sum(x['success'] for x in values)
        summary.append(dict(family=family,planner=name,episodes=len(values),success_rate=successes/len(values),
                            wilson95=wilson(successes,len(values)),
                            fall_rate=sum(x['status']=='fall' for x in values)/len(values),
                            timeout_rate=sum(x['status']=='timeout' for x in values)/len(values),
                            pocket_entries=sum(bool(x['pocket_entered']) for x in values),
                            escape_rate=sum(bool(x['pocket_escaped']) for x in values)/sum(bool(x['pocket_entered']) for x in values) if any(x['pocket_entered'] for x in values) else None,
                            mean_success_distance_m=mean(x['distance_m'] for x in values if x['success']) if successes else None))
    report=dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),physics=bool(checkpoint),
                controller_checkpoint=str(checkpoint),results=rows,summary=summary)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return output/'summary.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='benchmarks/test.json');p.add_argument('--create',action='store_true')
    p.add_argument('--count',type=int,default=10);p.add_argument('--split',choices=['train','validation','test'],default='test');p.add_argument('--row',type=int,default=0)
    p.add_argument('--output',default='outputs/benchmarks');p.add_argument('--checkpoint');p.add_argument('--planner-checkpoint');p.add_argument('--calibration')
    p.add_argument('--planners',nargs='+',choices=['oracle','greedy','optimistic','learned'],default=['oracle','greedy','optimistic']);a=p.parse_args()
    if a.create:print(create_manifest(a.manifest,a.count,a.split,a.row));return
    print(benchmark(a.manifest,a.output,a.checkpoint,a.planner_checkpoint,a.planners,a.calibration))

if __name__=='__main__':main()
