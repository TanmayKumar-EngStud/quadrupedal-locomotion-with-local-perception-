"""Report mean/std across independently trained seeds, separate from Wilson intervals."""
import argparse,json,statistics
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('summaries',nargs='+');a=p.parse_args();groups=defaultdict(list)
for path in a.summaries:
    report=json.load(open(path))
    for row in report['summary']:groups[row['family'],row['planner']].append(row['success_rate'])
print(json.dumps([dict(family=k[0],planner=k[1],runs=len(v),mean=statistics.mean(v),std=statistics.stdev(v) if len(v)>1 else None) for k,v in groups.items()],indent=2))
