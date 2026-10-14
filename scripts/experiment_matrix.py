"""Write auditable configs for the six equal-budget RQ1 runs; does not launch jobs."""
import json
from pathlib import Path
base=json.loads(Path('configs/cloud.json').read_text())
out=Path('configs/experiments');out.mkdir(exist_ok=True)
for curriculum in ['smooth','uniform']:
    for seed in [0,1,2]:
        cfg=json.loads(json.dumps(base));cfg['env'].update(curriculum=curriculum,seed=seed)
        cfg['output']=f'runs/{curriculum}-seed{seed}'
        (out/f'{curriculum}-seed{seed}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(out)
