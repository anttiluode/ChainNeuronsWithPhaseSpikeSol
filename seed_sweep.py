"""Paired-seed robustness audit for receiver_regeneration.py."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from receiver_regeneration import Settings, run_assay


def run(seeds=range(8), noise=.003, out='results/regeneration/seed_sweep.json'):
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(10000+seed)
        initial=rng.uniform(-np.pi,np.pi,size=2)
        cfg=Settings(seed=int(seed),phase_noise=noise)
        res=run_assay(cfg,receiver_init=initial)
        row={'seed':int(seed),'initial_receiver_phase':initial.tolist()}
        for arm in ('active','passive','ablated'):
            for key,val in res[arm].items():
                if isinstance(val,(int,float)):
                    row[f'{arm}/{key}']=val
        rows.append(row)
        print(f"seed={seed} write_gain={row['active/write_receiver_to_direct_contrast_ratio']:.3f} "
              f"retained={row['active/retained_phase_separation']:.3f} "
              f"susceptibility_delta={row['active/second_probe_conditional_response_rms_difference']:.6f}",flush=True)
    keys=['active/write_receiver_to_direct_contrast_ratio',
          'active/retained_phase_separation',
          'active/late_receiver_trace_rms_difference',
          'active/second_probe_conditional_response_rms_difference']
    stats={k:{'min':float(np.min([r[k] for r in rows])),
              'median':float(np.median([r[k] for r in rows])),
              'max':float(np.max([r[k] for r in rows]))} for k in keys}
    results={'n_seeds':len(rows),'noise':noise,'notes':'Shared random streams in paired arms; random receiver initial phases, otherwise identical protocol.',
             'counts':{'active_write_gain_gt_one':sum(r['active/write_receiver_to_direct_contrast_ratio']>1 for r in rows),
                       'active_memory_phase_above_0p1_rad':sum(r['active/retained_phase_separation']>.1 for r in rows)},
             'stats':stats,'rows':rows}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,default=8)
    ap.add_argument('--noise',type=float,default=.003)
    ap.add_argument('--out',default='results/regeneration/seed_sweep.json')
    args=ap.parse_args()
    print(json.dumps(run(range(args.seeds),args.noise,args.out)['stats'],indent=2))
