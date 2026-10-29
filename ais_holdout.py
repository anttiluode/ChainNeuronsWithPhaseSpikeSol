"""Small predeclared holdout; no re-tuning thresholds across seeds.

PASS requires 2/3 site-selective trained-vs-untrained effects plus a
history-dependent difference abolished by removal of the phase gate.
We report each distinct criterion separately and never substitute mean gain.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse,json,numpy as np
from ais_gate_study import run
from ais_chain_experiment import Params

def score(d):
    s=d['structural'];h=d['phase']; L=len(d['trained_gains']['left'])-1
    def count(key, idx):return int(s[key]['spikes'][idx])
    select_left=(count('probe0/trained_left',0)>count('probe0/trained_right',0)
                 and count('probe0/trained_left',0)>count('probe0/untrained',0))
    select_right=(count(f'probe{L}/trained_right',L)>count(f'probe{L}/trained_left',L)
                  and count(f'probe{L}/trained_right',L)>count(f'probe{L}/untrained',L))
    phase_counts=(h['history_0']['spikes'],h['history_pi']['spikes'])
    ablated=(h['history_0_gate_ablated']['spikes'],h['history_pi_gate_ablated']['spikes'])
    return {'left_selective':select_left,'right_selective':select_right,
            'phase_history_changes_spike_pattern':phase_counts[0]!=phase_counts[1],
            'gate_ablation_makes_history_invariant':ablated[0]==ablated[1],
            'phase_spikes':phase_counts,'ablated_spikes':ablated}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--start',type=int,default=101)
    ap.add_argument('--noise',type=float,default=.003)
    ap.add_argument('--out',default='results/ais/holdout.json')
    a=ap.parse_args()
    arr=[]
    for seed in range(a.start,a.start+a.seeds):
        p=Params(seed=seed,noise=a.noise,phase_init_mode='random')
        d=run(p,train=480,hold=180,test=220)
        v=score(d);v['seed']=seed
        v['phase_sep']=d['trained_phase_difference']['separation_after_hold']
        arr.append(v)
        print(seed,v,flush=True)
    sums={key:sum(bool(v[key]) for v in arr) for key in ['left_selective','right_selective','phase_history_changes_spike_pattern','gate_ablation_makes_history_invariant']}
    report={'scope':'holdout independent phase preparations; no per-seed tuning','noise':a.noise,'seeds':arr,'counts':sums}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(sums)
