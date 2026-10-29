"""Control: does signal-to-signal transfer show a spike-like threshold?"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from receiver_regeneration import Settings, TwoPairWorld, circ_sep


def run(gains=(0.,.025,.05,.1,.2,.4,.8,1.), out='results/regeneration/gain_sweep.json'):
    cfg=Settings(phase_noise=0.)
    rows=[]
    for gain in gains:
        runs=[]
        for phase in (0.,np.pi):
            w=TwoPairWorld(cfg,active=True)
            w.advance(cfg.write_steps-cfg.write_record,carrier_phase=phase,sender_gain=gain)
            w.advance(cfg.write_record,carrier_phase=phase,sender_gain=gain,record=True)
            runs.append((w.phase_difference(),np.asarray(w.receiver_history)))
        direct=np.sqrt(np.mean((runs[0][1][:,0]-runs[1][1][:,0])**2))
        relayed=np.sqrt(np.mean((runs[0][1][:,1]-runs[1][1][:,1])**2))
        rows.append({'gain':float(gain),'sender_bit_rms_far':float(direct),
                     'receiver_bit_rms_far':float(relayed),
                     'relative_phase_separation':circ_sep(runs[0][0],runs[1][0])})
        print('gain=',gain,' direct=',direct,' receiver=',relayed,flush=True)
    result={'scope':'stationary two-pair KG+phase law, no imposed threshold/refractory; one initial phase configuration',
            'rows':rows}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    run()
