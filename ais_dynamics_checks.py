"""Strength, stability, reversible localization, and paired-pulse controls."""
from pathlib import Path
import json,copy
import numpy as np
from ais_chain_experiment import Params, ChainAIS
from ais_gate_study import train_structure,common_world,test_world


def checks(p=Params(), train=1200,hold=500):
    w=ChainAIS(p,adaptive=True,phase_gate=False)
    w.run(250)
    w.run(train,cue_site=0,cue_amplitude=p.input_gain)
    after_left=w.gain.copy()
    w.run(hold)
    held=w.gain.copy()
    w.run(train,cue_site=p.pairs-1,cue_amplitude=p.input_gain)
    after_right=w.gain.copy()
    # No stimulus control: all FHN units should rest without artificial triggers.
    null=ChainAIS(p,adaptive=True,phase_gate=False)
    null.run(250+train+hold+train)
    # Sweep identical stimulus; all states except input scale fixed.
    virgin=common_world(p,train,hold)
    series=[]
    for amp in [0,.004,.008,.012,.018,.025,.035,.05,.075,.1]:
        result=test_world(virgin,0,amp,gain=held,phase_gate=False,steps=250)
        series.append({'amplitude':amp,'site0_spikes':result['spikes'][0],
                       'site1_spikes':result['spikes'][1],
                       'peak_site0':result['peak_u'][0],
                       'downstream_wave_rms':result['wave_rms']})
    pulses={}
    for gap in (20,50,120,250,450):
        z=copy.deepcopy(virgin);z.gain[:]=held;z.adaptive=False;z.phase_gate=False
        z.run(180,cue_site=0,cue_amplitude=.05)
        first=z.spikes.copy()
        z.run(gap)
        before_second=z.spikes.copy()
        reset_control=copy.deepcopy(z)
        # Restore only fast/recovery variables, not field, oscillator phase,
        # morphology, sensor, or simulation clock; isolate refractoriness.
        reset_control.u.fill(-1.2)
        reset_control.v.fill(-.62)
        z.run(180,cue_site=0,cue_amplitude=.05)
        reset_control.run(180,cue_site=0,cue_amplitude=.05)
        second=z.spikes-before_second
        restored=reset_control.spikes-before_second
        pulses[str(gap)]={'first':first.tolist(),'during_gap':(before_second-first).tolist(),
                          'second':second.tolist(),'reset_fast_recovery_second':restored.tolist(),
                          'events':z.strong_events}
    return {'scope':'added excitable FHN units with fixed phase gate OFF in this control; no structural movements or bonded mechanics',
            'settings':vars(p),'gain_left':after_left.tolist(), 'gain_left_after_hold':held.tolist(),
            'gain_after_reversed_training':after_right.tolist(),
            'null_events':null.strong_events,'strength_sweep':series,'paired_pulses':pulses}

if __name__=='__main__':
    data=checks()
    out=Path('results/ais/dynamics.json');out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(data,indent=2)+'\n')
    print('gain after L',np.round(data['gain_left'],3),'hold',np.round(data['gain_left_after_hold'],3),
          'retraining R',np.round(data['gain_after_reversed_training'],3))
    print('null events',len(data['null_events']))
    print('amplitude sweep')
    for row in data['strength_sweep']:print(row)
    print('paired pulse intervals')
    for gap,d in data['paired_pulses'].items():print(gap,d['first'],d['during_gap'],d['second'])
