"""Predeclared causal isolation tests for the engineered excitable chain.

The structural test grafts trained site gains into a COMMON virgin world with
identical field, phase, fast excitability and simulation time. The phase test
grafts history-dependent oscillator phases into another common world with
identical gains and all other variables. This separates an operator change from
phase storage, and forbids reading a residual field echo as retention.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import argparse
import copy
import json
import math
import numpy as np
from ais_chain_experiment import Params, ChainAIS, region_experiment


def train_structure(p, site, train=1300, hold=350):
    w=ChainAIS(p,adaptive=True)
    w.run(250)
    w.run(train,cue_site=site,cue_amplitude=p.input_gain)
    x=w.gain.copy();attrain=w.snapshot()
    w.run(hold)
    return w.gain.copy(),attrain,w.snapshot()


def train_phase(p, phase, site=0, train=1300, hold=350):
    w=ChainAIS(p,adaptive=False,excitable=False)
    w.run(250)
    w.run(train,cue_site=site,cue_amplitude=p.input_gain,cue_phase=phase)
    d_write=w.phase_memory().copy()
    w.run(hold)
    return w.phase.copy(),d_write,w.phase_memory().copy()


def common_world(p, train, hold):
    w=ChainAIS(p,adaptive=False)
    w.run(250+train+hold)
    w.matched_fast_reset()
    return w


def test_world(template, site, amplitude, gain=None, phase=None,
               phase_gate=True, excitable=True, steps=300):
    w=copy.deepcopy(template)
    w.phase_gate=phase_gate
    w.excitable=excitable
    w.adaptive=False
    if gain is not None:w.gain[:]=gain
    if phase is not None:w.phase[:]=phase
    w.run(steps,cue_site=site,cue_amplitude=amplitude,record=True)
    return {'spikes':w.spikes.tolist(), 'total_spikes':int(w.total_spikes),
            'events':w.strong_events,'peak_u':np.max(np.array(w.spike_records)[:,1:1+w.p.pairs],axis=0).tolist(),
            'end_D':w.phase_memory().tolist(),
            'downstream_peak_field':float(np.max(np.abs(w.field_records))),
            'wave_rms':float(np.sqrt(np.mean(np.array(w.field_records)**2)))}


def run(p,train=1300,hold=350,test=320,probe=None):
    gain_l,pre_l,held_l=train_structure(p,0,train,hold)
    gain_r,pre_r,held_r=train_structure(p,p.pairs-1,train,hold)
    ph0,dw0,dh0=train_phase(p,0.,train=train,hold=hold)
    ph1,dw1,dh1=train_phase(p,math.pi,train=train,hold=hold)
    virgin=common_world(p,train,hold)
    if probe is None:probe=p.input_gain*.35
    structural={}
    for site in [0,p.pairs-1]:
        for name,g in [('trained_left',gain_l),('trained_right',gain_r),('untrained',np.ones(p.pairs))]:
            structural[f'probe{site}/{name}']=test_world(virgin,site,probe,gain=g,phase_gate=False,steps=test)
    phase={}
    for sig,name in [(ph0,'history_0'),(ph1,'history_pi'),(virgin.phase,'erased_phase')]:
        phase[name]=test_world(virgin,0,probe,gain=gain_l,phase=sig,steps=test)
        phase[name+'_gate_ablated']=test_world(virgin,0,probe,gain=gain_l,phase=sig,phase_gate=False,steps=test)
    return {'scope':'KG + original phase update; added external-energy sensor, FHN excitability, phase-gated gain and local homeostasis. Pinned UNBONDED oscillator sequence. No inter-chain spike regeneration.','settings':vars(p), 'timing':{'train':train,'hold':hold,'test':test,'probe':probe},
            'trained_gains':{'left':gain_l.tolist(),'right':gain_r.tolist()},'structural_phase_gate':'disabled to isolate structural gain from accidental initial D',
            'trained_phase_difference':{'history_0_after_write':dw0.tolist(),'history_pi_after_write':dw1.tolist(),
                                          'history_0_after_hold':dh0.tolist(),'history_pi_after_hold':dh1.tolist(),
                                          'separation_after_hold':np.abs(np.angle(np.exp(1j*(dh1-dh0)))).tolist()},
            'training_spike_counts':{'left':pre_l['spikes'].tolist(),'right':pre_r['spikes'].tolist()},
            'held_spike_counts':{'left':held_l['spikes'].tolist(),'right':held_r['spikes'].tolist()},
            'structural':structural,'phase':phase}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/ais/gates.json')
    ap.add_argument('--quick',action='store_true');ap.add_argument('--probe',type=float,default=None)
    a=ap.parse_args(); p=Params()
    d=run(p,train=480 if a.quick else 1300,hold=180 if a.quick else 350,
          test=220 if a.quick else 320,probe=a.probe)
    dest=Path(a.out);dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(json.dumps(d,indent=2)+'\n')
    print('GAIN L/R',np.round(d['trained_gains']['left'],3),np.round(d['trained_gains']['right'],3))
    print('PHASE SEPARATION',np.round(d['trained_phase_difference']['separation_after_hold'],3))
    for kind in ['structural','phase']:
        for k,v in d[kind].items():
            print(kind,k,'spikes',v['spikes'],'peak',np.round(v['peak_u'],2),'RMS',round(v['wave_rms'],5))
