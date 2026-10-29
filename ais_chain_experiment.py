"""A deliberately engineered AIS-like chain in the original 2-D pumped KG field.

Four-to-six *pinned* A/T oscillator pairs form a synthetic chain. Their freely
rotating phase difference is the memory coordinate; a spatially concentrated
activity-gated FitzHugh--Nagumo process is a NEW excitable mechanism. A slow
normalized eligibility budget redistributes that mechanism to repeatedly active
sites. No bonds, molecular Na/K channels or spontaneous neurobiology are claimed.

This is a mechanistic fork: demonstrate (or falsify) whether memory-gated firing,
activity-dependent firing-region localization and recovery coexist in a shared
physical wave medium. Compare fixed gain, memory-gate ablation, and no-exciter.

Run: python ais_chain_experiment.py --out results/ais
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import math
from phase_spike_sol import kernel

@dataclass(frozen=True)
class Params:
    n: int = 48
    pairs: int = 5
    x: int = 24
    y0: float = 17.0
    pair_spacing: float = 5.0
    bead_separation: float = 2.8
    g_emit: float = .085
    source_x: int = 17
    g_phase: float = .02
    noise: float = 0.
    input_gain: float = .10
    dt_fhn: float = .10
    fhn_epsilon: float = .05
    envelope_tau: float = 18.
    sensor_gain: float = 12.0
    emission_gain: float = .13
    plastic_rate: float = .002
    recovery_rate: float = .001
    seed: int = 17
    phase_init_mode: str = "fixed"  # "random" for holdout tests

class ChainAIS:
    def __init__(self, p: Params, excitable=True, adaptive=True, phase_gate=True,
                 initial_phase=None):
        self.p=p
        self.excitable=excitable
        self.adaptive=adaptive
        self.phase_gate=phase_gate
        self.t=0
        self.fields=np.zeros((p.pairs*2+1,p.n,p.n))
        self.prev=np.zeros_like(self.fields)
        self.src=np.zeros_like(self.fields)
        self.coords=[]
        for k in range(p.pairs):
            cy=p.y0+k*p.pair_spacing
            self.coords.extend([(p.x,cy-p.bead_separation/2),(p.x,cy+p.bead_separation/2)])
        self.coords=np.array(self.coords)
        for j,(x,y) in enumerate(self.coords):
            idx,w=kernel(p.n,x,y)
            iy=np.array([a for a,b in idx]);ix=np.array([b for a,b in idx])
            np.add.at(self.src[j],(iy,ix),w)
        self.input_src=[]
        for k in range(p.pairs):
            cy=p.y0+k*p.pair_spacing
            idx,w=kernel(p.n,p.source_x,cy)
            a=np.zeros((p.n,p.n));iy=np.array([x for x,y in idx]);ix=np.array([y for x,y in idx])
            np.add.at(a,(iy,ix),w)
            self.input_src.append(a)
        self.input_src=np.array(self.input_src)
        self.pos_idx=[(np.array([a for a,b in kernel(p.n,x,y)[0]]),
                       np.array([b for a,b in kernel(p.n,x,y)[0]]),
                       np.asarray(kernel(p.n,x,y)[1])) for x,y in self.coords]
        self.om=np.tile([.03,.11],p.pairs)
        self.phase=np.tile([.2,-.7],p.pairs)
        if initial_phase is not None:
            self.phase=np.asarray(initial_phase,dtype=float).copy()
        self.rng=np.random.default_rng(p.seed)
        if p.phase_init_mode == "random" and initial_phase is None:
            self.phase=self.rng.uniform(0,2*np.pi,2*p.pairs)
        self.gain=np.ones(p.pairs)
        self.eligibility=np.zeros(p.pairs)
        self.u=np.full(p.pairs,-1.2)
        self.v=np.full(p.pairs,-.62)
        self.sense=np.zeros(p.pairs)
        self.spikes=np.zeros(p.pairs,dtype=int)
        self.total_spikes=0
        self.activity=np.zeros(p.pairs)
        self.last_drive=np.zeros(p.pairs)
        self.field_records=[]
        self.spike_records=[]
        self.w_records=[]
        self.strong_events=[]

    def phase_memory(self):
        # D_i is invariant for an unforced complementary pair rotating at .03 and .11
        a=self.phase[::2];b=self.phase[1::2]
        return np.angle(np.exp(1j*(a-b+.08*self.t)))

    def step(self, cue_site=None, cue_amplitude=0., cue_phase=0., record=False):
        p=self.p;t=self.t;f=self.fields
        q=np.cos(self.phase)
        heard=np.array([sum(float(np.dot(self.fields[j,iy,ix],w)) for j in range(len(self.phase)) if j!=i)
                        +float(np.dot(self.fields[-1,iy,ix],w))
                        for i,(iy,ix,w) in enumerate(self.pos_idx)])
        coupling=math.cos(.14*t)
        # Actual pumped oscillator phase law used in parent and Sol receiver tests
        self.phase=(self.phase+self.om-p.g_phase*coupling*heard*np.sin(self.phase)/self.om
                    +self.rng.normal(0,p.noise,len(self.phase)))%(2*math.pi)
        # Input-attributed local sensor: distinct from background bead emissions.
        # This is an explicit engineered sensor, not spontaneous detection.
        ext=np.array([float(np.dot(self.fields[-1,iy,ix],w)) for iy,ix,w in self.pos_idx])
        raw=np.array([.5*(abs(ext[2*k])+abs(ext[2*k+1])) for k in range(p.pairs)])
        self.sense+=(raw-self.sense)/p.envelope_tau
        if self.excitable:
            # Explicit mechanism added AFTER parent's field/phase law:
            # phase memory gates a fast excitable amplifier, slow recovery ends it.
            D=self.phase_memory()
            gate=(.5+.5*np.cos(D)) if self.phase_gate else np.ones(p.pairs)
            I=.05+p.sensor_gain*self.sense*gate*self.gain
            self.last_drive=I.copy()
            old=self.u.copy()
            self.u+=p.dt_fhn*(self.u-self.u**3/3-self.v+I)
            self.v+=p.dt_fhn*p.fhn_epsilon*(self.u+.7-.8*self.v)
            events=(old<=.5)&(self.u>.5)
            self.spikes+=events.astype(int)
            self.total_spikes+=int(events.sum())
            if np.any(events):
                self.strong_events.extend([(t,int(j)) for j in np.flatnonzero(events)])
            # The new active part of the field source, from excitable excursions.
            # Unlike forced periodic theta, it becomes a pulse in time.
            boost=np.clip((self.u-.1)*.5,0,1)
            emitted=np.repeat(boost,2)
            if self.adaptive:
                # Bounded local structural capacity: eligibility writes where
                # excitation occurred; fixed global budget avoids gain blowup.
                self.activity += p.plastic_rate*(raw-self.activity)
                # Learning is local but capacity competition is global.
                self.eligibility+=p.plastic_rate*(raw - self.eligibility)
                x=np.maximum(self.eligibility,0)
                if x.max()>1e-8:
                    strength=x/(x.sum()+1e-12)
                    target=.75+1.25*strength*p.pairs
                    self.gain+=p.recovery_rate*(target-self.gain)
            # Don't allow signal gain plasticity to exceed fixed conservative range.
            self.gain=np.clip(self.gain,.5,2.4)
        else:
            emitted=np.zeros(len(self.phase))
        # Parent KG stencil with separate source attribution; the last field is external.
        lap=np.roll(f,1,1)+np.roll(f,-1,1)+np.roll(f,1,2)+np.roll(f,-1,2)-4*f
        new=1.99*f-.99*self.prev+.3*lap-.02*f
        new[:-1]+=(p.g_emit*q+p.emission_gain*emitted)[:,None,None]*self.src[:-1]
        if cue_site is not None and cue_amplitude:
            # A single prescribed oscillation is a convenient repeatable wave probe.
            new[-1]+=cue_amplitude*math.cos(.11*t+cue_phase)*self.input_src[cue_site]
        self.prev,self.fields=self.fields,new
        self.t+=1
        if record:
            # Each recorded panel tracks local physical emission and internal state.
            self.spike_records.append((t,*self.u,*self.spikes))
            self.w_records.append(self.gain.copy())
            self.field_records.append(float(self.fields[:-1,int(round(p.y0+2*p.pair_spacing)),(p.x+7)%p.n].sum()))

    def run(self, steps, cue_site=None, cue_amplitude=0., cue_phase=0., record=False):
        for _ in range(steps):
            self.step(cue_site,cue_amplitude,cue_phase,record)

    def matched_fast_reset(self):
        # fast field AND FHN excitability/recovery identical; leave slow operator
        # and oscillator-phase memory intact. Useful causal ablations after writes.
        self.fields.fill(0);self.prev.fill(0)
        self.u.fill(-1.2);self.v.fill(-.62);self.sense.fill(0)
        self.spikes.fill(0);self.total_spikes=0;self.strong_events=[]

    def snapshot(self):
        return dict(gain=self.gain.copy(),D=self.phase_memory().copy(),
                    spikes=self.spikes.copy(),field_energy=float(np.sum(self.fields**2)),
                    events=list(self.strong_events),
                    abs_activity=self.activity.copy())


def region_experiment(p:Params, site=0, excitable=True, adaptive=True, phase_gate=True,
                      cue_phase=0.,train_steps=1400,hold_steps=500,test_steps=300):
    w=ChainAIS(p,excitable=excitable,adaptive=adaptive,phase_gate=phase_gate)
    w.run(250)
    before=w.snapshot()
    w.run(train_steps,cue_site=site,cue_amplitude=p.input_gain,cue_phase=cue_phase)
    learned=w.snapshot()
    w.run(hold_steps)
    held=w.snapshot()
    w.matched_fast_reset()
    w.run(test_steps,cue_site=site,cue_amplitude=p.input_gain*.6,cue_phase=cue_phase,record=True)
    probed=w.snapshot()
    return w,dict(before=_serial(before),learned=_serial(learned),held=_serial(held),
                  probed=_serial(probed),training_site=site,plastic=adaptive,excitable=excitable,
                  phase_gate=phase_gate)


def _serial(x):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in x.items()}


def run_experiments(p:Params, quick=False):
    train=480 if quick else 1400
    hold=180 if quick else 500
    test=220 if quick else 300
    res={}
    for name,kwargs in [('left_adaptive',dict(site=0,adaptive=True)),
                        ('right_adaptive',dict(site=p.pairs-1,adaptive=True)),
                        ('left_frozen',dict(site=0,adaptive=False)),
                        ('left_ungated',dict(site=0,adaptive=True,phase_gate=False)),
                        ('left_unexcitable',dict(site=0,adaptive=True,excitable=False))]:
        w,a=region_experiment(p,**kwargs,train_steps=train,hold_steps=hold,test_steps=test)
        res[name]=a
        print(name,'gain',np.round(w.gain,3),'spikes',w.spikes.tolist(),'learned',a['learned']['spikes'],flush=True)
    return dict(model='pinned six-to-ten oscillator synthetic chain; original KG/phase, ADDED FHN and homeostatic local gain; no bonds/motion',settings=vars(p),timing=dict(training=train,hold=hold,test=test),results=res)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--out',default='results/ais')
    ap.add_argument('--noise',type=float,default=0.)
    a=ap.parse_args();p=Params(noise=a.noise)
    d=run_experiments(p,a.quick)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(d,indent=2)+'\n')
