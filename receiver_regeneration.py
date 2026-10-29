"""Receiver-side follow-up to Sol's passive phase-beat audit.

Mechanism: fixed A/T sender and A/T receiver pairs are coupled through *the
actual damped Klein–Gordon lattice* and the original receiver-mode phase law.
No prescribed spike, threshold, Hebbian weight, bond or reinforcement exists.

We evolve separate source-attributed KG fields to remove each bead's own field
exactly from its phase hearing, following the original World's hear_self=False
intent. Phase and field histories are causal. Static geometry and a prescribed
sender are declared interventions, not an autonomous freely growing soup.

Run: python receiver_regeneration.py --out results/regeneration
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

from phase_spike_sol import kernel

FREQS = np.array([.03, .11, .03, .11])
PUMP_OMEGA = .07


@dataclass(frozen=True)
class Settings:
    n: int = 48
    source_x: float = 9.0
    receiver_x: float = 18.0
    y: float = 24.0
    separation: float = 3.6
    far_x: int = 27
    gain: float = .1
    g_phase: float = .02
    phase_noise: float = 0.0
    pump_on: bool = True
    seed: int = 42
    write_steps: int = 1100
    hold_steps: int = 1200
    probe_steps: int = 750
    write_record: int = 450


class TwoPairWorld:
    """Stationary four-bead reduction of parent World (same KG + phase update).

    Sender phases are externally prescribed, and can be silenced (without
    deleting the KG field). Receivers listen to every *other* source but never
    their own, avoiding the approximate self-field calibration of the parent.
    Does not include mechanical forces or bond changes, therefore 'chain' only
    refers to its two-bead spatial arrangement in this assay.
    """

    def __init__(self, cfg: Settings, active=True, receiver_init=None):
        self.cfg = cfg
        self.fields = np.zeros((4, cfg.n, cfg.n), dtype=np.float64)
        self.prev = np.zeros_like(self.fields)
        self.t = 0
        self.active = active
        self.phase = np.array([.1, -.2, .33, -.71], dtype=np.float64)
        if receiver_init is not None:
            self.phase[2:] = np.asarray(receiver_init, dtype=float)
        self.rng = np.random.default_rng(cfg.seed)
        self.stencils = []
        self.hear_index = []
        self.src_grids = np.zeros_like(self.fields)
        coords = ((cfg.source_x, cfg.y-cfg.separation/2),
                  (cfg.source_x, cfg.y+cfg.separation/2),
                  (cfg.receiver_x, cfg.y-cfg.separation/2),
                  (cfg.receiver_x, cfg.y+cfg.separation/2))
        for i,(x,y) in enumerate(coords):
            ids, weights = kernel(cfg.n,x,y)
            iy = np.asarray([p[0] for p in ids], dtype=int)
            ix = np.asarray([p[1] for p in ids], dtype=int)
            self.hear_index.append((iy,ix,weights))
            np.add.at(self.src_grids[i], (iy,ix), weights)
        self.record_near = (int(cfg.y) % cfg.n, int(cfg.receiver_x-3) % cfg.n)
        self.record_far = (int(cfg.y) % cfg.n, cfg.far_x % cfg.n)
        self.receiver_history = []

    def step(self, carrier_phase=0.0, sender_gain=1.0, noise=None, record=False):
        t=self.t
        # Prescribed clock-driving sender. Two carrier phases encode a memory
        # address in the *relative phase* of the transmitted tones.
        self.phase[0] = (.03*t + carrier_phase/2) % (2*math.pi)
        self.phase[1] = (.11*t - carrier_phase/2) % (2*math.pi)
        q = np.cos(self.phase)             # Parent emits with phase BEFORE update
        heard=[]
        for i in (2,3):
            iy, ix, w = self.hear_index[i]
            # Exact removal of the local bead's own contribution (parent intent)
            heard.append(sum(float(np.dot(self.fields[j,iy,ix], w)) for j in range(4) if j!=i))
        if noise is None:
            noise=self.rng.normal(0,self.cfg.phase_noise,2)
        if self.active:
            c=math.cos(2*PUMP_OMEGA*t) if self.cfg.pump_on else 0.
            self.phase[2:] = (self.phase[2:] + FREQS[2:] -
                self.cfg.g_phase*c*np.asarray(heard)*np.sin(self.phase[2:])/FREQS[2:]
                +noise) % (2*math.pi)
        else:
            self.phase[2:] = (self.phase[2:]+FREQS[2:]+noise)%(2*math.pi)
        # Field: each attributed KG component obeys precisely the parent's
        # five-point stencil and source kernel, with no backreaction on field mass.
        lap = (np.roll(self.fields,1,1)+np.roll(self.fields,-1,1)+
               np.roll(self.fields,1,2)+np.roll(self.fields,-1,2)-4*self.fields)
        gains=np.array([sender_gain, sender_gain, 1.,1.])*self.cfg.gain
        new = 1.99*self.fields-.99*self.prev+.3*lap-.02*self.fields
        new += (gains*q)[:,None,None]*self.src_grids
        self.prev,self.fields=self.fields,new
        self.t+=1
        if record:
            y0,x0=self.record_far
            # Same spatial readout for sender-only, receiver-only and both
            self.receiver_history.append((self.fields[:2,y0,x0].sum(),
                                          self.fields[2:,y0,x0].sum(),
                                          self.fields[:,y0,x0].sum(),
                                          *self.phase[2:]))

    def advance(self, count, carrier_phase=0., sender_gain=1., record=False):
        for _ in range(count):
            self.step(carrier_phase=carrier_phase, sender_gain=sender_gain,record=record)

    def graft_field(self, other=None):
        """Match ALL KG degrees of freedom, including previous time slices."""
        if other is None:
            self.fields.fill(0)
            self.prev.fill(0)
        else:
            np.copyto(self.fields,other.fields)
            np.copyto(self.prev,other.prev)

    def phase_difference(self):
        """Free oscillator coordinate in rotating frame, wrapped to [-pi,pi]."""
        d=self.phase[2]-self.phase[3]+(.11-.03)*self.t
        return float(np.angle(np.exp(1j*d)))

    def phase_sum(self):
        s=self.phase[2]+self.phase[3]-(.03+.11)*self.t
        return float(np.angle(np.exp(1j*s)))


def circ_sep(a,b):
    return float(abs(np.angle(np.exp(1j*(a-b)))))


def demod_complex(trace, start, freq):
    t=np.arange(start,start+len(trace),dtype=float)
    x=np.column_stack((np.cos(freq*t),np.sin(freq*t),np.ones(len(t))))
    co=np.linalg.lstsq(x,np.asarray(trace),rcond=None)[0]
    return complex(float(co[0]),-float(co[1]))


def measure_field_gain(arr, start):
    """Demodulate each tone and attribute far fields to actual source groups."""
    result={}
    for i,name in enumerate(('sender','receiver','total')):
        x=arr[:,i]
        a=demod_complex(x,start,.03)
        t=demod_complex(x,start,.11)
        result[name]={'A_magnitude':abs(a),'T_magnitude':abs(t),
                      'A_phase':float(np.angle(a)),'T_phase':float(np.angle(t)),
                      'rms':float(np.sqrt(np.mean(x*x))),
                      'contrast':float(2*min(abs(a),abs(t))/(abs(a)+abs(t)+1e-20))}
    return result


def run_trial(cfg, history, active=True, memory_reset=False, sender_gain=1.,
              test_gain=.25, history_carrier=None, history_phase=True, receiver_init=None):
    """Write -> silence -> match both wave slices -> identical test stimulus.

    The history bit changes only sender relative phase during its write window.
    `history_phase=False` gives a no-history control; `memory_reset=True`
    explicitly erases the receiver's internal memory before the test.
    """
    w=TwoPairWorld(cfg,active=active,receiver_init=receiver_init)
    initial_receiver_phase = w.phase[2:].copy()
    d=(0 if history==0 else math.pi) if history_phase else 0.
    if history_carrier is not None:
        d=history_carrier
    w.advance(cfg.write_steps-cfg.write_record,carrier_phase=d,sender_gain=sender_gain)
    w.advance(cfg.write_record,carrier_phase=d,sender_gain=sender_gain,record=True)
    write_trace=np.asarray(w.receiver_history)
    w.receiver_history.clear()
    written=w.phase_difference()
    # Silence sender: the active receiver remains in its normal background pump.
    w.advance(cfg.hold_steps,sender_gain=0.)
    retained=w.phase_difference()
    if memory_reset:
        w.phase[2:]=initial_receiver_phase + FREQS[2:]*w.t
        w.phase[2:]%=2*math.pi
    w.graft_field()            # entire field state identical across all histories
    w.receiver_history.clear()
    test_start=w.t
    # Identical weak second signal and drive schedule in both histories
    w.advance(cfg.probe_steps,carrier_phase=.25,sender_gain=test_gain,record=True)
    arr=np.array(w.receiver_history)
    return {'written_difference':written,'retained_difference':retained,
            'post_probe_difference':w.phase_difference(),
            'receiver_phase_final':[float(w.phase[2]),float(w.phase[3])],
            'far':measure_field_gain(arr,test_start),
            'trace':arr,'probe_start':test_start,
            'write_trace':write_trace,'write_start':cfg.write_steps-cfg.write_record}


def run_assay(cfg:Settings, receiver_init=None):
    """Paired causal tests of re-emission and conditional future response."""
    out={}
    for active in (False,True):
        name='active' if active else 'passive'
        # Conditions have identical starting phases and the same random noise.
        runs={(h,p):run_trial(cfg,h,active,test_gain=(.25 if p else 0.),
                              receiver_init=receiver_init) for h in (0,1) for p in (0,1)}
        a,b=runs[(0,1)],runs[(1,1)]
        a0,b0=runs[(0,0)],runs[(1,0)]
        # Write-window source-coded perturbation at the downstream probe.
        write_direct=b['write_trace'][:,0]-a['write_trace'][:,0]
        write_receiver=b['write_trace'][:,1]-a['write_trace'][:,1]
        write_total=b['write_trace'][:,2]-a['write_trace'][:,2]
        direct_rms=float(np.sqrt(np.mean(write_direct**2)))
        receiver_rms=float(np.sqrt(np.mean(write_receiver**2)))
        # Counterfactual difference in response to a subsequent identical probe,
        # rather than merely the difference in spontaneous receiver emission.
        resp_a=a['trace'][:,1]-a0['trace'][:,1]
        resp_b=b['trace'][:,1]-b0['trace'][:,1]
        out[name]={
            'write_phase_separation':circ_sep(a['written_difference'],b['written_difference']),
            'retained_phase_separation':circ_sep(a['retained_difference'],b['retained_difference']),
            'post_probe_phase_separation':circ_sep(a['post_probe_difference'],b['post_probe_difference']),
            'late_receiver_trace_rms_difference':float(np.sqrt(np.mean((b['trace'][:,1]-a['trace'][:,1])**2))),
            'second_probe_conditional_response_rms_difference':float(np.sqrt(np.mean((resp_b-resp_a)**2))),
            'second_probe_response_rms_history0':float(np.sqrt(np.mean(resp_a**2))),
            'second_probe_response_rms_history1':float(np.sqrt(np.mean(resp_b**2))),
            'write_direct_sender_bit_contrast_rms':direct_rms,
            'write_receiver_bit_contrast_rms':receiver_rms,
            'write_total_bit_contrast_rms':float(np.sqrt(np.mean(write_total**2))),
            'write_receiver_to_direct_contrast_ratio':receiver_rms/(direct_rms+1e-30),
            'probe':{'history0':a['far'],'history1':b['far']},
            '_traces':{'history0':a['trace'],'history1':b['trace'],
                       'history0_unprobed':a0['trace'],'history1_unprobed':b0['trace'],
                       'history0_write':a['write_trace'],'history1_write':b['write_trace']}}
    # Receiver-phase ablation must erase both the late emission difference and
    # the history-contingent susceptibility to the *same* second stimulus.
    runs={(h,p):run_trial(cfg,h,True,memory_reset=True,
                          test_gain=(.25 if p else 0.),receiver_init=receiver_init)
          for h in (0,1) for p in (0,1)}
    a,b=runs[(0,1)],runs[(1,1)]
    ra=a['trace'][:,1]-runs[(0,0)]['trace'][:,1]
    rb=b['trace'][:,1]-runs[(1,0)]['trace'][:,1]
    out['ablated']={
        'late_receiver_trace_rms_difference':float(np.sqrt(np.mean((a['trace'][:,1]-b['trace'][:,1])**2))),
        'second_probe_conditional_response_rms_difference':float(np.sqrt(np.mean((ra-rb)**2))),
        'post_probe_phase_separation':circ_sep(a['post_probe_difference'],b['post_probe_difference'])}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='results/regeneration')
    p.add_argument('--n',type=int,default=48)
    p.add_argument('--write',type=int,default=1100)
    p.add_argument('--hold',type=int,default=1200)
    p.add_argument('--probe',type=int,default=750)
    args=p.parse_args()
    cfg=Settings(n=args.n, y=args.n/2, write_steps=args.write,hold_steps=args.hold,probe_steps=args.probe)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results=run_assay(cfg)
    np.savez_compressed(out/'traces.npz',**{
        f'{arm}_{h}':tr for arm in ('passive','active') for h,tr in results[arm]['_traces'].items()})
    for arm in ('passive','active'):
        del results[arm]['_traces']
    report={'scope':'pinned independent A/T pairs; same KG stencil and receiver phase law; driven sender; no Brownian/ligation/bonds',
            'settings':vars(cfg),'results':results}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        dat=np.load(out/'traces.npz')
        fig,axs=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
        for i,arm in enumerate(('active','passive')):
            for h in ('history0','history1'):
                axs[i].plot(dat[f'{arm}_{h}'][:,1],label=h,alpha=.85,lw=.95)
            axs[i].set(title=f'{arm.title()} receiver emission at far field, matched field state',
                       xlabel='Steps after matched-present graft',ylabel='Receiver field contribution')
            axs[i].legend()
        fig.savefig(out/'traces.svg')
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    main()
