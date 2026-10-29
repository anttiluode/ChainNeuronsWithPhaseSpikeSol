"""Render the declared AIS-like chain controls; no fitted curves."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt

root=Path('results/ais')
dyn=json.loads((root/'dynamics.json').read_text())
hold=json.loads((root/'holdout_independent.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
a=np.arange(5)
ax[0].plot(a,dyn['gain_left'],marker='o',label='After left training')
ax[0].plot(a,dyn['gain_left_after_hold'],marker='s',label='After silent hold')
ax[0].plot(a,dyn['gain_after_reversed_training'],marker='^',label='After training right')
ax[0].set(xlabel='Chain segment (left → right)',ylabel='Activity-dependent gain',title='A firing region can relocate')
ax[0].legend(fontsize=8)
sweep=dyn['strength_sweep']
ax[1].plot([r['amplitude'] for r in sweep],[r['peak_site0'] for r in sweep],marker='o')
ax[1].axhline(.5,linestyle='--',linewidth=1,label='Event crossing')
ax[1].set(xlabel='Wave-source amplitude',ylabel='Peak excitable state',title='Added nonlinear threshold')
ax[1].legend(fontsize=8)
pp=dyn['paired_pulses'];g=[int(k) for k in pp];second=[pp[k]['second'][0] for k in pp]
ax[2].plot(g,second,marker='o')
ax[2].set(xlabel='Silent steps after first probe',ylabel='Second-probe events at site 0',
          title='Recovery after excitation',yticks=[0,1],ylim=(-.2,1.2))
for a in ax:a.grid(alpha=.18)
fig.suptitle('Pinned oscillator chain + explicitly added excitable controller (not emergent)',fontsize=12)
root.mkdir(parents=True,exist_ok=True)
fig.savefig(root/'figure.svg')
fig.savefig(root/'figure.png',dpi=140)
plt.close(fig)
print('Saved',root/'figure.svg')
