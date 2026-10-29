"""Regenerate the two headline plots from saved JSON results (no simulator run)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(folder='results/regeneration'):
    folder=Path(folder)
    sweep=json.loads((folder/'gain_sweep.json').read_text())['rows']
    noisy=json.loads((folder/'seed_sweep_default_noise.json').read_text())['rows']
    gains=[r['gain'] for r in sweep]
    a=[r['sender_bit_rms_far'] for r in sweep]
    b=[r['receiver_bit_rms_far'] for r in sweep]
    n=[r['seed'] for r in noisy]
    f=[r['active/write_receiver_to_direct_contrast_ratio'] for r in noisy]
    plt.rcParams['svg.fonttype']='none'
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    axes[0].plot(gains,a,'o-',label='Direct sender (far)')
    axes[0].plot(gains,b,'o-',label='Receiver emission (far)')
    axes[0].set(xlabel='Write strength relative to nominal',ylabel='Bit-contrast RMS at same far probe',
                title='Powered re-emission; no abrupt threshold')
    axes[0].legend(fontsize=8)
    axes[1].bar(n,f,label='Observed gain ratio')
    axes[1].axhline(1.,linestyle='--',label='Unity')
    axes[1].set(xlabel='Independent initial phase / noise seed',ylabel='Receiver/direct bit contrast',
                title='Gain is not guaranteed across initial states')
    axes[1].set_ylim(0,max(1.8,max(f)*1.1))
    axes[1].legend(fontsize=8)
    fig.savefig(folder/'summary.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
