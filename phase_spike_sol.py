"""Sol's independent phase-spike audit for pumped complementary oscillators.

This is a prescribed-phase emission / exact linear field propagation assay.
It is NOT the full coupled World: phases are imposed, particles do not move,
and feedback, ligation, plasticity and stochastic phase noise are disabled.
Run: python phase_spike_sol.py --out results
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

OMEGA = 0.07
A_FREQ, T_FREQ = 0.03, 0.11


def pair_source(t, difference=0.0, phase_sum=0.0, gain=1.0):
    """Two equal-amplitude oscillators; sum is the phase offset from pump."""
    t = np.asarray(t, dtype=float)
    a = (phase_sum + difference) / 2
    b = (phase_sum - difference) / 2
    return gain * (np.cos(A_FREQ * t + a) + np.cos(T_FREQ * t + b))


def envelope(t, difference=0.0, gain=1.0):
    """Signed beat envelope; magnitude is NOT a physiological action potential."""
    return 2 * gain * np.cos((A_FREQ - T_FREQ) * np.asarray(t) / 2 + difference / 2)


def metrics(x):
    x = np.asarray(x, dtype=float)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    duty = float(np.mean(np.abs(x) >= 0.8 * peak))
    return {"crest_factor": peak / rms, "duty_fraction_above_80pct_peak": duty,
            "peak": peak, "rms": rms}


def kernel(n, x, y):
    """Source stencil identical to chains/world.py _kernel for a fixed bead."""
    ix, iy = int(np.rint(x)), int(np.rint(y))
    indices, weights = [], []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ddx, ddy = ix + dx - x, iy + dy - y
            d2 = ddx * ddx + ddy * ddy
            w = math.exp(-d2 / (2 * 0.8**2)) if d2 <= 2.2**2 else 0.0
            indices.append(((iy + dy) % n, (ix + dx) % n))
            weights.append(w)
    weights = np.array(weights)
    weights /= np.sum(weights)
    return indices, weights


def trace_field(difference, n=64, burn=1100, record=2500, gain=0.1,
                offsets=(5, 10, 16, 24), y_separation=3.6):
    """Exactly the source-driven 2-D Klein-Gordon stencil of World.step.

    The pair sits side by side perpendicular to the probe direction, avoiding
    the geometric advantage one frequency would have if it were closer.
    """
    psi = np.zeros((n, n), dtype=np.float64)
    prev = np.zeros_like(psi)
    x, y = 20.0, n / 2
    k1 = kernel(n, x, y - y_separation / 2)
    k2 = kernel(n, x, y + y_separation / 2)
    pos = [(int(y) % n, (int(x) + d) % n) for d in offsets]
    traces = np.empty((record, len(pos)))
    for t in range(burn + record):
        lap = (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
               np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4 * psi)
        new = 1.99 * psi - 0.99 * prev + 0.3 * lap - 0.02 * psi
        for (indices, w), value in zip((k1, k2),
                                      (gain * np.cos(A_FREQ * t + difference / 2),
                                       gain * np.cos(T_FREQ * t - difference / 2))):
            for (py, px), wi in zip(indices, w):
                new[py, px] += value * wi
        prev, psi = psi, new
        if t >= burn:
            traces[t - burn, :] = [psi[py, px] for py, px in pos]
    return traces


def tone_amplitudes(traces, burn=1100):
    """Joint two-frequency least-squares readout, not an NN or spike detector."""
    t = np.arange(burn, burn + len(traces), dtype=float)
    x = np.column_stack((np.cos(A_FREQ*t), np.sin(A_FREQ*t),
                         np.cos(T_FREQ*t), np.sin(T_FREQ*t), np.ones_like(t)))
    coef = np.linalg.lstsq(x, traces, rcond=None)[0]
    return np.hypot(coef[0], coef[1]), np.hypot(coef[2], coef[3])


def run(out, n=64, burn=1100, record=2500):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    time = np.arange(0, 5000, dtype=float)
    pair = pair_source(time)
    monomer = np.cos(A_FREQ * time)
    analytic_error = float(np.max(np.abs(pair -
                         2 * np.cos(OMEGA*time) * np.cos(-0.04*time))))
    p = metrics(pair)
    m = metrics(monomer)
    d0 = trace_field(0, n=n, burn=burn, record=record)
    d_pi = trace_field(np.pi, n=n, burn=burn, record=record)
    a, t = tone_amplitudes(d0, burn=burn)
    contrast = 2 * np.minimum(a, t) / np.maximum(a + t, 1e-30)
    d05 = trace_field(0, n=n, burn=burn, record=record, gain=0.05)
    scaling_error = float(np.max(np.abs(d0 - 2*d05)))
    difference_rms = np.sqrt(np.mean((d0 - d_pi)**2, axis=0))
    result = {
        "scope": "prescribed-phase source + exact linear KG field, no feedback/bonds",
        "n": n, "burn": burn, "record": record, "probe_offsets": [5,10,16,24],
        "analytic_identity_max_error": analytic_error,
        "single_monomer": m, "complementary_pair": p,
        "source_ratio_pair_over_single_crest": p["crest_factor"] / m["crest_factor"],
        "field_amplitude_A_by_distance": a.tolist(),
        "field_amplitude_T_by_distance": t.tolist(),
        "field_two_tone_modulation_contrast": contrast.tolist(),
        "phase_state_field_difference_rms": difference_rms.tolist(),
        "field_source_scaling_max_error": scaling_error,
        "envelope_signed_period": 2*math.pi/0.04,
        "envelope_absolute_peak_interval": math.pi/0.04,
        "pump_carrier_period": 2*math.pi/OMEGA,
        "interpretation": "beats exist for any two coherent tones; source identity alone gives no active spike threshold or regeneration",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
        sel = time[:700]
        ax[0].plot(sel, pair[:700], label="A+T emitted signal", lw=1.2)
        ax[0].plot(sel, envelope(sel), "--", label="signed envelope", lw=1, alpha=.7)
        ax[0].plot(sel, -envelope(sel), "--", lw=1, alpha=.7)
        ax[0].set(title="Pair source: periodic beating, not yet an action potential", xlabel="simulation step", ylabel="source amplitude")
        ax[0].legend(fontsize=8)
        ax[1].plot(np.arange(record), d0[:, 0], label="D=0 near field", lw=.9)
        ax[1].plot(np.arange(record), d_pi[:, 0], label="D=pi near field", lw=.9, alpha=.75)
        ax[1].set(title="Field reads the prescribed phase state", xlabel="steps after burn", ylabel="near probe field")
        ax[1].legend(fontsize=8)
        ax[2].plot([5,10,16,24], contrast, "o-", label="two-tone contrast")
        ax[2].set(xlabel="probe distance (grid cells)", ylabel="2 min(A,T)/(A+T)", ylim=(0,1.05), title="Frequency-dependent propagation changes the beat")
        fig.savefig(out / "audit.png", dpi=130)
        plt.close(fig)
    except ImportError:
        pass
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--burn", type=int, default=1100)
    ap.add_argument("--record", type=int, default=2500)
    args = ap.parse_args()
    result = run(args.out, n=args.n, burn=args.burn, record=args.record)
    print(json.dumps({k:v for k,v in result.items() if k not in ("single_monomer", "complementary_pair")}, indent=2))
