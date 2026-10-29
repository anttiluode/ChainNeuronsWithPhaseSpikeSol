"""Fast, deterministic guards against claiming a normal beat is a spike."""
import math
import unittest
import numpy as np
from phase_spike_sol import pair_source, envelope, metrics, tone_amplitudes, kernel, trace_field


class PhaseSpikeTests(unittest.TestCase):
    def test_exact_beat_identity_with_phase_memory(self):
        t = np.arange(2048)
        for d in (0, .7, math.pi, -2):
            lhs = pair_source(t, difference=d, phase_sum=.21)
            rhs = 2*np.cos(.07*t+.105)*np.cos(-.04*t+d/2)
            self.assertLess(np.max(np.abs(lhs-rhs)), 2e-13)

    def test_phase_state_changes_future_emission(self):
        t = np.arange(3000, 3600)
        self.assertGreater(np.sqrt(np.mean((pair_source(t, 0)-pair_source(t, math.pi))**2)), .2)

    def test_ordinary_single_tone_also_crosses_a_high_threshold(self):
        t = np.arange(10000)
        self.assertGreater(metrics(np.cos(.03*t))["duty_fraction_above_80pct_peak"], .1)

    def test_emission_scale_is_linear(self):
        t = np.arange(3000)
        self.assertTrue(np.array_equal(3*pair_source(t), pair_source(t,gain=3)))

    def test_stencil_normalized(self):
        indices, weights=kernel(64,20,31.2)
        self.assertAlmostEqual(sum(weights),1,places=13)
        self.assertEqual(len(indices),25)

    def test_tone_demodulation(self):
        tt=np.arange(500,4000)
        obs=(2*np.cos(.03*tt+.33)+.7*np.cos(.11*tt-.88))[:,None]
        a,b=tone_amplitudes(obs, burn=500)
        self.assertAlmostEqual(float(a[0]),2,places=10)
        self.assertAlmostEqual(float(b[0]),.7,places=10)

    def test_field_superposition_is_linear_in_prescribed_sources(self):
        a=trace_field(0, n=32, burn=60, record=100, gain=.1, offsets=(5,))
        b=trace_field(0, n=32, burn=60, record=100, gain=.05, offsets=(5,))
        self.assertLess(np.max(np.abs(a-2*b)),1e-12)


if __name__=="__main__":
    unittest.main()
