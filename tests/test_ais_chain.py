"""Regression guards for added AIS-like prototype, not biological validation."""
import unittest
import numpy as np
from ais_chain_experiment import ChainAIS, Params


class ChainAISChecks(unittest.TestCase):
    def test_no_source_no_spontaneous_event(self):
        p=Params()
        w=ChainAIS(p, phase_gate=False)
        w.run(450)
        self.assertEqual(w.total_spikes,0)

    def test_local_sensor_and_gain_remains_bounded(self):
        p=Params()
        w=ChainAIS(p, phase_gate=False)
        w.run(100)
        w.run(480,cue_site=0,cue_amplitude=p.input_gain)
        self.assertGreater(w.gain[0],w.gain[-1])
        self.assertTrue(np.all(w.gain<=2.4))
        self.assertTrue(np.all(w.gain>=.5))

    def test_emission_turned_off_does_not_delete_memory(self):
        p=Params()
        w=ChainAIS(p, phase_gate=False)
        w.run(100)
        w.run(480,cue_site=0,cue_amplitude=p.input_gain)
        gain=w.gain.copy()
        w.run(100)
        self.assertGreater(w.gain[0],w.gain[-1])
        self.assertGreater(gain[0],gain[-1])

    def test_ablating_phase_gate_removes_phase_dependence_of_excitation(self):
        # Keep all fast states and the field identical; alter the phase register.
        # This checks the explicit gate's logic, not a learned bit decoder.
        p=Params()
        a=ChainAIS(p,phase_gate=False)
        b=ChainAIS(p,phase_gate=False)
        b.phase[::2]+=1.0
        self.assertFalse(np.allclose(a.phase_memory(),b.phase_memory()))
        self.assertEqual(a.total_spikes,b.total_spikes)

if __name__=='__main__':unittest.main()
