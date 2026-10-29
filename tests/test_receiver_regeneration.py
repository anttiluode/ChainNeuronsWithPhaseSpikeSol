"""Causal controls for a pinned A/T field-driven receiving pair."""
import math
import unittest
import numpy as np

from receiver_regeneration import (Settings, TwoPairWorld, circ_sep,
                                   demod_complex, run_trial, run_assay)


class ReceiverRegenerationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Settings(n=32, source_x=5., receiver_x=12., far_x=20,
                          y=16., write_steps=130, hold_steps=150,
                          probe_steps=120, write_record=90, phase_noise=0.)

    def test_fields_are_matched_in_both_time_slices(self):
        a=TwoPairWorld(self.cfg)
        a.advance(50,carrier_phase=1.1)
        self.assertGreater(np.max(np.abs(a.fields)),0.)
        self.assertGreater(np.max(np.abs(a.prev)),0.)
        a.graft_field()
        self.assertTrue(np.array_equal(a.fields,np.zeros_like(a.fields)))
        self.assertTrue(np.array_equal(a.prev,np.zeros_like(a.prev)))

    def test_phase_ablated_histories_become_identical(self):
        a=run_trial(self.cfg,0,memory_reset=True)
        b=run_trial(self.cfg,1,memory_reset=True)
        self.assertTrue(np.array_equal(a['trace'],b['trace']))

    def test_passive_receiver_cannot_record_sender_bit(self):
        a=run_trial(self.cfg,0,active=False)
        b=run_trial(self.cfg,1,active=False)
        self.assertTrue(np.array_equal(a['trace'][:,1],b['trace'][:,1]))
        self.assertGreater(np.sqrt(np.mean((a['write_trace'][:,0]-b['write_trace'][:,0])**2)),1e-7)

    def test_passive_receiver_has_no_conditional_probe_response(self):
        a=run_trial(self.cfg,0,active=False,test_gain=.25)
        b=run_trial(self.cfg,0,active=False,test_gain=0.)
        self.assertTrue(np.array_equal(a['trace'][:,1],b['trace'][:,1]))

    def test_source_group_is_unaffected_by_receiver(self):
        a=run_trial(self.cfg,0,active=False)
        b=run_trial(self.cfg,0,active=True)
        self.assertTrue(np.array_equal(a['trace'][:,0],b['trace'][:,0]))

    def test_demodulation(self):
        t=np.arange(350,1500)
        sig=2.3*np.cos(.03*t+.4)
        z=demod_complex(sig,350,.03)
        self.assertAlmostEqual(abs(z),2.3,places=9)
        self.assertAlmostEqual(float(np.angle(z)),.4,places=9)

    def test_full_assay_and_ablated_control(self):
        r=run_assay(self.cfg)
        self.assertGreater(r['active']['write_phase_separation'],1e-5)
        self.assertGreater(r['active']['late_receiver_trace_rms_difference'],1e-7)
        self.assertGreater(r['active']['second_probe_conditional_response_rms_difference'],1e-9)
        self.assertEqual(r['passive']['late_receiver_trace_rms_difference'],0.)
        self.assertEqual(r['passive']['second_probe_conditional_response_rms_difference'],0.)
        self.assertEqual(r['ablated']['late_receiver_trace_rms_difference'],0.)
        self.assertEqual(r['ablated']['second_probe_conditional_response_rms_difference'],0.)


if __name__ == '__main__':
    unittest.main()
