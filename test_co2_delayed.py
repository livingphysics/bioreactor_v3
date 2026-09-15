"""Independent finite-step plant checks for delayed fast/slow gas responses."""
import math
import unittest
from dataclasses import replace
import numpy as np
from src.co2_mpc import GasModel, MPCSettings, PulseMPC
from src.co2_control import CO2Control
from tools.fit_co2_delayed import fit, predict


class DelayedModelTests(unittest.TestCase):
    def test_response_matches_independent_two_reservoir_integration(self):
        m=GasModel(5000,120,180,4e-5,slow_fraction=.3,slow_delay_s=600,slow_mixing_s=900)
        fast=slow=excess=0.0
        for t in range(1,7201):
            if t==120:fast+=.7
            if t==600:slow+=.3
            incoming=fast/180+slow/900
            fast-=fast/180;slow-=slow/900
            excess+=incoming-4e-5*excess
            if t%60==0:self.assertAlmostEqual(m.response(t),excess,delta=.006)

    def test_old_profiles_remain_single_path(self):
        m=GasModel(100,30,60,0)
        self.assertAlmostEqual(m.response(90),1-math.exp(-1))
        self.assertEqual(m.settling_s(5),330)

    def test_invalid_fraction_and_slow_time(self):
        for kwargs in [{'slow_fraction':1.1},{'slow_fraction':-1},{'slow_mixing_s':0}]:
            with self.assertRaises(ValueError):GasModel(100,30,60,0,**kwargs)

    def test_horizon_and_restart_cover_slow_gas(self):
        m=GasModel(100,30,60,0,slow_fraction=.4,slow_delay_s=1000,slow_mixing_s=500)
        with self.assertRaises(ValueError):PulseMPC(m,MPCSettings(horizon_s=1800))
        s=MPCSettings(horizon_s=3600,planning_interval_s=300)
        PulseMPC(m,s)
        from dataclasses import asdict
        worker=CO2Control(lambda:(1000,2000),lambda _:None,
            {'validated':True,'model':asdict(m),'settings':asdict(s)},clock=lambda:2000)
        worker._last_stop=0
        with self.assertRaisesRegex(RuntimeError,'settle'):worker.start(50000)

    def test_pending_slow_gas_suppresses_new_pulse(self):
        m=GasModel(100000,30,30,0,slow_fraction=.9,slow_delay_s=600,slow_mixing_s=60)
        s=MPCSettings(target_ppm=50000,horizon_s=1200,min_pulse_s=.25,max_pulse_s=.5,pulse_quantum_s=.25)
        c=PulseMPC(m,s);c.observe(48000,0,0);c.committed(0,.25)
        c.observe(50000,120,120)
        self.assertEqual(c.propose(120),0)
        self.assertGreater(c.predict(1000),65000)

    def test_coarse_planning_keeps_sensor_freshness(self):
        s=MPCSettings(horizon_s=21600,prediction_step_s=30,planning_interval_s=600)
        self.assertEqual(s.sample_s,5);self.assertEqual(s.stale_s,20)
        c=PulseMPC(GasModel(1000,120,200,4e-5),s)
        c.observe(1000,0,0)
        with self.assertRaises(ValueError):c.propose(21)

    def test_fresh_but_frozen_sensor_faults_after_expected_rise(self):
        m=GasModel(10000,60,60,0)
        c=PulseMPC(m,MPCSettings())
        c.observe(4000,0,0);c.committed(0,.5)
        for t in range(5,600,5):c.observe(4000,t,t)
        with self.assertRaisesRegex(ValueError,'not responding'):c.observe(4000,600,600)

    def test_constant_reading_without_injection_is_not_a_stuck_fault(self):
        c=PulseMPC(GasModel(10000,60,60,0),MPCSettings())
        for t in range(0,1201,5):c.observe(4000,t,t)


class DelayedFitTests(unittest.TestCase):
    def test_prediction_of_held_out_pulse_from_independent_plant(self):
        t=np.arange(0,18001,30,dtype=float);y=[];d=np.zeros(len(t))
        events={600:.5,6600:.3,12600:.4}
        fast=slow=0.;excess=2580.
        for step in range(18001):
            if step in events:d[step//30]=events[step]
            if step%30==0:y.append(420+excess+5*math.sin(step/40))
            fast+=events.get(step-120,0)*5000*.7
            slow+=events.get(step-600,0)*5000*.3
            incoming=fast/180+slow/900
            fast-=fast/180;slow-=slow/900;excess+=incoming-4e-5*excess
        y=np.array(y);train=t<=12000
        profile,_=fit(t[train],y[train],d[train],search_samples=250)
        predicted=predict(t,y[0],list(events.items()),profile)
        self.assertFalse(profile['validated'])
        self.assertLess(np.sqrt(np.mean((predicted[~train]-y[~train])**2)),100)

    def test_rejects_bad_data(self):
        t=np.arange(100)*10.;y=t*0+1000;d=t*0
        with self.assertRaises(ValueError):fit(t,y,d)
        d[1]=.2;y[10]=np.nan
        with self.assertRaises(ValueError):fit(t,y,d)


if __name__=='__main__':unittest.main()
