import math
import threading
import time
import unittest
from dataclasses import replace
from src.co2_mpc import GasModel, MPCSettings, PulseMPC
from src.co2_control import CO2Control


MODEL = GasModel(18000, 90, 60, 0.00015)
SETTINGS = MPCSettings(horizon_s=480, sample_s=10, min_interval_s=60,
                       min_pulse_s=0.1, pulse_quantum_s=0.1, max_pulse_s=0.5)


class ModelTests(unittest.TestCase):
    def test_delay_mixing_and_leak(self):
        self.assertEqual(MODEL.response(89),0)
        self.assertGreater(MODEL.response(200),0)
        self.assertLess(MODEL.response(10000),MODEL.response(500))
        equal = GasModel(100,0,100,0.01)
        self.assertAlmostEqual(equal.response(100),math.exp(-1))

    def test_input_validation(self):
        for value in [float('nan'),float('inf'),-1,True]:
            with self.assertRaises(ValueError): GasModel(value,0,10,0)
        with self.assertRaises(ValueError): MPCSettings(target_ppm=95000)
        with self.assertRaises(ValueError): PulseMPC(MODEL,replace(SETTINGS,horizon_s=100))

    def test_freshness_and_invalid_sensor(self):
        c=PulseMPC(MODEL,SETTINGS)
        for value,stamp in [(None,100),(float('nan'),100),(96000,100),(1000,1),(1000,101)]:
            with self.assertRaises(ValueError):c.observe(value,stamp,100)
        self.assertTrue(c.observe(1000,100,100))
        self.assertFalse(c.observe(1000,100,101))
        with self.assertRaises(ValueError): c.propose(121)

    def test_pending_gas_prevents_double_dose(self):
        c=PulseMPC(GasModel(100000,90,60,0),SETTINGS)
        c.observe(49000,100,100)
        c.committed(100,0.5)
        c.observe(49000,160,160)  # concentration unchanged; dose still in transit
        self.assertEqual(c.propose(160),0)

    def test_closed_loop_with_delay_leak_noise(self):
        c=PulseMPC(MODEL,SETTINGS)
        pulses=[];values=[]
        # Independent plant uses a slightly different gain/leakage and deterministic noise.
        plant=replace(MODEL,gain_ppm_per_s=19000,leak_per_s=0.00018)
        for t in range(0,3601,10):
            value=plant.ambient_ppm+3000*math.exp(-plant.leak_per_s*t)
            value+=sum(plant.dose_ppm(d)*plant.response(t-start) for start,d in pulses)
            values.append(value)
            c.observe(value+20*math.sin(t),t,t)
            dose=c.propose(t)
            if dose:
                self.assertIn(dose,SETTINGS.pulses())
                if pulses:self.assertGreaterEqual(t-pulses[-1][0],SETTINGS.min_interval_s)
                pulses.append((t,dose));c.committed(t,dose)
        self.assertTrue(pulses)
        self.assertLess(max(values),SETTINGS.max_ppm)
        self.assertLess(abs(sum(values[-60:])/60-50000),3000)


class WorkerTests(unittest.TestCase):
    def profile(self):
        from dataclasses import asdict
        return {'validated':True,'model':asdict(MODEL),'settings':asdict(SETTINGS)}

    def test_unvalidated_refused_without_valve_write(self):
        writes=[]
        c=CO2Control(lambda:(1000,time.monotonic()),writes.append)
        with self.assertRaises(ValueError):c.start(50000)
        self.assertEqual(writes,[])

    def test_stop_closes_and_excludes_other_owner(self):
        writes=[]
        c=CO2Control(lambda:(1000,time.monotonic()),writes.append,self.profile())
        c.start(50000,'program')
        with self.assertRaises(RuntimeError):c.start(50000,'api')
        c.stop()
        self.assertFalse(c.active)
        self.assertFalse(writes[-1])
        self.assertFalse(c._thread.is_alive())

    def test_sensor_failure_latches_fault_and_closes(self):
        writes=[];calls=[]
        def read():
            calls.append(1)
            return (1000 if len(calls)==1 else None,time.monotonic())
        c=CO2Control(read,writes.append,self.profile())
        c.start(50000)
        c._thread.join(2)
        self.assertFalse(c.active)
        self.assertIsNotNone(c.fault)
        self.assertFalse(writes[-1])


if __name__=='__main__':unittest.main()
