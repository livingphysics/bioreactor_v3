import unittest
import numpy as np
from tools.fit_co2 import fit,response


class FitTests(unittest.TestCase):
    def test_recovers_known_delayed_leaky_pulse(self):
        t=np.arange(0,3601,10,dtype=float)
        doses=np.zeros(len(t));doses[6]=0.5
        y=420+2580*np.exp(-0.0003*t)+18000*0.5*response(t-60,90,120,0.0003)
        profile,pred=fit(t,y,doses)
        self.assertFalse(profile['validated'])
        self.assertLess(profile['fit']['rmse_ppm'],50)
        self.assertAlmostEqual(profile['model']['delay_s'],90,delta=15)
        self.assertAlmostEqual(profile['model']['gain_ppm_per_s'],18000,delta=1500)
        self.assertAlmostEqual(profile['model']['leak_per_s'],0.0003,delta=0.00005)
        self.assertTrue(any('pulse' in w for w in profile['fit']['warnings']))


if __name__=='__main__':unittest.main()
