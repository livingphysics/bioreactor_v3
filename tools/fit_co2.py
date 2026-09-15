"""Fit a delayed mixing/leak model from t_s,co2_ppm,dose_s CSV (NumPy only).

Outputs a provisional profile, never marks it validated. Multiple pulse lengths
and a sufficiently long valve-off decay are needed to establish valve resolution
and leakage independently. --ambient is the assumed external CO2 concentration.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def response(t, delay, tau, leak):
    a = np.maximum(0, t-delay)
    if abs(1-leak*tau) < 1e-7:
        return a/tau*np.exp(-a/tau)
    return (np.exp(-leak*a)-np.exp(-a/tau))/(1-leak*tau)


def fit(t, y, doses, ambient=420):
    if not math.isfinite(ambient) or ambient < 0:
        raise ValueError('ambient must be a finite nonnegative ppm value')
    t, y, doses = (np.asarray(v, dtype=float) for v in (t, y, doses))
    if len(t) < 20 or not all(np.all(np.isfinite(v)) for v in (t,y,doses)):
        raise ValueError('at least 20 finite samples required')
    if np.any(np.diff(t) <= 0) or np.any(y < 0) or np.any(doses < 0):
        raise ValueError('timestamps must increase; readings/doses must be nonnegative')
    t = t-t[0]
    pulses = [(a,d) for a,d in zip(t,doses) if d>0]
    if not pulses:
        raise ValueError('no dose events in data')
    dt = float(np.median(np.diff(t)))
    span = t[-1]-pulses[0][0]
    if span < 60:
        raise ValueError('record more of the post-pulse response')
    unique = sorted(set(d for _,d in pulses))
    def evaluate(delay, tau, leak, dead):
        # The two exponential poles can otherwise swap with a compensating gain.
        # This rig model assumes leakage is slower than mixing; report that assumption.
        if leak*tau >= 0.5:
            return math.inf, np.zeros(2), np.zeros_like(y)
        h = sum(max(0,d-dead)*response(t-a,delay,tau,leak) for a,d in pulses)
        h0 = response(t,0,tau,leak)  # unknown mixing already present at recording start
        baseline = ambient+(y[0]-ambient)*np.exp(-leak*t)
        X = np.column_stack([h,h0])
        target = y-baseline
        # Exact two-column nonnegative least squares via active-set enumeration.
        candidates = [np.zeros(2)]
        beta = np.linalg.lstsq(X,target,rcond=None)[0]
        if np.all(beta>=0): candidates.append(beta)
        for j in range(2):
            b = np.zeros(2)
            b[j] = max(0, X[:,j]@target / max(1e-20,X[:,j]@X[:,j]))
            candidates.append(b)
        beta = min(candidates,key=lambda b: np.mean((X@b-target)**2))
        pred = baseline+X@beta
        return float(np.mean((pred-y)**2)), beta, pred
    best = None
    delays = np.linspace(0,min(600,span/2),13)
    taus = np.geomspace(max(2,dt),max(20,span),14)
    leaks = [0, *np.geomspace(1e-7,0.01,13)]
    deadtimes = [0] if len(unique)<2 else np.linspace(0,max(unique)*0.9,8)
    for delay in delays:
        for tau in taus:
            for leak in leaks:
                for dead in deadtimes:
                    result = evaluate(delay,tau,leak,dead)
                    if best is None or result[0] < best[0]:
                        best = (*result, [float(delay),float(tau),float(leak),float(dead)])
    # Refine the nonlinear parameters without requiring SciPy on the Pi.
    steps = [max(dt,span/24),best[3][1]/2,max(1e-7,best[3][2]/2),min(unique)/6]
    for _ in range(9):
        for j in range(4 if len(unique)>1 else 3):
            for sign in (-1,1):
                params = best[3][:]
                params[j] += sign*steps[j]
                if params[0]<0 or params[1]<1 or not 0<=params[2]<=0.02 or not 0<=params[3]<max(unique):
                    continue
                result = evaluate(*params)
                if result[0]<best[0]: best=(*result,params)
        steps=[s/1.6 for s in steps]
    mse,beta,pred,(delay,tau,leak,dead)=best
    peak=int(np.argmax(y))
    warnings=[]
    if len(unique)<2:
        warnings.append('One pulse size: valve dead time/minimum useful pulse is not identifiable.')
    if t[-1]-t[peak] < max(300,2*tau) or y[peak]-y[-1] < max(100,5*math.sqrt(mse)):
        warnings.append('Insufficient resolved decay: leakage remains provisional; extend valve-off observation.')
    if beta[0] <= 0 or max(y)-min(y)<max(100,5*math.sqrt(mse)):
        warnings.append('Pulse gain is not well resolved above model residual/noise.')
    if beta[1]>100:
        warnings.append('Fit includes pre-existing mixing at recording start; repeat from a stable baseline.')
    model={'gain_ppm_per_s':float(beta[0]),'delay_s':delay,'mixing_s':tau,
           'leak_per_s':leak,'ambient_ppm':ambient,'valve_dead_s':dead}
    return {'validated':False,'model':model,'fit':{
        'rmse_ppm':math.sqrt(mse),'samples':len(t),'pulse_sizes_s':unique,
        'initial_pending_ppm':float(beta[1]), 'warnings':warnings,
        'ambient_assumed_ppm':ambient, 'pole_order_assumption':'leak_per_s * mixing_s < 0.5',
        'dose_timing':'CSV energized duration; commanded unless independently measured',
        'leak_half_life_s':math.log(2)/leak if leak>0 else None}}, pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv');p.add_argument('--ambient',type=float,default=420)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    with open(args.csv) as f: rows=list(csv.DictReader(f))
    profile,_=fit(*[[float(r[k]) for r in rows] for k in ('t_s','co2_ppm','dose_s')],ambient=args.ambient)
    Path(args.output).write_text(json.dumps(profile,indent=2,allow_nan=False)+'\n')
    print(json.dumps(profile,indent=2,allow_nan=False))


if __name__=='__main__': main()
