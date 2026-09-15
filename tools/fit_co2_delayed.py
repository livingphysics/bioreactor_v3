"""Fit fast and slow delayed mixing paths with shared net loss (NumPy only).

CSV columns: t_s, co2_ppm, dose_s. Dose must be accounted electrical duration,
not an assumed command duration. Models always remain unvalidated. An extra
path is a phenomenological candidate, not evidence of a physical reservoir.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
from tools.fit_co2 import response


def nonnegative_fit(x, y):
    """Exact active-set least squares for at most three nonnegative amplitudes."""
    best = (float(y@y), np.zeros(x.shape[1]))
    for count in range(1, x.shape[1]+1):
        for active in itertools.combinations(range(x.shape[1]), count):
            b = np.linalg.lstsq(x[:,active], y, rcond=None)[0]
            if np.any(b < 0):
                continue
            beta = np.zeros(x.shape[1]); beta[list(active)] = b
            loss = float(np.sum((x@beta-y)**2))
            if loss < best[0]:best = loss,beta
    return best[1]


def predict(t, initial_ppm, pulses, profile):
    m = profile['model']; t = np.asarray(t,dtype=float)
    y = m['ambient_ppm']+(initial_ppm-m['ambient_ppm'])*np.exp(-m['leak_per_s']*t)
    for start,duration in pulses:
        amount = m['gain_ppm_per_s']*max(0,duration-m['valve_dead_s'])
        y += amount*((1-m['slow_fraction'])*response(t-start,m['delay_s'],m['mixing_s'],m['leak_per_s'])
                     +m['slow_fraction']*response(t-start,m['slow_delay_s'],m['slow_mixing_s'],m['leak_per_s']))
    y += profile['fit']['initial_pending_ppm']*response(t,0,m['mixing_s'],m['leak_per_s'])
    return y


def fit(t,y,doses,ambient=420,*,slow=True,search_samples=400,seed=7):
    t,y,doses=(np.asarray(v,dtype=float) for v in (t,y,doses))
    if any(v.ndim!=1 for v in (t,y,doses)) or not len(t)==len(y)==len(doses) or len(t)<30:
        raise ValueError('at least 30 aligned one-dimensional samples required')
    if not all(np.all(np.isfinite(v)) for v in (t,y,doses)) or not math.isfinite(ambient) or ambient<0:
        raise ValueError('finite data and nonnegative ambient required')
    if np.any(np.diff(t)<=0) or np.any(y<0) or np.any(doses<0):
        raise ValueError('increasing timestamps and nonnegative readings/doses required')
    t=t-t[0]; pulses=[(a,d) for a,d in zip(t,doses) if d>0]
    if not pulses or t[-1]-pulses[-1][0]<600:
        raise ValueError('known pulses and at least ten minutes of subsequent observations required')
    if search_samples<1:raise ValueError('positive search_samples required')
    # Equal elapsed-time weighting; cap gaps rather than giving missing periods weight.
    gaps=np.minimum(np.diff(t),60)
    weights=np.sqrt(np.r_[gaps[0],(gaps[:-1]+gaps[1:])/2,gaps[-1]])
    weights/=np.sqrt(np.mean(weights**2))
    span=t[-1]
    # Parameters: fast delay, log fast tau, slow delay increment, log slow tau ratio, log loss.
    lo=np.array([0,math.log(10),0,0,math.log(1e-7)])
    hi=np.array([min(600,span/4),math.log(min(1200,span/4)),
                 min(7200,span/2),math.log(30),math.log(.002)])
    if not slow: hi[2:4]=0
    def evaluate(p):
        delay,tau,sd,st,k=p[0],math.exp(p[1]),p[0]+p[2],math.exp(p[1]+p[3]),math.exp(p[4])
        if k*tau>=.5 or (slow and k*st>=.8):return math.inf,None
        fast=sum(d*response(t-a,delay,tau,k) for a,d in pulses)
        cols=[fast]
        if slow:cols.append(sum(d*response(t-a,sd,st,k) for a,d in pulses))
        cols.append(response(t,0,tau,k))
        x=np.column_stack(cols)
        baseline=ambient+(y[0]-ambient)*np.exp(-k*t)
        beta=nonnegative_fit(x*weights[:,None],(y-baseline)*weights)
        return float(np.mean(((baseline+x@beta-y)*weights)**2)),beta
    rng=np.random.default_rng(seed)
    starts=[np.array([150,math.log(200),1800,math.log(5),math.log(4e-5)])]
    starts.extend(lo+(hi-lo)*rng.random(5) for _ in range(search_samples))
    candidates=[]
    for p in starts:
        p=np.clip(p,lo,hi);loss,beta=evaluate(p)
        if np.isfinite(loss):candidates.append((loss,p,beta))
    best=None
    for loss,p,beta in sorted(candidates,key=lambda x:x[0])[:6]:
        steps=(hi-lo)/8
        for _ in range(45):
            improved=False
            for j in range(5):
                if steps[j]==0:continue
                for sign in [-1,1]:
                    q=p.copy();q[j]=np.clip(q[j]+sign*steps[j],lo[j],hi[j])
                    score,b=evaluate(q)
                    if score<loss:
                        loss,p,beta=score,q,b;improved=True
            if not improved:steps*=.55
        if best is None or loss<best[0]:best=loss,p,beta
    loss,p,beta=best
    gain=float(sum(beta[:-1]));fraction=float(beta[1]/gain) if slow and gain>0 else 0
    model={'gain_ppm_per_s':gain,'delay_s':float(p[0]),'mixing_s':math.exp(p[1]),
           'leak_per_s':math.exp(p[4]),'ambient_ppm':ambient,'valve_dead_s':0,
           'slow_fraction':fraction,'slow_delay_s':float(p[0]+p[2]),
           'slow_mixing_s':math.exp(p[1]+p[3])}
    warnings=['Valve dead time is not fitted; use repeat pulses with known electrical durations.',
              'A second path does not establish a physical reservoir; validate on independent injections.',
              'Initial pending gas is represented only by the fast path; unknown earlier doses can bias parameters.']
    if len(pulses)<3:warnings.append('Fewer than three pulses: model identification remains weak.')
    if gain<=0:warnings.append('No positive injection gain identified; profile cannot be used for control.')
    result={'validated':False,'model':model,'fit':{'weighted_rmse_ppm':math.sqrt(loss),
            'initial_pending_ppm':float(beta[-1]),'samples':len(t),'pulse_count':len(pulses),
            'slow_path_enabled':slow,'warnings':warnings}}
    pred=predict(t,y[0],pulses,result)
    result['fit']['rmse_ppm']=float(np.sqrt(np.mean((pred-y)**2)))
    return result,pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv');p.add_argument('--output',required=True)
    p.add_argument('--ambient',type=float,default=420);p.add_argument('--single-path',action='store_true')
    args=p.parse_args()
    with open(args.csv) as f:rows=list(csv.DictReader(f))
    result,_=fit(*[[float(r[k]) for r in rows] for k in ['t_s','co2_ppm','dose_s']],
                 ambient=args.ambient,slow=not args.single_path)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
