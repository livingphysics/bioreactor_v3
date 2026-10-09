#!/usr/bin/env python3
"""Replay a recorded master against a simulated follower. Never imports hardware.

Input: API history JSON {points:[{t:<UTC ms>,co2:<ppm>,ring:[r,g,b]}]}.
Profile: CO2_MPC JSON, or a snapshot containing profile. Predictions use the MPC;
the plant independently integrates transport reservoirs and nonlinear loss at 1 s.
"""
import argparse
from collections import deque
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.co2_mpc import GasModel, MPCSettings, ResponseUncertainty, make_controller
from src.co2_average import AverageCorrectionSettings
from src.co2_reference import ReferenceTrajectory, ReferenceUnavailable, number, tracking_metrics


def load_trace(path):
    data=json.loads(Path(path).read_text())
    points=data['points'] if isinstance(data,dict) else data
    # Missing values break interpolation by leaving explicit timestamp gaps.
    selected={p['t']/1000:p['co2'] for p in points
              if number(p.get('t')) and number(p.get('co2')) and 0<p['co2']<100000}
    if len(selected)<2:raise ValueError('trace needs at least two valid samples')
    return sorted(selected.items()), points


def simulate(trace,profile,delay_s,*,initial_ppm=420,plan_s=300,max_hours=24):
    for name,value in (('delay_s',delay_s),('initial_ppm',initial_ppm),('plan_s',plan_s),('max_hours',max_hours)):
        if not number(value) or value <= 0:
            raise ValueError(name+' must be finite and positive')
    model=GasModel(**profile['model'])
    settings=MPCSettings(**{**profile['settings'],'target_ppm':trace[0][1]})
    uncertainty=ResponseUncertainty(**profile['uncertainty']) if profile.get('uncertainty') else None
    average=AverageCorrectionSettings(**profile['average_correction']) if profile.get('average_correction') else None
    engine=make_controller(model,settings,uncertainty,average)
    origin=trace[0][0]
    reference=ReferenceTrajectory(tuple((t-origin+delay_s,v) for t,v in trace),max_gap_s=30)
    engine.set_reference(reference)
    # Start only once a complete known horizon exists. The simulated wall clock
    # at 'now' has seen source points through now, never later master samples.
    start=delay_s
    end=min(trace[-1][0]-origin, start+max_hours*3600)
    end=min(end,trace[-1][0]-origin+delay_s-settings.horizon_s-settings.prediction_step)
    if end<=start:raise ValueError('trace shorter than delay and replay duration')
    value=initial_ppm;reservoirs=[0.,0.];queue=deque();pulses=[];rows=[]
    no_input=value;gaps=0;next_plan=start
    t=start
    while t<end:
        if round(t-start)%round(settings.sample_s)==0:
            try:
                ref=reference.value_at(t)
                engine.set_reference(reference)
                engine.settings=replace(settings,target_ppm=ref)
                engine.observe(value,t,t)
                if t>=next_plan:
                    horizon_end=t+math.ceil(settings.horizon_s/settings.prediction_step)*settings.prediction_step
                    if horizon_end-delay_s>t+1e-6:
                        raise ReferenceUnavailable('delay does not cover the prediction horizon')
                    reference.require_span(t,horizon_end)
                    pulse=engine.propose(t)
                    next_plan=t+plan_s
                    if pulse:
                        engine.committed(t,pulse);pulses.append((t,pulse))
                        for index,fraction,delay in ((0,1-model.slow_fraction,model.delay_s),(1,model.slow_fraction,model.slow_delay_s)):
                            if fraction:queue.append((t+delay,index,model.gain_ppm_per_s*max(0,pulse-model.valve_dead_s)*fraction))
                        queue=deque(sorted(queue))
                rows.append({'time_s':origin+t,'reference_ppm':ref,'follower_ppm':value,
                             'master_time_s':origin+t-delay_s,'no_input_ppm':no_input})
            except ReferenceUnavailable:
                gaps+=1;engine.pause_reference()
                rows.append({'time_s':origin+t,'reference_ppm':None,'follower_ppm':value})
        while queue and queue[0][0]<=t:
            _,index,amount=queue.popleft();reservoirs[index]+=amount
        released=0
        for index,tau in enumerate((model.mixing_s,model.slow_mixing_s)):
            amount=reservoirs[index]*(-math.expm1(-1/tau));reservoirs[index]-=amount;released+=amount
        # Independent explicit 1-second mass balance, not the planner's advance().
        value += released-model.loss_rate(value)
        no_input -= model.loss_rate(no_input)
        t+=1
    baseline=[{**r,'follower_ppm':r.get('no_input_ppm')} for r in rows]
    result={'delay_s':delay_s,'initial_ppm':initial_ppm,'planning_cadence_s':plan_s,
            'simulation_hours':(end-start)/3600,'pulse_count':len(pulses),
            'total_valve_s':sum(d for _,d in pulses),'reference_gap_samples':gaps,
            'metrics':tracking_metrics(rows),'no_input_metrics':tracking_metrics(baseline),
            'note':'Model-based exploratory simulation; not hardware validation. No N2 actuator. No fitted time shift.'}
    return result,rows,pulses


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',required=True);parser.add_argument('--profile',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--delays-min',nargs='+',type=float,default=[15,30,61])
    parser.add_argument('--hours',type=float,default=24);parser.add_argument('--initial-ppm',type=float,default=420)
    parser.add_argument('--plan-s',type=float,default=300)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    trace,raw=load_trace(args.trace);profile=json.loads(Path(args.profile).read_text());profile=profile.get('profile',profile)
    values=[v for _,v in trace]
    report={'master':{'samples':len(trace),'hours':(trace[-1][0]-trace[0][0])/3600,'min_ppm':min(values),
                      'max_ppm':max(values),'median_ppm':statistics.median(values)},
            'nominal_minimum_injected_ppm':profile['model']['gain_ppm_per_s']*max(0,profile['settings']['min_pulse_s']-profile['model'].get('valve_dead_s',0)),
            'cases':[]}
    for minutes in args.delays_min:
        if minutes*60<profile['settings']['horizon_s']:
            report['cases'].append({'delay_s':minutes*60,'state':'insufficient_preview',
                'reason':'Delay shorter than the configured MPC horizon; reduce/revalidate horizon or choose a longer delay.'})
            continue
        result,rows,pulses=simulate(trace,profile,minutes*60,initial_ppm=args.initial_ppm,plan_s=args.plan_s,max_hours=args.hours)
        report['cases'].append(result)
        (out/f'replay-{minutes:g}min.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
            start=rows[0]['time_s'];x=[(r['time_s']-start)/3600 for r in rows]
            axes[0].plot(x,[r['reference_ppm'] for r in rows],label='Master shifted by declared delay')
            axes[0].plot(x,[r['follower_ppm'] for r in rows],label='Simulated follower')
            axes[0].set_ylabel('CO₂ (ppm)');axes[0].legend()
            axes[1].plot(x,[r['follower_ppm']-r['reference_ppm'] if r['reference_ppm'] is not None else math.nan for r in rows])
            axes[1].axhline(0,color='gray',lw=.7);axes[1].set_ylabel('Tracking error (ppm)');axes[1].set_xlabel('Hours of replay')
            fig.suptitle(f'Model simulation · {minutes:g} min delay · {len(pulses)} pulses · {args.plan_s:g} s planning cadence')
            fig.tight_layout();fig.savefig(out/f'replay-{minutes:g}min.png',dpi=150);plt.close(fig)
        except ImportError:
            pass
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
