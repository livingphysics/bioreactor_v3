"""Offline delayed-plant stress scenarios; never accesses hardware or an API."""
import argparse
from dataclasses import replace,asdict
import json
import math
from pathlib import Path
from src.co2_mpc import GasModel,MPCSettings,PulseMPC


def simulate(model,settings,*,gain=1,delay=1,mixing=1,loss=1,failure=None,duration=21600):
    """Finite-step reservoir plant, independent of GasModel.response()."""
    plant=replace(model,gain_ppm_per_s=model.gain_ppm_per_s*gain,
                  delay_s=model.delay_s*delay,slow_delay_s=model.slow_delay_s*delay,
                  mixing_s=model.mixing_s*mixing,slow_mixing_s=model.slow_mixing_s*mixing,
                  leak_per_s=model.leak_per_s*loss)
    c=PulseMPC(model,settings)
    step=5;control_step=60;fast=slow=0.;excess=4000-model.ambient_ppm
    events=[];pulses=[];values=[];fault=None;fault_at=None;frozen=None
    for t in range(0,duration+1,step):
        for arrival,path,amount in events[:]:
            if arrival<=t:
                if path=='fast':fast+=amount
                else:slow+=amount
                events.remove((arrival,path,amount))
        value=plant.ambient_ppm+excess;values.append(value)
        if t%control_step==0 and not fault:
            measured=value+15*math.sin(t/37)
            stamp=t
            if failure and t>=1800:
                if failure=='missing':measured=None
                elif failure=='stale':stamp=t-60
                elif failure=='frozen':
                    if frozen is None:frozen=measured
                    measured=frozen
            try:
                c.observe(measured,stamp,t)
                pulse=c.propose(t)
            except ValueError as e:
                fault=str(e);fault_at=t;pulse=0
            if pulse:
                pulses.append((t,pulse));c.committed(t,pulse)
                amount=plant.gain_ppm_per_s*max(0,pulse-plant.valve_dead_s)
                events.extend([(t+plant.delay_s,'fast',amount*(1-plant.slow_fraction)),
                               (t+plant.slow_delay_s,'slow',amount*plant.slow_fraction)])
        # Exact reservoir release per step with first-order chamber loss.
        release_fast=fast*(-math.expm1(-step/plant.mixing_s))
        release_slow=slow*(-math.expm1(-step/plant.slow_mixing_s))
        fast-=release_fast;slow-=release_slow
        excess=excess*math.exp(-plant.leak_per_s*step)+release_fast+release_slow
    tail=values[-int(7200/step):]
    return {'peak_ppm':max(values),'final_ppm':values[-1],
            'last_two_hours_mean_ppm':sum(tail)/len(tail),
            'last_two_hours_max_target_error_ppm':max(abs(v-settings.target_ppm) for v in tail),
            'pulse_count':len(pulses),'total_energized_s':sum(d for _,d in pulses),
            'minimum_pulse_interval_s':min((b[0]-a[0] for a,b in zip(pulses,pulses[1:])),default=None),
            'fault':fault,'fault_at_s':fault_at,
            'last_plan':c.last_plan,
            'ceiling_passed':max(values)<settings.max_ppm,
            'tracking_passed':not fault and max(abs(v-settings.target_ppm) for v in tail)<=2500,
            'fault_response_passed':bool(fault) if failure else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile');p.add_argument('--output',required=True)
    a=p.parse_args();profile=json.loads(Path(a.profile).read_text());m=GasModel(**profile['model'])
    horizon=math.ceil((m.settling_s()+600)/600)*600
    s=MPCSettings(horizon_s=horizon,prediction_step_s=60,planning_interval_s=600,
                  max_pulse_s=.5,min_pulse_s=.25,pulse_quantum_s=.25,gain_safety_factor=1.5,
                  pulse_cost=.0001,tracking_time_s=1800)
    scenarios={'nominal':{},'half_gain':{'gain':.5},'double_gain':{'gain':2},
               'double_delay':{'delay':2},'double_mixing':{'mixing':2},
               'zero_loss':{'loss':0},'double_loss':{'loss':2},
               'missing_sensor':{'failure':'missing'},'stale_sensor':{'failure':'stale'},
               'frozen_sensor':{'failure':'frozen'}}
    results={}
    for name,kwargs in scenarios.items():
        results[name]=simulate(m,s,**kwargs)
        print(name,json.dumps({k:v for k,v in results[name].items() if k!='last_plan'}),flush=True)
        Path(a.output).write_text(json.dumps({'simulation_only':True,'duration_s':21600,
            'plant_step_s':5,'controller_step_s':60,'model':asdict(m),'settings':asdict(s),
            'scenarios':results},indent=2)+'\n')


if __name__=='__main__':main()
