"""Opt-in nonlinear pulse MPC and rolling, overlapping-dose gain estimation.

All valve commands still pass the fixed no-loss, pending-dose budget. Learning
changes the operating gain only; it never lowers that fixed safety gain.
"""
from collections import deque
from dataclasses import replace
import math
from .co2_mpc import PulseMPC, finite


class NonlinearPulseMPC(PulseMPC):
    def __init__(self, model, settings, uncertainty=None, average_correction=None):
        super().__init__(model, settings, uncertainty)
        if uncertainty:
            if uncertainty.learning_mode != 'window':
                raise ValueError('nonlinear loss requires window gain learning')
            if uncertainty.learning_window_s < model.settling_s()*uncertainty.kinetics_factor:
                raise ValueError('learning window must cover the slow response')
            if uncertainty.learning_window_s/settings.sample_s > 10000:
                raise ValueError('learning window exceeds 10000 samples')
        self._learning = deque(maxlen=10000)
        self._last_fit_at = -math.inf
        self._estimate = None
        self._pulse_count = 0
        from .co2_average import AverageErrorCorrection
        self.average = AverageErrorCorrection(average_correction, settings) if average_correction else None

    def average_status(self):
        return self.average.status(self.settings) if self.average else {'enabled': False}

    def committed(self, start, actual_s):
        finite(start, 'start'); finite(actual_s, 'actual pulse', strict=True)
        self.doses.append((start, actual_s))
        self.last_dose = start
        self._pulse_count += 1
        # A pulse joins the existing learning window; it never resets it.

    def _released_seconds(self, model, start, end, doses=None):
        return sum(max(0, d-model.valve_dead_s) *
                   (model.arrived_fraction(end-t)-model.arrived_fraction(start-t))
                   for t,d in (self.doses if doses is None else doses))

    def _trajectory(self, model, initial, start, times, doses=None):
        value = initial
        result = []
        for end in times:
            # Small steps are used by observation/predict; planning precomputes
            # the same reservoir integrals at its configured prediction cadence.
            count = max(1, math.ceil((end-start)/self.settings.prediction_step))
            dt = (end-start)/count
            for _ in range(count):
                following = start+dt
                amount = model.gain_ppm_per_s*self._released_seconds(model,start,following,doses)
                value = model.advance(value,amount,dt)
                start = following
            result.append(value)
        return result

    def _learn_window(self, value, stamp):
        u, m = self.uncertainty, self.model
        if not u:
            return
        if self._learning and stamp-self._learning[-1][0] > self.settings.stale_s:
            self._learning.clear()
            self.last_gain_fit = {'state':'rejected','reason':'measurement gap in response'}
        self._learning.append((stamp,value))
        while len(self._learning)>1 and self._learning[1][0] <= stamp-u.learning_window_s:
            self._learning.popleft()
        if len(self._learning)<12 or stamp-self._learning[0][0] < u.learning_window_s:
            if self.last_gain_fit.get('state') != 'rejected':
                self.last_gain_fit = {'state':'collecting_window','reason':'waiting for complete learning window'}
            return
        if stamp-self._last_fit_at < u.learning_interval_s:
            return
        self._last_fit_at = stamp
        first, baseline = self._learning[0]
        # Integral mass balance: change in measured CO2 + measured net loss
        # equals gain * delivered valve seconds. Includes EVERY overlapping dose
        # and gas from pulses before the window that arrives inside it.
        x = integral_loss = 0.0
        xx = xy = 0.0
        points = []
        previous = (first, baseline)
        relevant = [(t,d) for t,d in self.doses if t <= stamp]
        for t,v in list(self._learning)[1:]:
            pt,pv = previous
            integral_loss += (m.loss_rate(pv)+m.loss_rate(v))*(t-pt)/2
            x += self._released_seconds(m,pt,t,relevant)
            y = v-baseline+integral_loss
            points.append((x,y));xx += x*x;xy += x*y
            previous = t,v
        gain = xy/xx if xx>0 else 0.0
        response = gain*x
        rmse = math.sqrt(sum((y-gain*a)**2 for a,y in points)/len(points))
        fit = {'state':'rejected','reason':'unresolved or poor fit',
               'gain_ppm_per_s':gain,'rmse_ppm':rmse,'samples':len(points),
               'window_s':stamp-first,'arrived_valve_s':x,
               'pulse_count':sum(first<=t<=stamp for t,d in relevant)}
        self.last_gain_fit = fit
        if response < u.min_response_ppm or rmse > u.max_relative_rmse*response:
            return
        upper = m.gain_ppm_per_s*self.settings.gain_safety_factor
        if gain > upper:
            fit['reason'] = 'estimated gain exceeds fixed safety gain'
            raise ValueError(fit['reason'])
        self.gain_estimate = (1-u.learning_rate)*self.gain_estimate+u.learning_rate*gain
        self.gain_low = min(u.gain_min_ppm_per_s,self.gain_estimate*0.8,gain*0.8)
        self.gain_high = min(upper,max(u.gain_max_ppm_per_s,self.gain_estimate*1.2,gain*1.2))
        self.gain_updates += 1
        fit.update(state='accepted',reason='rolling mass balance with fixed loss and kinetics')

    def uncertainty_status(self):
        status = super().uncertainty_status()
        status.update(learning_mode='window',loss_exponent=self.model.loss_exponent,
                      loss_reference_ppm=self.model.loss_reference_ppm,
                      issued_pulses=self._pulse_count)
        return status

    def observe(self, value, measured_at, now):
        finite(now,'now');finite(value,'CO2');finite(measured_at,'sample time')
        if measured_at>now or now-measured_at>self.settings.stale_s:
            raise ValueError('CO2 sample is stale or has a future timestamp')
        if value>=self.settings.max_ppm:
            raise ValueError('CO2 upper limit reached')
        if self.last_sample is not None and measured_at<=self.last_sample:
            return False
        m = replace(self.model,gain_ppm_per_s=self.gain_estimate)
        prediction = value if self.time is None else self._trajectory(m,self._estimate,self.time,[measured_at])[0]
        self._learn_window(value,measured_at)
        self._observations.append((measured_at,value))
        cutoff = measured_at-self.settings.stuck_window_s
        while len(self._observations)>1 and self._observations[1][0]<=cutoff:
            self._observations.popleft()
        start,initial = self._observations[0]
        if measured_at-start>=self.settings.stuck_window_s:
            values = [v for _,v in self._observations]
            if max(values)-min(values)<=self.settings.stuck_tolerance_ppm:
                expected = self._trajectory(m,initial,start,[measured_at])[0]-initial
                if expected>=self.settings.stuck_expected_rise_ppm:
                    raise ValueError('CO2 reading is not responding to the predicted injected gas')
        self._estimate = prediction+self.settings.observer_gain*(value-prediction)
        if self.average:
            planned = self.last_plan.get('planned_pulses_s', [])
            constrained = (self.last_plan.get('dose_budget_blocked', False)
                           or self.last_plan.get('feasible') is False
                           or bool(planned and all(d >= self.settings.max_pulse_s for d in planned)))
            self.average.observe(value, measured_at, self.settings, constrained=constrained)
        self.time = self.last_sample = measured_at
        self.last_value = value
        # Keep all unresolved gas and all pulses that can contribute to the
        # learning/stuck windows; discard only negligible ancient tails.
        window = max(self.settings.stuck_window_s,
                     self.uncertainty.learning_window_s if self.uncertainty else 0)
        tail = self.model.settling_s(32)*max(self.settings.delay_safety_factor,self.settings.mixing_safety_factor)
        oldest = measured_at-window-tail
        self.doses[:] = [(t,d) for t,d in self.doses if t>=oldest]
        return True

    def predict(self, t):
        finite(t,'prediction time')
        if self.time is None or t<self.time:
            raise ValueError('prediction must follow an observation')
        return self._trajectory(replace(self.model,gain_ppm_per_s=self.gain_estimate),
                                self._estimate,self.time,[t])[0]

    def propose(self, now):
        s,m = self.settings,self.model
        tracking_target = self.average.target(s) if self.average else s.target_ppm
        if self.time is None or now-self.last_sample>s.stale_s:
            raise ValueError('fresh CO2 observation required')
        if now-self.last_dose<s.min_interval_s:
            return 0.0
        upper = m.gain_ppm_per_s*s.gain_safety_factor
        pending = sum(upper*d*m.remaining_fraction(self.last_sample-t,
                      s.delay_safety_factor,s.mixing_safety_factor) for t,d in self.doses)
        ceiling = s.max_ppm-s.margin_ppm
        choices = [d for d in s.pulses() if d==0 or self.last_value+pending+upper*d<ceiling]
        budget = {'pending_upper_ppm':pending,'dose_budget_ceiling_ppm':ceiling,
                  'dose_budget_blocked':len(choices)==1}
        if len(choices)==1:
            self.last_plan = {**budget,'pulse_s':0.0,'planned_pulses_s':[],
                              'feasible':self.last_value+pending<ceiling}
            return 0.0
        times = [now+i*s.prediction_step for i in range(1,math.ceil(s.horizon_s/s.prediction_step)+1)]
        previous = [self.last_sample]+times[:-1]
        spans = [b-a for a,b in zip(previous,times)]
        moves = [now+i*s.planning_interval for i in range(math.ceil(s.horizon_s/s.planning_interval))]
        models = [replace(m,gain_ppm_per_s=self.gain_estimate)]
        if self.uncertainty:
            f = self.uncertainty.kinetics_factor
            models = [replace(m,gain_ppm_per_s=g,delay_s=m.delay_s*k,mixing_s=m.mixing_s*k,
                       slow_delay_s=m.slow_delay_s*k,slow_mixing_s=m.slow_mixing_s*k,
                       leak_per_s=m.leak_per_s/k)
                      for g in (self.gain_low,self.gain_estimate,self.gain_high) for k in (1/f,1.0,f)]
        # Fixed upper gain forecast supplements the immediate no-loss gas budget.
        models.append(replace(m,gain_ppm_per_s=upper,leak_per_s=m.leak_per_s/s.mixing_safety_factor))
        bases=[];kernels=[]
        for model in models:
            bases.append([model.gain_ppm_per_s*self._released_seconds(model,a,b) for a,b in zip(previous,times)])
            kernels.append([[model.gain_ppm_per_s*(model.arrived_fraction(b-t)-model.arrived_fraction(a-t))
                            for a,b in zip(previous,times)] for t in moves])
        amounts=[b[:] for b in bases]
        weights=[math.exp(-(t-now)/s.tracking_time_s) if s.tracking_time_s else 1 for t in times]
        weight_sum=sum(weights);scale=max(s.target_ppm,1000)
        risk=self.uncertainty.risk_weight if self.uncertainty else 0
        def score(inputs,effort):
            forecasts=[];costs=[]
            for model,releases in zip(models,inputs):
                value=self.last_value;pred=[]
                for released,dt in zip(releases,spans):
                    value=model.advance(value,released,dt)
                    if not math.isfinite(value) or value>=ceiling:
                        return math.inf,[]
                    pred.append(value)
                forecasts.append(pred)
                costs.append(sum(w*max(0,abs(v-tracking_target)-s.deadband_ppm)**2*
                                 (4 if v>tracking_target else 1)/scale**2 for w,v in zip(weights,pred))/weight_sum)
            costs=costs[:-1] # last forecast is a guard, not an operating scenario
            return (1-risk)*sum(costs)/len(costs)+risk*max(costs)+s.pulse_cost*effort,forecasts
        doses=[0.0]*len(moves)
        best,forecasts=score(amounts,0)
        for _ in range(2):
            for j in range(len(moves)):
                old=doses[j];effort=sum(doses)-old
                base=[[max(0,a-max(0,old-m.valve_dead_s)*h) for a,h in zip(v,ks[j])]
                      for v,ks in zip(amounts,kernels)]
                chosen=old
                for dose in (choices if j==0 else s.pulses()):
                    trial=[[a+max(0,dose-m.valve_dead_s)*h for a,h in zip(v,ks[j])]
                           for v,ks in zip(base,kernels)]
                    cost,pred=score(trial,effort+dose)
                    if cost<best-1e-12:
                        best,chosen,amounts,forecasts=cost,dose,trial,pred
                doses[j]=chosen
        self.last_plan={**budget,'pulse_s':doses[0],'planned_pulses_s':doses,'feasible':math.isfinite(best),
                        'model_type':'nonlinear_loss','scenario_count':len(models)-1,
                        'tracking_target_ppm':tracking_target}
        if forecasts:
            nominal=forecasts[4] if self.uncertainty else forecasts[0]
            self.last_plan.update(predicted_peak_ppm=max(nominal),predicted_end_ppm=nominal[-1],
                scenario_peak_min_ppm=min(max(p) for p in forecasts[:-1]),
                scenario_peak_max_ppm=max(max(p) for p in forecasts[:-1]),
                safety_predicted_peak_ppm=max(forecasts[-1]))
        return doses[0] if math.isfinite(best) else 0.0
