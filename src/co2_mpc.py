"""Discrete-pulse, receding-horizon CO2 control; no hardware dependencies.

Model: a pulse arrives after delay_s, mixes with time constant mixing_s, then
leaks toward ambient_ppm at leak_per_s. Gain is ppm per effective valve second.
State includes all issued pulses, including gas still in transit. Only the first
move of the optimized pulse sequence is executed; fresh data replans the rest.
"""
from dataclasses import dataclass, asdict, replace
from collections import deque
import math


def finite(value, name, lo=0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if not math.isfinite(value) or value < lo or (strict and value == lo):
        raise ValueError(f'invalid {name}: {value}')
    return float(value)


@dataclass(frozen=True)
class GasModel:
    gain_ppm_per_s: float
    delay_s: float
    mixing_s: float
    leak_per_s: float
    ambient_ppm: float = 420
    valve_dead_s: float = 0
    slow_fraction: float = 0
    slow_delay_s: float = 0
    slow_mixing_s: float = 1
    loss_exponent: float = 1
    loss_reference_ppm: float = 50000

    def __post_init__(self):
        for k, v in asdict(self).items():
            finite(v, k, strict=k in ('gain_ppm_per_s', 'mixing_s', 'slow_mixing_s', 'loss_reference_ppm'))
        if not 1 <= self.loss_exponent <= 3:
            raise ValueError('loss_exponent must be between 1 and 3')
        if self.slow_fraction > 1:
            raise ValueError('slow_fraction must be between zero and one')

    def loss_rate(self, ppm):
        """Net ppm/s loss; leak_per_s is the fractional rate at the reference excess."""
        excess = ppm - self.ambient_ppm
        return math.copysign(self.leak_per_s * self.loss_reference_ppm *
                             (abs(excess) / self.loss_reference_ppm) ** self.loss_exponent, excess)

    def decay(self, ppm, seconds):
        """Exact no-input decay for the linear or concentration-dependent loss law."""
        excess = ppm - self.ambient_ppm
        if self.loss_exponent == 1:
            return self.ambient_ppm + excess * math.exp(-self.leak_per_s * seconds)
        n = self.loss_exponent - 1
        scale = 1 + n*self.leak_per_s*seconds*(abs(excess)/self.loss_reference_ppm)**n
        return self.ambient_ppm + excess / scale**(1/n)

    def advance(self, ppm, released_ppm, seconds):
        """Symmetric split step; released_ppm is integrated reservoir release."""
        return self.decay(self.decay(ppm, seconds/2) + released_ppm, seconds/2)

    def arrived_fraction(self, age_s):
        def path(delay, tau):
            return -math.expm1(-max(0.0, age_s-delay)/tau)
        return ((1-self.slow_fraction)*path(self.delay_s, self.mixing_s)
                + self.slow_fraction*path(self.slow_delay_s, self.slow_mixing_s))

    def settling_s(self, constants=4):
        """Cover every enabled transport/mixing path, including delayed gas."""
        fast = self.delay_s + constants*self.mixing_s if self.slow_fraction < 1 else 0
        slow = self.slow_delay_s + constants*self.slow_mixing_s if self.slow_fraction else 0
        return max(fast, slow)

    def response(self, age_s):
        """Concentration per ppm injected, including transport/mixing/leakage."""
        if self.loss_exponent != 1:
            raise ValueError('Nonlinear loss has no additive impulse response; use trajectory prediction')
        def path(delay, tau):
            t = age_s-delay
            if t <= 0:
                return 0.0
            k = self.leak_per_s
            if abs(1-k*tau) < 1e-7:
                return t/tau * math.exp(-t/tau)
            return (math.exp(-k*t)-math.exp(-t/tau))/(1-k*tau)
        fast=(1-self.slow_fraction)*path(self.delay_s,self.mixing_s) if self.slow_fraction<1 else 0
        slow=self.slow_fraction*path(self.slow_delay_s,self.slow_mixing_s) if self.slow_fraction else 0
        return fast+slow

    def dose_ppm(self, seconds):
        return self.gain_ppm_per_s * max(0, seconds-self.valve_dead_s)

    def remaining_fraction(self, age_s, delay_factor=1, mixing_factor=1):
        """Unarrived fraction with inflated delay/mixing and no credit for loss."""
        def path(delay,tau):
            elapsed=max(0,age_s-delay*delay_factor)
            return math.exp(-elapsed/(tau*mixing_factor))
        return ((1-self.slow_fraction)*path(self.delay_s,self.mixing_s)+
                self.slow_fraction*path(self.slow_delay_s,self.slow_mixing_s))


@dataclass(frozen=True)
class MPCSettings:
    target_ppm: float = 50000
    max_ppm: float = 95000
    margin_ppm: float = 5000
    min_pulse_s: float = 0.25
    max_pulse_s: float = 1.0
    pulse_quantum_s: float = 0.05
    min_interval_s: float = 60
    sample_s: float = 5
    horizon_s: float = 1800
    stale_s: float = 20
    missing_timeout_s: float = 30
    recovery_samples: int = 2
    gain_safety_factor: float = 1.5
    deadband_ppm: float = 300
    observer_gain: float = 0.7
    pulse_cost: float = 0.002
    prediction_step_s: float = 0  # zero uses sample_s; sensing cadence is unchanged
    planning_interval_s: float = 0  # zero uses min_interval_s
    stuck_window_s: float = 600
    stuck_tolerance_ppm: float = 30
    stuck_expected_rise_ppm: float = 1000
    tracking_time_s: float = 0  # zero weights the whole horizon equally
    delay_safety_factor: float = 2
    mixing_safety_factor: float = 2

    @property
    def prediction_step(self):
        return self.prediction_step_s or self.sample_s

    @property
    def planning_interval(self):
        return self.planning_interval_s or self.min_interval_s

    def __post_init__(self):
        for k, v in asdict(self).items():
            finite(v, k, strict=k not in ('deadband_ppm', 'pulse_cost',
                                        'prediction_step_s', 'planning_interval_s', 'tracking_time_s'))
        if self.recovery_samples < 2 or int(self.recovery_samples) != self.recovery_samples:
            raise ValueError('recovery_samples must be an integer >= 2')
        if not self.target_ppm < self.max_ppm-self.margin_ppm:
            raise ValueError('target must be below max_ppm minus margin_ppm')
        if not self.min_pulse_s <= self.max_pulse_s < self.min_interval_s:
            raise ValueError('require min_pulse <= max_pulse < min_interval')
        if self.gain_safety_factor < 1 or self.observer_gain > 1:
            raise ValueError('gain safety factor >= 1; observer_gain <= 1')
        if self.delay_safety_factor<1 or self.mixing_safety_factor<1:
            raise ValueError('delay/mixing safety factors must be >= 1')
        if self.stale_s < self.sample_s or self.horizon_s < self.min_interval_s:
            raise ValueError('stale/horizon shorter than sample/move interval')
        if self.stuck_window_s < self.sample_s or self.stuck_expected_rise_ppm <= self.stuck_tolerance_ppm:
            raise ValueError('stuck-reading check needs a full sample window and resolved expected rise')
        if self.prediction_step < self.sample_s or self.prediction_step > self.min_interval_s:
            raise ValueError('prediction step must be between sample and minimum pulse interval')
        if self.tracking_time_s and self.tracking_time_s < self.prediction_step:
            raise ValueError('tracking time must cover at least one prediction step')
        if self.planning_interval < self.min_interval_s or self.planning_interval > self.horizon_s:
            raise ValueError('planning interval must be between minimum interval and horizon')
        if self.horizon_s/self.prediction_step > 1000 or self.horizon_s/self.planning_interval > 60:
            raise ValueError('horizon exceeds bounded solver size (1000 samples / 60 moves)')
        if (self.max_pulse_s-self.min_pulse_s)/self.pulse_quantum_s > 40:
            raise ValueError('at most 41 nonzero pulse choices')

    def pulses(self):
        n = int((self.max_pulse_s-self.min_pulse_s)/self.pulse_quantum_s+1e-8)
        return [0.0] + [round(self.min_pulse_s+i*self.pulse_quantum_s, 8) for i in range(n+1)]


@dataclass(frozen=True)
class ResponseUncertainty:
    """Opt-in operating scenarios; never replace the fixed safety envelope."""
    gain_min_ppm_per_s: float
    gain_max_ppm_per_s: float
    kinetics_factor: float = 1.5
    risk_weight: float = 0.5
    learning_rate: float = 0.3
    min_response_ppm: float = 200
    max_relative_rmse: float = 0.25
    learning_mode: str = 'isolated'
    learning_window_s: float = 2700
    learning_interval_s: float = 300

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k != 'learning_mode':
                finite(v, k, strict=True)
        if self.learning_mode not in ('isolated', 'window'):
            raise ValueError('learning_mode must be isolated or window')
        if not self.learning_interval_s <= self.learning_window_s <= 21600:
            raise ValueError('learning interval <= window <= 21600 s required')
        if self.gain_min_ppm_per_s > self.gain_max_ppm_per_s:
            raise ValueError('uncertainty gains must be ordered')
        if self.kinetics_factor < 1:
            raise ValueError('kinetics_factor must be >= 1')
        if max(self.risk_weight, self.learning_rate, self.max_relative_rmse) > 1:
            raise ValueError('uncertainty weights and relative RMSE must be <= 1')


class PulseMPC:
    def __init__(self, model, settings, uncertainty=None):
        self.model, self.settings = model, settings
        self.uncertainty = uncertainty
        if settings.horizon_s < model.settling_s() + settings.planning_interval:
            raise ValueError('horizon must cover all delay/mixing paths plus one planned move')
        if settings.min_pulse_s <= model.valve_dead_s:
            raise ValueError('minimum pulse must exceed valve dead time')
        if uncertainty:
            if not uncertainty.gain_min_ppm_per_s <= model.gain_ppm_per_s <= uncertainty.gain_max_ppm_per_s:
                raise ValueError('nominal gain must lie inside uncertainty range')
            if uncertainty.gain_max_ppm_per_s > model.gain_ppm_per_s*settings.gain_safety_factor:
                raise ValueError('operating gain range exceeds fixed safety gain')
            if uncertainty.kinetics_factor > min(settings.delay_safety_factor, settings.mixing_safety_factor):
                raise ValueError('operating kinetics exceed safety factors')
            if settings.horizon_s < model.settling_s()*uncertainty.kinetics_factor+settings.planning_interval:
                raise ValueError('horizon must cover slow uncertainty scenario plus one move')
        self.doses = []  # (monotonic start, measured energized seconds)
        self.last_dose = -math.inf
        self.time = None
        self.residual = 0.0
        self.last_sample = None
        self.last_value = None
        self.last_plan = {}
        self._observations = deque(maxlen=10000)
        self.gain_estimate = model.gain_ppm_per_s
        self.gain_low = uncertainty.gain_min_ppm_per_s if uncertainty else self.gain_estimate
        self.gain_high = uncertainty.gain_max_ppm_per_s if uncertainty else self.gain_estimate
        self._dose_gains = []
        self._fit = None
        self.gain_updates = 0
        self.last_gain_fit = {'state': 'no_pulse'}

    def committed(self, start, actual_s):
        """Call only after a confirmed pulse; never record a rejected command."""
        finite(start, 'start'); finite(actual_s, 'actual pulse', strict=True)
        self.doses.append((start, actual_s))
        self.last_dose = start
        self._dose_gains.append(self.gain_estimate)
        if self.uncertainty:
            # A new pulse ends an incomplete identification window. Never fit
            # two unknown overlapping responses as a single pulse gain.
            self._fit = None
            self.last_gain_fit = {'state': 'waiting_for_response', 'pulse_index': len(self.doses)-1}
            if self.last_sample is not None and 0 <= start-self.last_sample <= self.settings.stale_s:
                self._fit = {'index': len(self.doses)-1, 'baseline_time': self.last_sample,
                             'baseline': self.last_value, 'points': deque(maxlen=2048), 'last_time': self.last_sample}
            else:
                self.last_gain_fit = {'state': 'rejected', 'reason': 'no fresh pre-pulse baseline'}

    def _injected(self, t):
        return sum((self._dose_gains[i] if self.uncertainty else self.model.gain_ppm_per_s)
                   *max(0,d-self.model.valve_dead_s)*self.model.response(t-start)
                   for i,(start, d) in enumerate(self.doses))

    def uncertainty_status(self):
        return {'enabled': self.uncertainty is not None,
                'nominal_gain_ppm_per_s': self.gain_estimate,
                'gain_min_ppm_per_s': self.gain_low, 'gain_max_ppm_per_s': self.gain_high,
                'safety_gain_ppm_per_s': self.model.gain_ppm_per_s*self.settings.gain_safety_factor,
                'accepted_updates': self.gain_updates, 'last_fit': dict(self.last_gain_fit)}

    def _learn_response(self, value, stamp):
        f, u, m = self._fit, self.uncertainty, self.model
        if f is None:
            return
        start, duration = self.doses[f['index']]
        if stamp <= start:
            return
        if stamp-f['last_time'] > self.settings.stale_s:
            self.last_gain_fit = {'state': 'rejected', 'reason': 'measurement gap in response'}
            self._fit = None
            return
        f['last_time'] = stamp
        decay = math.exp(-m.leak_per_s*(stamp-f['baseline_time']))
        baseline = m.ambient_ppm+(f['baseline']-m.ambient_ppm)*decay
        for i,(s,d) in enumerate(self.doses[:f['index']]):
            baseline += self._dose_gains[i]*max(0,d-m.valve_dead_s)*(m.response(stamp-s)-m.response(f['baseline_time']-s)*decay)
        x = max(0,duration-m.valve_dead_s)*m.response(stamp-start)
        f['points'].append((x,value-baseline))
        if stamp-start < m.settling_s()*u.kinetics_factor:
            return
        self._fit = None
        points = f['points']
        denom = sum(x*x for x,y in points)
        gain = sum(x*y for x,y in points)/denom if denom else 0
        rmse = math.sqrt(sum((y-gain*x)**2 for x,y in points)/len(points))
        response = gain*max(x for x,y in points)
        fit = {'state': 'rejected', 'gain_ppm_per_s': gain, 'rmse_ppm': rmse,
               'samples': len(points), 'pulse_index': f['index'], 'reason': 'unresolved or poor fit'}
        self.last_gain_fit = fit
        if len(points)<12 or response<u.min_response_ppm or rmse>u.max_relative_rmse*response:
            return
        upper = m.gain_ppm_per_s*self.settings.gain_safety_factor
        if gain > upper:
            fit['reason'] = 'estimated gain exceeds fixed safety gain'
            raise ValueError(fit['reason'])
        old_injected = self._injected(self.time)
        self._dose_gains[f['index']] = gain
        # Reparameterizing an issued dose must not jump the observer state.
        self.residual += old_injected-self._injected(self.time)
        self.gain_estimate = (1-u.learning_rate)*self.gain_estimate+u.learning_rate*gain
        # Evidence can widen the operating range, never erase earlier variation.
        self.gain_low = min(self.gain_low, gain*0.8)
        self.gain_high = min(upper, max(self.gain_high, gain*1.2))
        self.gain_updates += 1
        fit.update(state='accepted', reason='fixed-kinetics conditional estimate')

    def observe(self, value, measured_at, now):
        finite(now, 'now'); finite(value, 'CO2'); finite(measured_at, 'sample time')
        if measured_at > now or now-measured_at > self.settings.stale_s:
            raise ValueError('CO2 sample is stale or has a future timestamp')
        if value >= self.settings.max_ppm:
            raise ValueError('CO2 upper limit reached')
        if self.last_sample is not None and measured_at <= self.last_sample:
            return False
        self._learn_response(value, measured_at)
        self._observations.append((measured_at,value))
        cutoff=measured_at-self.settings.stuck_window_s
        while len(self._observations)>1 and self._observations[1][0]<=cutoff:
            self._observations.popleft()
        start,initial=self._observations[0]
        if measured_at-start>=self.settings.stuck_window_s:
            decay=math.exp(-self.model.leak_per_s*(measured_at-start))
            expected_rise=(initial-self.model.ambient_ppm)*(decay-1)
            expected_rise+=self._injected(measured_at)-self._injected(start)*decay
            values=[v for _,v in self._observations]
            if (expected_rise>=self.settings.stuck_expected_rise_ppm and
                    max(values)-min(values)<=self.settings.stuck_tolerance_ppm):
                raise ValueError('CO2 reading is not responding to the predicted injected gas')
        if self.time is None:
            self.residual = value-self.model.ambient_ppm-self._injected(measured_at)
        else:
            self.residual *= math.exp(-self.model.leak_per_s*(measured_at-self.time))
            prediction = self.model.ambient_ppm+self.residual+self._injected(measured_at)
            self.residual += self.settings.observer_gain*(value-prediction)
        self.time = self.last_sample = measured_at
        self.last_value = value
        return True

    def predict(self, t):
        return self.model.ambient_ppm + self.residual*math.exp(
            -self.model.leak_per_s*(t-self.time)) + self._injected(t)

    def propose(self, now):
        """Bounded coordinate search over a multi-move discrete pulse horizon.

        This is an approximate optimizer, not a globally optimal mixed-integer
        solver. Overshoot is weighted more heavily than undershoot. Every trial
        sequence is checked against the concentration ceiling with excess gain.
        """
        s, m = self.settings, self.model
        if self.time is None or now-self.last_sample > s.stale_s:
            raise ValueError('fresh CO2 observation required')
        if now-self.last_dose < s.min_interval_s:
            return 0.0
        # Bound ALL gas not yet seen at the last observation. Do not count
        # nominal leakage as permission to inject: it may disappear/change.
        # Full energized duration is conservative even if valve dead time is wrong.
        upper_gain=m.gain_ppm_per_s*s.gain_safety_factor
        unarrived=sum(upper_gain*d*m.remaining_fraction(
            self.last_sample-start,s.delay_safety_factor,s.mixing_safety_factor)
            for start,d in self.doses)
        budget_ceiling=s.max_ppm-s.margin_ppm
        immediate_choices=[p for p in s.pulses()
                           if p==0 or self.last_value+unarrived+upper_gain*p<budget_ceiling]
        budget={'pending_upper_ppm':unarrived,'dose_budget_ceiling_ppm':budget_ceiling,
                'dose_budget_blocked':len(immediate_choices)==1}
        if len(immediate_choices)==1:
            self.last_plan={**budget,'pulse_s':0.0,'planned_pulses_s':[],
                            'feasible':self.last_value+unarrived<budget_ceiling}
            return 0.0
        times = [now+i*s.prediction_step for i in range(1, math.ceil(s.horizon_s/s.prediction_step)+1)]
        # Keep the original fixed-gain safety forecast independent of learning.
        fixed_injected = lambda t: sum(m.dose_ppm(d)*m.response(t-start) for start,d in self.doses)
        injected = [fixed_injected(t) for t in times]
        injected_now = fixed_injected(now)
        fixed_residual = self.residual+self._injected(self.time)-fixed_injected(self.time)
        baseline = [m.ambient_ppm+fixed_residual*math.exp(-m.leak_per_s*(t-self.time))+v
                    for t,v in zip(times,injected)]
        # Preserve conservative uncertainty for gas already issued but not observed.
        pending = [max(0, v-injected_now*math.exp(-m.leak_per_s*(t-now)))
                   for t,v in zip(times,injected)]
        moves = [now+i*s.planning_interval for i in range(math.ceil(s.horizon_s/s.planning_interval))]
        kernels = [[m.response(t-start) for t in times] for start in moves]
        models = [m]
        baselines = [baseline]
        all_kernels = [kernels]
        if self.uncertainty:
            factor = self.uncertainty.kinetics_factor
            # Nine operating scenarios: three gains by three kinetic corners.
            # They are sensitivity scenarios, not a proof for every possible plant.
            for gain in (self.gain_low, self.gain_estimate, self.gain_high):
                for speed in (1/factor, 1.0, factor):
                    scenario = replace(m, gain_ppm_per_s=gain, delay_s=m.delay_s*speed,
                                       mixing_s=m.mixing_s*speed, slow_delay_s=m.slow_delay_s*speed,
                                       slow_mixing_s=m.slow_mixing_s*speed, leak_per_s=m.leak_per_s/speed)
                    models.append(scenario)
                    def injected_at(t):
                        return sum(scenario.dose_ppm(d)*scenario.response(t-start) for start,d in self.doses)
                    observed_input = injected_at(self.last_sample)
                    baselines.append([scenario.ambient_ppm+
                        (self.last_value-scenario.ambient_ppm-observed_input)*math.exp(-scenario.leak_per_s*(t-self.last_sample))+
                        injected_at(t) for t in times])
                    all_kernels.append([[scenario.response(t-start) for t in times] for start in moves])
        doses = [0.0]*len(moves)
        forecasts = [b[:] for b in baselines]
        scale = max(s.target_ppm, 1000)
        weights=[math.exp(-(t-now)/s.tracking_time_s) if s.tracking_time_s else 1 for t in times]
        total_weight=sum(weights)
        def score(predictions, effort):
            pred = predictions[0]
            if any(v+(s.gain_safety_factor-1)*(max(0,v-b)+p) >= s.max_ppm-s.margin_ppm
                   for v,b,p in zip(pred,baseline,pending)):
                return math.inf
            scenarios = predictions[1:] if self.uncertainty else predictions
            if any(v >= budget_ceiling for scenario in scenarios for v in scenario):
                return math.inf
            costs = [sum(w*max(0,abs(v-s.target_ppm)-s.deadband_ppm)**2*
                         (4 if v>s.target_ppm else 1)/scale**2 for v,w in zip(scenario,weights))/total_weight
                     for scenario in scenarios]
            risk = self.uncertainty.risk_weight if self.uncertainty else 0
            cost = (1-risk)*sum(costs)/len(costs)+risk*max(costs)
            return cost + s.pulse_cost*effort
        best = score(forecasts, 0)
        # Two coordinate sweeps optimize future moves as well as the immediate one.
        for _ in range(2):
            for j in range(len(moves)):
                old = doses[j]
                bases = [[v-model.dose_ppm(old)*h for v,h in zip(pred,ks[j])]
                         for model,pred,ks in zip(models,forecasts,all_kernels)]
                effort = sum(doses)-old
                chosen = old
                for pulse in (immediate_choices if j==0 else s.pulses()):
                    preds = [[v+model.dose_ppm(pulse)*h for v,h in zip(base,ks[j])]
                             for model,base,ks in zip(models,bases,all_kernels)]
                    cost = score(preds, effort+pulse)
                    if cost < best-1e-12:
                        best, chosen, forecasts = cost, pulse, preds
                doses[j] = chosen
        forecast = forecasts[5] if self.uncertainty else forecasts[0]  # center scenario
        self.last_plan = {**budget,'predicted_peak_ppm': max(forecast),
                          'predicted_end_ppm': forecast[-1], 'pulse_s': doses[0],
                          'planned_pulses_s': doses, 'feasible': math.isfinite(best)}
        if self.uncertainty:
            self.last_plan.update(scenario_peak_min_ppm=min(max(p) for p in forecasts[1:]),
                                  scenario_peak_max_ppm=max(max(p) for p in forecasts[1:]),
                                  scenario_count=len(forecasts)-1)
        return doses[0] if math.isfinite(best) else 0.0


def make_controller(model, settings, uncertainty=None):
    """Keep existing linear profiles unchanged; nonlinear/window mode is opt-in."""
    if model.loss_exponent != 1 or (uncertainty and uncertainty.learning_mode == 'window'):
        from .co2_nonlinear import NonlinearPulseMPC
        return NonlinearPulseMPC(model, settings, uncertainty)
    return PulseMPC(model, settings, uncertainty)
