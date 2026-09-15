"""Discrete-pulse, receding-horizon CO2 control; no hardware dependencies.

Model: a pulse arrives after delay_s, mixes with time constant mixing_s, then
leaks toward ambient_ppm at leak_per_s. Gain is ppm per effective valve second.
State includes all issued pulses, including gas still in transit. Only the first
move of the optimized pulse sequence is executed; fresh data replans the rest.
"""
from dataclasses import dataclass, asdict
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

    def __post_init__(self):
        for k, v in asdict(self).items():
            finite(v, k, strict=k in ('gain_ppm_per_s', 'mixing_s', 'slow_mixing_s'))
        if self.slow_fraction > 1:
            raise ValueError('slow_fraction must be between zero and one')

    def settling_s(self, constants=4):
        """Cover every enabled transport/mixing path, including delayed gas."""
        fast = self.delay_s + constants*self.mixing_s if self.slow_fraction < 1 else 0
        slow = self.slow_delay_s + constants*self.slow_mixing_s if self.slow_fraction else 0
        return max(fast, slow)

    def response(self, age_s):
        """Concentration per ppm injected, including transport/mixing/leakage."""
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
        if not self.target_ppm < self.max_ppm-self.margin_ppm:
            raise ValueError('target must be below max_ppm minus margin_ppm')
        if not self.min_pulse_s <= self.max_pulse_s < self.min_interval_s:
            raise ValueError('require min_pulse <= max_pulse < min_interval')
        if self.gain_safety_factor < 1 or self.observer_gain > 1:
            raise ValueError('gain safety factor >= 1; observer_gain <= 1')
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


class PulseMPC:
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        if settings.horizon_s < model.settling_s() + settings.planning_interval:
            raise ValueError('horizon must cover all delay/mixing paths plus one planned move')
        if settings.min_pulse_s <= model.valve_dead_s:
            raise ValueError('minimum pulse must exceed valve dead time')
        self.doses = []  # (monotonic start, measured energized seconds)
        self.last_dose = -math.inf
        self.time = None
        self.residual = 0.0
        self.last_sample = None
        self.last_plan = {}
        self._observations = deque(maxlen=10000)

    def committed(self, start, actual_s):
        """Call only after a confirmed pulse; never record a rejected command."""
        finite(start, 'start'); finite(actual_s, 'actual pulse', strict=True)
        self.doses.append((start, actual_s))
        self.last_dose = start

    def _injected(self, t):
        return sum(self.model.dose_ppm(d)*self.model.response(t-start)
                   for start, d in self.doses)

    def observe(self, value, measured_at, now):
        finite(now, 'now'); finite(value, 'CO2'); finite(measured_at, 'sample time')
        if measured_at > now or now-measured_at > self.settings.stale_s:
            raise ValueError('CO2 sample is stale or has a future timestamp')
        if value >= self.settings.max_ppm:
            raise ValueError('CO2 upper limit reached')
        if self.last_sample is not None and measured_at <= self.last_sample:
            return False
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
        times = [now+i*s.prediction_step for i in range(1, math.ceil(s.horizon_s/s.prediction_step)+1)]
        injected = [self._injected(t) for t in times]
        injected_now = self._injected(now)
        baseline = [m.ambient_ppm+self.residual*math.exp(-m.leak_per_s*(t-self.time))+v
                    for t,v in zip(times,injected)]
        # Preserve conservative uncertainty for gas already issued but not observed.
        pending = [max(0, v-injected_now*math.exp(-m.leak_per_s*(t-now)))
                   for t,v in zip(times,injected)]
        moves = [now+i*s.planning_interval for i in range(math.ceil(s.horizon_s/s.planning_interval))]
        kernels = [[m.response(t-start) for t in times] for start in moves]
        doses = [0.0]*len(moves)
        forecast = baseline[:]
        scale = max(s.target_ppm, 1000)
        weights=[math.exp(-(t-now)/s.tracking_time_s) if s.tracking_time_s else 1 for t in times]
        total_weight=sum(weights)
        def score(pred, effort):
            if any(v+(s.gain_safety_factor-1)*(max(0,v-b)+p) >= s.max_ppm-s.margin_ppm
                   for v,b,p in zip(pred,baseline,pending)):
                return math.inf
            errors = [(abs(v-s.target_ppm)-s.deadband_ppm) for v in pred]
            cost = sum(w*max(0,e)**2*(4 if v>s.target_ppm else 1)/scale**2
                       for e,v,w in zip(errors,pred,weights))/total_weight
            return cost + s.pulse_cost*effort
        best = score(forecast, 0)
        # Two coordinate sweeps optimize future moves as well as the immediate one.
        for _ in range(2):
            for j, kernel in enumerate(kernels):
                old = doses[j]
                base = [v-m.dose_ppm(old)*h for v,h in zip(forecast,kernel)]
                effort = sum(doses)-old
                chosen = old
                for pulse in s.pulses():
                    pred = [v+m.dose_ppm(pulse)*h for v,h in zip(base,kernel)]
                    cost = score(pred, effort+pulse)
                    if cost < best-1e-12:
                        best, chosen, forecast = cost, pulse, pred
                doses[j] = chosen
        self.last_plan = {'predicted_peak_ppm': max(forecast),
                          'predicted_end_ppm': forecast[-1], 'pulse_s': doses[0],
                          'planned_pulses_s': doses, 'feasible': math.isfinite(best)}
        return doses[0] if math.isfinite(best) else 0.0
