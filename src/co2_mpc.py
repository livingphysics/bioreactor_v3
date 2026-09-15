"""Discrete-pulse, receding-horizon CO2 control; no hardware dependencies.

Model: a pulse arrives after delay_s, mixes with time constant mixing_s, then
leaks toward ambient_ppm at leak_per_s. Gain is ppm per effective valve second.
State includes all issued pulses, including gas still in transit. Only the first
move of the optimized pulse sequence is executed; fresh data replans the rest.
"""
from dataclasses import dataclass, asdict
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

    def __post_init__(self):
        for k, v in asdict(self).items():
            finite(v, k, strict=k in ('gain_ppm_per_s', 'mixing_s'))

    def response(self, age_s):
        """Concentration per ppm injected, including transport/mixing/leakage."""
        t = age_s - self.delay_s
        if t <= 0:
            return 0.0
        k, tau = self.leak_per_s, self.mixing_s
        if abs(1-k*tau) < 1e-7:
            return t/tau * math.exp(-t/tau)
        return (math.exp(-k*t)-math.exp(-t/tau))/(1-k*tau)

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

    def __post_init__(self):
        for k, v in asdict(self).items():
            finite(v, k, strict=k not in ('deadband_ppm', 'pulse_cost'))
        if not self.target_ppm < self.max_ppm-self.margin_ppm:
            raise ValueError('target must be below max_ppm minus margin_ppm')
        if not self.min_pulse_s <= self.max_pulse_s < self.min_interval_s:
            raise ValueError('require min_pulse <= max_pulse < min_interval')
        if self.gain_safety_factor < 1 or self.observer_gain > 1:
            raise ValueError('gain safety factor >= 1; observer_gain <= 1')
        if self.stale_s < self.sample_s or self.horizon_s < self.min_interval_s:
            raise ValueError('stale/horizon shorter than sample/move interval')
        if self.horizon_s/self.sample_s > 1000 or self.horizon_s/self.min_interval_s > 60:
            raise ValueError('horizon exceeds bounded solver size (1000 samples / 60 moves)')
        if (self.max_pulse_s-self.min_pulse_s)/self.pulse_quantum_s > 40:
            raise ValueError('at most 41 nonzero pulse choices')

    def pulses(self):
        n = int((self.max_pulse_s-self.min_pulse_s)/self.pulse_quantum_s+1e-8)
        return [0.0] + [round(self.min_pulse_s+i*self.pulse_quantum_s, 8) for i in range(n+1)]


class PulseMPC:
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        if settings.horizon_s < model.delay_s + 4*model.mixing_s + settings.min_interval_s:
            raise ValueError('horizon must cover delay + four mixing constants + one move')
        if settings.min_pulse_s <= model.valve_dead_s:
            raise ValueError('minimum pulse must exceed valve dead time')
        self.doses = []  # (monotonic start, measured energized seconds)
        self.last_dose = -math.inf
        self.time = None
        self.residual = 0.0
        self.last_sample = None
        self.last_plan = {}

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
        times = [now+i*s.sample_s for i in range(1, int(s.horizon_s/s.sample_s)+1)]
        baseline = [self.predict(t) for t in times]
        # Preserve conservative uncertainty for gas already issued but not observed.
        pending = [max(0, self._injected(t)-self._injected(now)*math.exp(-m.leak_per_s*(t-now)))
                   for t in times]
        moves = [now+i*s.min_interval_s for i in range(int(s.horizon_s/s.min_interval_s))]
        kernels = [[m.response(t-start) for t in times] for start in moves]
        doses = [0.0]*len(moves)
        forecast = baseline[:]
        scale = max(s.target_ppm, 1000)
        def score(pred, effort):
            if any(v+(s.gain_safety_factor-1)*(max(0,v-b)+p) >= s.max_ppm-s.margin_ppm
                   for v,b,p in zip(pred,baseline,pending)):
                return math.inf
            errors = [(abs(v-s.target_ppm)-s.deadband_ppm) for v in pred]
            cost = sum(max(0,e)**2*(4 if v>s.target_ppm else 1)/scale**2
                       for e,v in zip(errors,pred))/len(pred)
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
