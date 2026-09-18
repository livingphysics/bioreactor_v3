"""Bounded integral correction of persistent, time-averaged tracking error.

Adjusts only the optimizer's internal tracking target, never a measurement,
physical-model parameter, user setpoint, or concentration/dose safety limit.
"""
from collections import deque
from dataclasses import dataclass, asdict
from .co2_mpc import finite


@dataclass(frozen=True)
class AverageCorrectionSettings:
    window_s: float = 1800
    integral_time_s: float = 3600
    max_offset_ppm: float = 5000
    activation_band_ppm: float = 5000
    deadband_ppm: float = 100
    max_rate_ppm_per_s: float = 2

    def __post_init__(self):
        for name, value in asdict(self).items():
            finite(value, name, strict=name != 'deadband_ppm')
        if not 60 <= self.window_s <= 21600:
            raise ValueError('average correction window must be 60..21600 seconds')
        if self.integral_time_s < self.window_s:
            raise ValueError('average integral time must be at least one window')
        if self.deadband_ppm >= self.activation_band_ppm:
            raise ValueError('average deadband must be smaller than activation band')


class AverageErrorCorrection:
    def __init__(self, config, settings):
        if config.window_s < 12*settings.sample_s or config.window_s/settings.sample_s > 10000:
            raise ValueError('average window needs 12..10000 samples')
        self.config = config
        self.points = deque(maxlen=10000)
        self.setpoint = None
        self.offset = 0.0
        self.mean = None
        self.state = 'collecting'
        self.sync(settings)

    def sync(self, settings):
        if self.setpoint != settings.target_ppm:
            self.setpoint = settings.target_ppm
            self.offset = 0.0
            self.points.clear()
            self.mean = None
            self.state = 'collecting'
        # Bound both absolute shift and its fraction of the requested target.
        bound = min(self.config.max_offset_ppm, self.setpoint*.1)
        self.lower = max(1, self.setpoint-bound)
        self.upper = min(self.setpoint+bound, settings.max_ppm-settings.margin_ppm-1)
        self.offset = min(self.upper-self.setpoint, max(self.lower-self.setpoint, self.offset))

    def observe(self, value, stamp, settings, *, constrained=False):
        self.sync(settings)
        if self.points and stamp <= self.points[-1][0]:
            return
        band = self.config.activation_band_ppm
        if abs(value-self.setpoint) > band:
            # Do not integrate startup ramps or large disturbances/capacity limits.
            self.points.clear(); self.mean = None; self.state = 'outside_activation_band'
            return
        previous = self.points[-1][0] if self.points else None
        if previous is not None and stamp-previous > settings.stale_s:
            self.points.clear(); previous = None
        self.points.append((stamp,value))
        cutoff = stamp-self.config.window_s
        while len(self.points)>1 and self.points[1][0] <= cutoff:
            self.points.popleft()
        if len(self.points)<12 or self.points[0][0]>cutoff:
            self.mean = None; self.state = 'collecting'
            return
        points = list(self.points)
        if points[0][0] < cutoff:
            t0,v0 = points[0]; t1,v1 = points[1]
            points[0] = (cutoff,v0+(v1-v0)*(cutoff-t0)/(t1-t0))
        self.mean = sum((v0+v1)/2*(t1-t0) for (t0,v0),(t1,v1) in zip(points,points[1:]))/self.config.window_s
        error = self.setpoint-self.mean
        if abs(error) <= self.config.deadband_ppm:
            self.state = 'within_deadband'
            return
        if error > 0 and constrained:
            self.state = 'limited_by_dosing_guard'
            return
        dt = stamp-previous if previous is not None else 0
        rate = max(-self.config.max_rate_ppm_per_s,
                   min(self.config.max_rate_ppm_per_s,error/self.config.integral_time_s))
        self.offset += rate*dt
        unclamped = self.offset
        self.sync(settings)  # clamping is anti-windup; no hidden integral builds up
        self.state = 'offset_limit' if self.offset != unclamped else 'correcting'

    def target(self, settings):
        self.sync(settings)
        return self.setpoint+self.offset

    def status(self, settings):
        target = self.target(settings)
        return {'enabled':True, 'state':self.state, 'window_s':self.config.window_s,
                'mean_ppm':self.mean, 'mean_error_ppm':self.mean-self.setpoint if self.mean is not None else None,
                'offset_ppm':self.offset, 'tracking_target_ppm':target,
                'requested_target_ppm':self.setpoint,
                'collected_s':self.points[-1][0]-self.points[0][0] if self.points else 0}
