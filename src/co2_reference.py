"""Immutable, bounded reference trajectories on the controller's monotonic clock."""
from bisect import bisect_left
from dataclasses import dataclass
import math


class ReferenceUnavailable(ValueError):
    """Missing, stale or discontinuous reference: inhibit dosing, retain gas history."""


def number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True)
class ReferenceTrajectory:
    points: tuple
    max_gap_s: float = 30
    hold_s: float = 0

    def __post_init__(self):
        points = tuple(tuple(p) for p in self.points)
        if not points or len(points) > 100000:
            raise ValueError('reference needs 1..100000 points')
        if not number(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError('reference maximum gap must be positive')
        if not number(self.hold_s) or not 0 <= self.hold_s <= self.max_gap_s:
            raise ValueError('reference endpoint hold must be within maximum gap')
        previous = -math.inf
        for point in points:
            if len(point) != 2:
                raise ValueError('reference points must contain time and ppm')
            stamp, value = point
            if not number(stamp) or not number(value) or value <= 0 or stamp <= previous:
                raise ValueError('reference points must be finite, positive and strictly ordered')
            previous = stamp
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, '_times', tuple(p[0] for p in points))

    def value_at(self, stamp):
        if not number(stamp):
            raise ReferenceUnavailable('invalid reference time')
        index = bisect_left(self._times, stamp)
        if index < len(self.points) and self.points[index][0] == stamp:
            return self.points[index][1]
        if index == 0:
            raise ReferenceUnavailable('reference history has not filled the delay')
        if index == len(self.points):
            if stamp - self.points[-1][0] <= self.hold_s:
                return self.points[-1][1]
            raise ReferenceUnavailable('reference preview is incomplete')
        t0, v0 = self.points[index-1]
        t1, v1 = self.points[index]
        if t1-t0 > self.max_gap_s:
            raise ReferenceUnavailable('gap in reference history')
        return v0+(v1-v0)*(stamp-t0)/(t1-t0)

    def require_span(self, start, end):
        self.value_at(start)
        self.value_at(end)
        first, last = bisect_left(self._times, start), bisect_left(self._times, end)
        for index in range(max(1, first), min(len(self.points), last+1)):
            if self.points[index][0]-self.points[index-1][0] > self.max_gap_s:
                raise ReferenceUnavailable('gap in reference preview')


def tracking_metrics(rows):
    """Time-weighted errors against the declared reference, without fitted time shifts.

    Invalid rows break coverage; gaps over 60 seconds are not interpolated.
    Each row: time_s, reference_ppm, follower_ppm. Positive error is overshoot.
    """
    duration = area = square = absolute = rising_area = falling_area = 0.0
    rising_s = falling_s = 0.0
    errors = []
    previous = None
    for row in rows:
        t, ref, value = (row.get(k) for k in ('time_s', 'reference_ppm', 'follower_ppm'))
        if not all(number(v) for v in (t, ref, value)):
            previous = None
            continue
        error = value-ref
        errors.append(abs(error))
        if previous:
            pt, pref, pe = previous
            dt = t-pt
            if 0 < dt <= 60:
                duration += dt
                area += dt*(pe+error)/2
                square += dt*(pe*pe+pe*error+error*error)/3
                # Exact absolute integral of the linearly interpolated error.
                a = dt*(abs(pe)+abs(error))/2 if pe*error >= 0 else dt*(pe*pe+error*error)/(2*(abs(pe)+abs(error)))
                absolute += a
                if ref > pref:
                    rising_s += dt; rising_area += a
                elif ref < pref:
                    falling_s += dt; falling_area += a
        previous = t, ref, error
    errors.sort()
    return {'covered_s': duration, 'samples': len(errors),
            'bias_ppm': area/duration if duration else None,
            'mae_ppm': absolute/duration if duration else None,
            'rmse_ppm': math.sqrt(square/duration) if duration else None,
            'p95_abs_error_ppm': errors[max(0, math.ceil(.95*len(errors))-1)] if errors else None,
            'max_abs_error_ppm': max(errors) if errors else None,
            'rising_mae_ppm': rising_area/rising_s if rising_s else None,
            'falling_mae_ppm': falling_area/falling_s if falling_s else None}
