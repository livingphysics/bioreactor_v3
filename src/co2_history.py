"""Durable CO2 dose state. Monotonic timestamps are valid only within one Pi boot.

A pending marker is fsynced BEFORE opening the valve. Completed doses are saved
only AFTER closure. Corrupt/missing/incompatible state or an interrupted pulse
requires a settling interval from confirmed closure, never an assumed zero dose.
"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .co2_mpc import GasModel, MPCSettings, finite


def atomic_json(path, state):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as f:
            temporary = f.name
            json.dump(state, f, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


def invalidate_unconfigured(path):
    """An API without a model may issue manual doses; never reuse its old state."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path)+'.lock', 'a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(path, {'invalid': 'CO2 model not configured'})


def hardware_identity(config):
    return {name: getattr(config, name, None) for name in (
        'RELAYS', 'CO2_SENSOR_TYPE', 'CO2_SENSOR_I2C_BUS', 'CO2_SENSOR_I2C_ADDRESS')}


class DoseHistory:
    def __init__(self, path, profile, *, clock=time.monotonic, identity=None, boot_id=None):
        self.path = Path(path).expanduser()
        self.clock = clock
        self.model = GasModel(**profile['model'])
        self.settings = MPCSettings(**profile.get('settings', {}))
        fingerprint = {'model': asdict(self.model), 'settings': asdict(self.settings),
                       'uncertainty': profile.get('uncertainty'), 'hardware': identity}
        fingerprint['settings'].pop('target_ppm')
        self.fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        if boot_id is None:
            try: boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            except OSError: boot_id = None
        self.boot_id = boot_id
        self.error = None
        self.reason = 'not loaded'
        self.state = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(str(self.path)+'.lock', 'a')
        try: fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise RuntimeError('CO2 dose history is already owned by another process')

    def close(self):
        self._lock.close()

    def _empty(self):
        return {'version': 1, 'boot_id': self.boot_id, 'fingerprint': self.fingerprint,
                'saved_at': self.clock(), 'wait_until': 0.0, 'pending': None,
                'doses': [], 'gain': None}

    def _check(self, state):
        now = self.clock()
        if state['version'] != 1 or not self.boot_id or state['boot_id'] != self.boot_id:
            raise ValueError('different boot or unsupported history')
        if state['fingerprint'] != self.fingerprint:
            raise ValueError('model, limits or hardware configuration changed')
        finite(state['saved_at'], 'saved time')
        finite(state['wait_until'], 'wait deadline')
        if state['saved_at'] > now or state['wait_until'] > state['saved_at']+self.model.settling_s(5)+1:
            raise ValueError('history clock is inconsistent')
        if state['pending'] not in (None, 'control', 'external'):
            raise ValueError('invalid pending marker')
        if not isinstance(state['doses'], list) or len(state['doses']) > 10000:
            raise ValueError('invalid dose list')
        upper = self.model.gain_ppm_per_s*self.settings.gain_safety_factor
        previous = -math.inf
        for dose in state['doses']:
            if len(dose) != 3: raise ValueError('invalid dose')
            start, duration, gain = dose
            finite(start, 'dose time'); finite(duration, 'duration', strict=True)
            finite(gain, 'gain', strict=True)
            if not previous <= start <= state['saved_at'] or start+duration > state['saved_at']+1e-6 or gain > upper:
                raise ValueError('invalid dose timing or gain')
            # Completed pulses may exceed their command by the worker tolerance.
            if duration > self.settings.max_pulse_s + max(.1, self.settings.max_pulse_s*.25):
                raise ValueError('history contains an overrun')
            previous = start
        gain = state['gain']
        if gain is not None:
            for name in ('estimate', 'low', 'high'): finite(gain[name], name, strict=True)
            if not gain['low'] <= gain['estimate'] <= gain['high'] <= upper:
                raise ValueError('invalid learned gain bounds')
        return state

    def load_after_closure(self):
        """Caller must have successfully commanded the valve OFF first."""
        try:
            if self.path.stat().st_size > 2_000_000: raise ValueError('history file too large')
            state = self._check(json.loads(self.path.read_text()))
            if state['pending']: raise ValueError('interrupted or untracked valve operation')
            self.state, self.reason = state, 'restored'
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            self.state = self._empty()
            self.quarantine('history missing, incompatible or uncertain')

    def _write(self):
        if self._lock.closed: raise RuntimeError('CO2 dose history has been closed')
        if self.error: raise RuntimeError(self.error)
        self.state['saved_at'] = self.clock()
        try:
            atomic_json(self.path, self.state)
        except Exception as exc:
            self.error = f'CO2 dose history could not be saved: {exc}'
            raise RuntimeError(self.error) from exc

    def quarantine(self, reason):
        self.state = self._empty()
        self.state['wait_until'] = self.clock()+self.model.settling_s(5)
        self.reason = reason
        self._write()

    def wait_s(self):
        if self.state['pending']: return self.model.settling_s(5)
        return max(0.0, self.state['wait_until']-self.clock())

    def assert_ready(self):
        if self._lock.closed: raise RuntimeError('CO2 dose history has been closed')
        if self.error: raise RuntimeError(self.error)
        if self.state['pending'] or self.wait_s():
            raise RuntimeError('wait for prior injected gas to settle before restarting CO2 control')

    def _snapshot(self, engine):
        if engine is None: return
        retention = self.model.settling_s(32)*max(self.settings.delay_safety_factor,
                                                self.settings.mixing_safety_factor)
        retention += max(21600, self.settings.min_interval_s)
        cutoff = self.clock()-retention
        gains = engine._dose_gains
        self.state['doses'] = [[t,d,gains[i] if gains else engine.gain_estimate]
                              for i,(t,d) in enumerate(engine.doses) if t >= cutoff]
        if len(self.state['doses']) > 10000:
            raise RuntimeError('too many CO2 doses to persist')
        self.state['gain'] = {'estimate':engine.gain_estimate,
                              'low':engine.gain_low, 'high':engine.gain_high}

    def begin(self, engine=None, *, external=False):
        if not external: self.assert_ready()
        self._snapshot(engine)
        self.state['pending'] = 'external' if external else 'control'
        self._write()  # durable before any attempt to energize

    def complete(self, engine):
        self._snapshot(engine)
        self.state['pending'] = None
        self._write()

    def checkpoint(self, engine):
        if self.state['pending']: return
        self._snapshot(engine)
        self._write()

    def restore(self, engine):
        self.assert_ready()
        gain = self.state['gain']
        if gain:
            engine.gain_estimate, engine.gain_low, engine.gain_high = gain['estimate'],gain['low'],gain['high']
        for start, duration, gain in self.state['doses']:
            engine.committed(start, duration)
            if engine._dose_gains: engine._dose_gains[-1] = gain
        engine._fit = None  # old fit observations are not restored
        engine.last_gain_fit = {'state':'restored_history','reason':'waiting for new measurements'}

    def status(self):
        return {'enabled': True, 'reason': self.reason, 'error': self.error,
                'pending': self.state['pending'], 'stored_doses': len(self.state['doses']),
                'restart_wait_s': self.wait_s()}
