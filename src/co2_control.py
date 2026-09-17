"""Shared CO2 worker for API and standalone use; injected sensor/valve adapters.

read_sample returns (ppm, monotonic acquisition time). set_valve(True) energizes
the CO2 relay and must raise on a failed write. It must not wait on an I2C lock:
GPIO closure must remain independent of slow sensor traffic.
"""
from dataclasses import asdict, replace
import json
import math
import threading
import time
from .co2_mpc import GasModel, MPCSettings, PulseMPC, ResponseUncertainty, finite


class CO2Control:
    def __init__(self, read_sample, set_valve, profile=None, *, clock=time.monotonic,
                 log=None):
        self.read_sample = read_sample
        self.set_valve = set_valve
        self.profile = profile
        self.clock = clock
        self.log = log
        self.lock = threading.RLock()  # also used by manual relay writers
        self._stop = threading.Event()
        self._thread = None
        self.engine = None
        self.active = False
        self.fault = None
        self.owner = None
        self.last = {}
        self._last_stop = None
        self.deadline = None
        self._last_valid_at = None
        self._recovering = False
        self._recovery_count = 0

    def validate(self, target):
        if not isinstance(self.profile, dict):
            raise ValueError('CO2_MPC needs a validated rig-specific model before control')
        if self.profile.get('validated') is not True:
            trial = self.profile.get('trial', {})
            if not isinstance(trial, dict) or trial.get('enabled') is not True:
                raise ValueError('CO2_MPC needs a validated model or an explicitly enabled bounded trial')
            finite(trial.get('target_max_ppm'), 'trial target maximum', strict=True)
            finite(trial.get('max_duration_s'), 'trial maximum duration', strict=True)
            finite(target, 'target', strict=True)
            if target > trial['target_max_ppm']:
                raise ValueError('CO2 target exceeds the configured trial maximum')
        model = GasModel(**self.profile['model'])
        settings = MPCSettings(**{**self.profile.get('settings', {}), 'target_ppm': target})
        PulseMPC(model, settings, self._uncertainty())  # validate model/horizon together
        return model, settings

    def _uncertainty(self):
        config = (self.profile or {}).get('uncertainty')
        return ResponseUncertainty(**config) if config is not None else None

    def allows_indefinite(self):
        profile = self.profile or {}
        trial = profile.get('trial', {})
        return (profile.get('validated') is True or
                (isinstance(trial, dict) and trial.get('enabled') is True
                 and trial.get('allow_indefinite') is True))

    def validate_duration(self, duration_s):
        # Zero is the public indefinite sentinel; None is used internally/programs.
        if duration_s is not None:
            duration_s = finite(duration_s, 'duration')
            if duration_s == 0:
                duration_s = None
        if self.profile.get('validated') is not True:
            if duration_s is None:
                if not self.allows_indefinite():
                    raise ValueError('Indefinite CO2 control requires trial.allow_indefinite=true')
            elif duration_s > self.profile['trial']['max_duration_s']:
                raise ValueError('CO2 trial duration exceeds its configured maximum')
        return duration_s

    def start(self, target, owner='api', duration_s=None):
        model, settings = self.validate(target)
        duration_s = self.validate_duration(duration_s)
        trial_mode = self.profile.get('validated') is not True
        with self.lock:
            if self._thread and self._thread.is_alive() and not self.active:
                raise RuntimeError('CO2 worker is still stopping')
            if self.active:
                if owner != self.owner:
                    raise RuntimeError(f'CO2 valve is owned by {self.owner}')
                # Setpoint changes preserve the observer and in-flight gas history.
                self.engine.settings = replace(self.engine.settings, target_ppm=target)
                deadline = self.clock()+duration_s if duration_s else None
                # Timed trial updates cannot extend a timed deadline. An explicitly
                # permitted indefinite request removes it; adding a timer bounds it.
                self.deadline = (min(self.deadline, deadline)
                                 if trial_mode and self.deadline is not None and deadline is not None
                                 else deadline)
                return
            # A restart loses unobserved gas state. Refuse until it has mixed.
            if self._last_stop is not None and self.clock()-self._last_stop < model.settling_s(5):
                raise RuntimeError('wait for prior injected gas to settle before restarting CO2 control')
            value, measured_at = self.read_sample()
            engine = PulseMPC(model, settings, self._uncertainty())
            engine.observe(value, measured_at, self.clock())
            self.set_valve(False)
            self.engine, self.owner = engine, owner
            self._last_valid_at = measured_at
            self._recovering = False
            self._recovery_count = 0
            self.last = {}
            self.active, self.fault = True, None
            self.deadline = self.clock()+duration_s if duration_s else None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='co2-mpc', daemon=True)
            self._thread.start()

    def stop(self, owner=None, *, join=True):
        if owner is None:
            self._stop.set()  # interrupt wait/optimization before waiting for its lock
        with self.lock:
            if owner is not None and self.owner != owner:
                return
            was_active = self.active
            self.active = False
            self._stop.set()
            if was_active:
                self._last_stop = self.clock()
            self.set_valve(False)
            thread = self._thread
        if join and thread and thread is not threading.current_thread():
            thread.join(timeout=3)

    def status(self):
        with self.lock:
            validated = bool(self.profile and self.profile.get('validated') is True)
            trial = (self.profile or {}).get('trial', {})
            trial_mode = not validated and isinstance(trial, dict) and trial.get('enabled') is True
            restart_wait = 0.0
            if self._last_stop is not None and self.profile:
                try:
                    restart_wait = max(0.0, GasModel(**self.profile['model']).settling_s(5)
                                       - (self.clock()-self._last_stop))
                except (KeyError, TypeError, ValueError):
                    pass
            return {'active': self.active, 'owner': self.owner, 'fault': self.fault,
                    'configured': validated or trial_mode,
                    'indefinite': self.active and self.deadline is None,
                    'model_validated': validated, 'trial_mode': trial_mode,
                    'remaining_s': max(0.0, self.deadline-self.clock()) if self.active and self.deadline else None,
                    'restart_wait_s': restart_wait,
                    'measurement_paused': self.active and self._recovering,
                    'recovery_samples': self._recovery_count,
                    'last_valid_age_s': (max(0.0, self.clock()-self._last_valid_at)
                                         if self._last_valid_at is not None else None),
                    'target_ppm': self.engine.settings.target_ppm if self.engine else None,
                    'response_uncertainty': self.engine.uncertainty_status() if self.engine else None,
                    'last': dict(self.last)}

    def _plan_sample(self):
        """Pause on brief read gaps; never reset the observer or dose history."""
        settings = self.engine.settings
        now = self.clock()
        if now-self._last_valid_at >= settings.missing_timeout_s:
            raise ValueError('CO2 measurement outage exceeded missing_timeout_s')
        try:
            value, acquired = self.read_sample()
        except OSError:
            value, acquired = None, None
        now = self.clock()
        def numeric(v):
            return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
        # Concentration limits remain immediate, including during recovery.
        if numeric(value) and value >= settings.max_ppm:
            raise ValueError('CO2 upper limit reached')
        valid = (numeric(value) and value > 0 and numeric(acquired)
                 and 0 <= acquired <= now and now-acquired <= settings.stale_s)
        if now-self._last_valid_at >= settings.missing_timeout_s:
            raise ValueError('CO2 measurement outage exceeded missing_timeout_s')
        if not valid:
            self._recovering = True
            self._recovery_count = 0
            self.set_valve(False)
            self.last = {'co2_ppm': None, 'sample_age_s': None,
                         'measurement_state': 'waiting_for_fresh_readings',
                         'commanded_pulse_s': 0.0}
            return 0.0
        fresh = self.engine.observe(value, acquired, now)
        if fresh:
            self._last_valid_at = acquired
            if self._recovering:
                self._recovery_count += 1
                if self._recovery_count >= settings.recovery_samples:
                    self._recovering = False
        pulse = self.engine.propose(now) if fresh and not self._recovering else 0.0
        self.last = {'co2_ppm': value, 'sample_age_s': now-acquired,
                     **(self.engine.last_plan if not self._recovering else {}),
                     'measurement_state': 'recovering' if self._recovering else 'ready',
                     'commanded_pulse_s': pulse}
        return pulse

    def _run(self):
        try:
            while not self._stop.is_set():
                with self.lock:
                    if not self.active or (self.deadline and self.clock() >= self.deadline):
                        break
                    pulse = self._plan_sample()
                    acquired = self.engine.last_sample
                    if pulse:
                        if self._stop.is_set():
                            break
                        if self.clock()-acquired > self.engine.settings.stale_s:
                            raise ValueError('CO2 sample became stale during optimization')
                        if self.deadline and self.clock()+pulse >= self.deadline:
                            break
                        # No solver or sensor reads occur while the valve is energized.
                        start = self.clock()
                        self.set_valve(True)
                if pulse:
                    try:
                        self._stop.wait(pulse)
                    finally:
                        with self.lock:
                            self.set_valve(False)
                            actual = self.clock()-start
                            self.engine.committed(start, actual)
                            self.last['actual_pulse_s'] = actual
                            if actual > pulse + max(0.1, pulse*0.25):
                                raise RuntimeError('valve pulse overran timing tolerance')
                if self.log:
                    self.log({'time': time.time(), **self.status()})
                self._stop.wait(self.engine.settings.sample_s)
        except Exception as e:
            with self.lock:
                self.fault = str(e)
        finally:
            with self.lock:
                self.active = False
                self._last_stop = self.clock()
                try:
                    self.set_valve(False)
                except Exception as e:
                    self.fault = f'CO2 valve closure failed: {e}'


def run_standalone(config, target, duration_s, log_path):
    """Run just CO2 sensor + relays. Does not initialize pumps or temperature PID."""
    import signal
    from .bioreactor import Bioreactor
    from . import io
    finite(duration_s, 'duration')
    class GasOnly(type(config)):
        INIT_COMPONENTS = {'i2c': True, 'co2_sensor': True, 'relays': True}
    cfg = GasOnly()
    # Preserve instance-specific configuration as well as class attributes.
    cfg.__dict__.update(vars(config))
    cfg.INIT_COMPONENTS = GasOnly.INIT_COMPONENTS.copy()
    bio = Bioreactor(cfg)
    def read():
        v = io.read_co2(bio)
        return v, time.monotonic()
    def valve(on):
        if not (io.relay_on if on else io.relay_off)(bio, 'CO2'):
            raise RuntimeError('CO2 relay write failed')
    done = threading.Event()
    old = {}
    controller = None
    try:
        with open(log_path, 'x') as f:
            def log(row):
                f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
            controller = CO2Control(read, valve, getattr(cfg, 'CO2_MPC', None), log=log)
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, lambda *_: done.set())
            controller.start(target, owner='standalone', duration_s=duration_s)
            while controller.active and not done.wait(0.5):
                pass
            controller.stop()
            if controller.fault:
                raise RuntimeError(controller.fault)
    finally:
        if controller:
            controller.stop()
        bio.finish()
        for sig, handler in old.items():
            signal.signal(sig, handler)
