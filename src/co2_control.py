"""Shared CO2 worker for API and standalone use; injected sensor/valve adapters.

read_sample returns (ppm, monotonic acquisition time). set_valve(True) energizes
the CO2 relay and must raise on a failed write. It must not wait on an I2C lock:
GPIO closure must remain independent of slow sensor traffic.
"""
from dataclasses import asdict, replace
import json
import threading
import time
from .co2_mpc import GasModel, MPCSettings, PulseMPC, finite


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

    def validate(self, target):
        if not isinstance(self.profile, dict) or self.profile.get('validated') is not True:
            raise ValueError('CO2_MPC needs a validated rig-specific model before control')
        model = GasModel(**self.profile['model'])
        settings = MPCSettings(**{**self.profile.get('settings', {}), 'target_ppm': target})
        PulseMPC(model, settings)  # validate model/horizon together
        return model, settings

    def start(self, target, owner='api', duration_s=None):
        model, settings = self.validate(target)
        if duration_s is not None:
            finite(duration_s, 'duration', strict=True)
        with self.lock:
            if self._thread and self._thread.is_alive() and not self.active:
                raise RuntimeError('CO2 worker is still stopping')
            if self.active:
                if owner != self.owner:
                    raise RuntimeError(f'CO2 valve is owned by {self.owner}')
                # Setpoint changes preserve the observer and in-flight gas history.
                self.engine.settings = replace(self.engine.settings, target_ppm=target)
                self.deadline = self.clock()+duration_s if duration_s else None
                return
            # A restart loses unobserved gas state. Refuse until it has mixed.
            if self._last_stop is not None and self.clock()-self._last_stop < model.delay_s+5*model.mixing_s:
                raise RuntimeError('wait for prior injected gas to settle before restarting CO2 control')
            value, measured_at = self.read_sample()
            engine = PulseMPC(model, settings)
            engine.observe(value, measured_at, self.clock())
            self.set_valve(False)
            self.engine, self.owner = engine, owner
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
            return {'active': self.active, 'owner': self.owner, 'fault': self.fault,
                    'configured': bool(self.profile and self.profile.get('validated')),
                    'target_ppm': self.engine.settings.target_ppm if self.engine else None,
                    'last': dict(self.last)}

    def _run(self):
        try:
            while not self._stop.is_set():
                with self.lock:
                    if not self.active or (self.deadline and self.clock() >= self.deadline):
                        break
                    value, acquired = self.read_sample()
                    now = self.clock()
                    fresh = self.engine.observe(value, acquired, now)
                    pulse = self.engine.propose(now) if fresh else 0.0
                    self.last = {'co2_ppm': value, 'sample_age_s': now-acquired,
                                 **self.engine.last_plan, 'commanded_pulse_s': pulse}
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
    finite(duration_s, 'duration', strict=True)
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
            controller.start(target, owner='standalone')
            end = time.monotonic()+duration_s
            while controller.active and time.monotonic() < end and not done.wait(0.5):
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
