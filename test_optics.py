"""
Hardware-free tests for the optical configuration layer (src/optics.py) and the
label/column/record behaviour that depends on it.

Run from the repo root:   python3 -m unittest test_optics -v      (or: python3 test_optics.py)
Needs numpy (src/utils.py imports it); the ADS1115/eyespy drivers are never imported.
The module temporarily creates src/config.py from config_default.py if it is missing.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_optics():
    # Import by path so src/__init__.py (which needs the gitignored src/config.py)
    # is not triggered for the pure-resolver tests.
    spec = importlib.util.spec_from_file_location('optics', os.path.join(HERE, 'src', 'optics.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['optics'] = mod          # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


optics = _load_optics()


def cfg(**attrs):
    """A bare config object with the given attributes."""
    return types.SimpleNamespace(**attrs)


LEGACY_INIT = {'optical_density': True, 'eyespy_adc': True}


# =======================================================================================
# Resolver: legacy configs
# =======================================================================================
class LegacyResolution(unittest.TestCase):
    def test_default_channels_become_sources_and_od(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS=LEGACY_INIT,
            OD_ADC_CHANNELS={'135': 'A0', 'Ref': 'A1', '90': 'A2'},
            EYESPY_ADC={'eyespy1': {'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0},
                        'eyespy2': {'i2c_address': '0x4a', 'i2c_bus': 1, 'gain': 1.0}},
        ))
        self.assertTrue(plan.legacy)
        self.assertEqual(plan.errors, [])
        self.assertEqual(list(plan.sources), ['135', 'Ref', '90', 'eyespy1', 'eyespy2'])
        self.assertEqual(plan.sources['eyespy2'].i2c_address, 0x4a)      # string address accepted
        self.assertEqual(plan.od, {'OD_ref': 'Ref', 'OD_90': '90', 'OD_135': '135'})
        # labels are byte-identical to the historical CSV columns
        self.assertEqual(plan.od_label('OD_ref'), 'OD_Ref_V')
        self.assertEqual(plan.od_key('OD_ref'), 'od_ref')
        self.assertEqual(plan.logged_columns(), [
            ('od_135', 'OD_135_V', 'optical_density'),
            ('od_ref', 'OD_Ref_V', 'optical_density'),
            ('od_90', 'OD_90_V', 'optical_density'),
            ('eyespy_eyespy1_raw', 'Eyespy_eyespy1_raw', 'eyespy_adc'),
            ('eyespy_eyespy1_voltage', 'Eyespy_eyespy1_V', 'eyespy_adc'),
            ('eyespy_eyespy2_raw', 'Eyespy_eyespy2_raw', 'eyespy_adc'),
            ('eyespy_eyespy2_voltage', 'Eyespy_eyespy2_V', 'eyespy_adc'),
        ])
        # a legacy rig with eyespy boards is told the EKF stays idle (as before)
        self.assertTrue(any('stays idle' in w for w in plan.warnings))

    def test_non_canonical_channel_names_are_plain_sources(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS={'optical_density': True},
            OD_ADC_CHANNELS={'Trx': 'A0', 'Ref': 'A1', 'Sct': 'A2'},
        ))
        self.assertEqual(plan.od, {'OD_ref': 'Ref'})
        self.assertEqual([c[1] for c in plan.logged_columns()], ['OD_Trx_V', 'OD_Ref_V', 'OD_Sct_V'])

    def test_disabled_components_contribute_nothing(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS={'optical_density': False, 'eyespy_adc': False},
            OD_ADC_CHANNELS={'135': 'A0'}, EYESPY_ADC={'ref': {}},
        ))
        self.assertTrue(plan.legacy)
        self.assertEqual(plan.sources, {})
        self.assertEqual(plan.od, {})

    def test_legacy_ekf_is_the_historical_rule(self):
        """The old driver never read EKF_OD_CHANNEL: it always used '135', and an initialised
        eyespy component took precedence (-> eyespy_135_voltage, which never exists)."""
        both = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS=LEGACY_INIT,
            OD_ADC_CHANNELS={'135': 'A0', 'Ref': 'A1', '90': 'A2'},
            EYESPY_ADC={'sct': {'i2c_address': 0x4a}}, EKF_OD_CHANNEL='sct',
        ))
        self.assertEqual(both.resolve_ekf_channel('sct'), ('eyespy_135_voltage', 'Eyespy_135_V'))
        self.assertEqual(both.resolve_ekf_channel('anything', eyespy_initialized=False),
                         ('od_135', 'OD_135_V'))
        self.assertIsNone(both.resolve_ekf_channel('135', od_initialized=False, eyespy_initialized=False))
        self.assertTrue(any("EKF_OD_CHANNEL='sct' is ignored" in w for w in both.warnings))
        od_only = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS={'optical_density': True},
            OD_ADC_CHANNELS={'Trx': 'A0'}, EKF_OD_CHANNEL='135'))
        self.assertEqual(od_only.resolve_ekf_channel('135'), ('od_135', 'OD_135_V'))   # missing column, as before
        self.assertEqual(od_only.warnings, [])

    def test_missing_config_keys_use_driver_defaults_but_log_nothing(self):
        """Component enabled without its dict key: initialised from the driver defaults
        (as before) but no CSV column (the old label step required the key)."""
        plan = optics.resolve_optical_config(cfg(INIT_COMPONENTS=LEGACY_INIT))
        self.assertEqual(list(plan.sources), ['Trx', 'Ref', 'Sct', 'eyespy1'])
        self.assertEqual(plan.sources['eyespy1'].i2c_address, 0x49)
        self.assertFalse(any(s.logged for s in plan.sources.values()))
        self.assertEqual(plan.logged_columns(), [])
        self.assertIn('[unlogged]', optics.describe(plan))

    def test_invalid_pin_keeps_its_column(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS={'optical_density': True},
            OD_ADC_CHANNELS={'135': 'A0', 'Bad': 'A7'}))
        self.assertEqual(list(plan.sources), ['135', 'Bad'])
        self.assertIsNone(plan.sources['Bad'].channel)
        self.assertEqual([c[1] for c in plan.logged_columns()], ['OD_135_V', 'OD_Bad_V'])
        self.assertEqual(len(plan.errors), 1)

    def test_malformed_legacy_entries_never_raise(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS=LEGACY_INIT, OD_ADC_CHANNELS=['135'],
            EYESPY_ADC={'a': 0x49, 'b': {'i2c_address': 'zz'}, 'c': {'i2c_bus': None}, 'ok': {}}))
        self.assertEqual(list(plan.sources), ['ok'])
        self.assertEqual(len(plan.errors), 4)

    def test_board_colliding_with_channel_name_is_left_to_init_fallback(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS=LEGACY_INIT, OD_ADC_CHANNELS={'Ref': 'A1'}, EYESPY_ADC={'Ref': {}}))
        self.assertEqual(list(plan.sources), ['Ref'])
        self.assertEqual(plan.sources['Ref'].kind, 'adc')
        self.assertTrue(any('collides' in e for e in plan.errors))


# =======================================================================================
# Resolver: new-style configs
# =======================================================================================
class NewResolution(unittest.TestCase):
    SOURCES = {
        'pd_135': {'kind': 'adc', 'channel': 'A0'},
        'pd_ref': {'kind': 'adc', 'channel': 'A1'},
        'pd_90': 'adc:A2',
        'eyespy1': {'kind': 'eyespy', 'i2c_address': '0x49', 'i2c_bus': 1, 'gain': 1.0},
        'eyespy2': 'eyespy:0x4a',
    }

    def test_full_form(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS=LEGACY_INIT, VOLTAGE_SOURCES=self.SOURCES,
            OD_MEASUREMENTS={'OD_45': {'enabled': False}, 'OD_ref': {'enabled': True, 'source': 'pd_ref'},
                             'OD_90': 'pd_90', 'OD_135': 'eyespy1'},
        ))
        self.assertFalse(plan.legacy)
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.warnings, [])
        self.assertEqual(plan.od, {'OD_ref': 'pd_ref', 'OD_90': 'pd_90', 'OD_135': 'eyespy1'})
        self.assertEqual(plan.sources['pd_90'].channel, 'A2')
        self.assertEqual(plan.sources['eyespy2'].i2c_address, 0x4a)
        self.assertEqual(plan.unmapped_sources(), ['pd_135', 'eyespy2'])
        self.assertEqual(plan.logged_columns(), [
            ('od_ref', 'OD_ref_V', 'optical_density'),
            ('od_90', 'OD_90_V', 'optical_density'),
            ('od_135', 'OD_135_V', 'eyespy_adc'),          # OD from an eyespy board
            ('voltage_pd_135', 'pd_135_V', 'optical_density'),
            ('voltage_eyespy2', 'eyespy2_V', 'eyespy_adc'),
        ])
        self.assertEqual(plan.resolve_ekf_channel('135'), ('od_135', 'OD_135_V'))
        self.assertEqual(plan.resolve_ekf_channel('OD_90'), ('od_90', 'OD_90_V'))
        self.assertEqual(plan.resolve_ekf_channel('eyespy1'), ('od_135', 'OD_135_V'))   # mapped source -> its OD
        self.assertEqual(plan.resolve_ekf_channel('eyespy2'), ('voltage_eyespy2', 'eyespy2_V'))
        self.assertIsNone(plan.resolve_ekf_channel('OD_45'))                            # disabled
        self.assertIsNone(plan.resolve_ekf_channel('nope'))

    def test_true_shorthand_means_pd_named_source(self):
        plan = optics.resolve_optical_config(cfg(
            VOLTAGE_SOURCES={'pd_135': 'adc:A0', 'pd_ref': 'adc:A1', '135': 'adc:A2'},
            OD_MEASUREMENTS={'OD_135': True, 'OD_ref': True, 'OD_90': True},
        ))
        self.assertEqual(plan.od, {'OD_ref': 'pd_ref', 'OD_135': 'pd_135'})
        self.assertEqual(len(plan.errors), 2)
        self.assertIn("invalid source name '135'", plan.errors[0])
        self.assertIn("True needs a VOLTAGE_SOURCE named 'pd_90'", plan.errors[1])

    def test_reserved_names(self):
        plan = optics.resolve_optical_config(cfg(
            VOLTAGE_SOURCES={'led': 'adc:A0', 'ref': 'adc:A1', 'OD_135': 'adc:A2', 'od_ref': 'adc:A3',
                             'Voltage': 'eyespy:0x49', 'fine': 'eyespy:0x4a'},
            OD_MEASUREMENTS={}))
        self.assertEqual(list(plan.sources), ['fine'])
        self.assertEqual(sum('reserved name' in e for e in plan.errors), 5)
        self.assertTrue(any('no OD_MEASUREMENTS is enabled' in w for w in plan.warnings))

    def test_errors_are_collected_not_raised(self):
        plan = optics.resolve_optical_config(cfg(
            VOLTAGE_SOURCES={'a': 'adc:A0', 'b': 'adc:A0', 'c': 'adc:A9', 'd': {'kind': 'laser'},
                             'e': 'eyespy:zz', 'F': 'adc:A1', 'f': 'adc:A2', 'g': 7,
                             'h': {'kind': 'eyespy', 'i2c_address': 0x49, 'i2c_bus': 'one'},
                             'i': {'kind': 'eyespy', 'i2c_address': None},
                             'j': {'kind': 'eyespy', 'i2c_address': 0x4b, 'gain': 'x'}},
            OD_MEASUREMENTS={'OD_180': 'a', 'OD_135': 'missing', 'OD_90': 'a', 'OD_ref': 42},
        ))
        self.assertEqual(list(plan.sources), ['a', 'F'])
        self.assertEqual(plan.od, {'OD_90': 'a'})
        joined = '\n'.join(plan.errors)
        for frag in ("already used by 'a'", "channel must be one of", "kind must be one of",
                     "i2c_address must be an integer", "collides with 'F'", "must be a dict or an",
                     "i2c_bus must be an integer", "gain must be numeric",
                     "'OD_180' is not one of", "unknown source 'missing'"):
            self.assertIn(frag, joined)

    def test_half_defined_falls_back_to_legacy(self):
        plan = optics.resolve_optical_config(cfg(
            INIT_COMPONENTS={'optical_density': True},
            OD_ADC_CHANNELS={'135': 'A0'}, VOLTAGE_SOURCES={'x': 'adc:A0'},
        ))
        self.assertTrue(plan.legacy)
        self.assertEqual(plan.od, {'OD_135': '135'})
        self.assertEqual(len(plan.errors), 1)

    def test_describe(self):
        plan = optics.resolve_optical_config(cfg(
            VOLTAGE_SOURCES={'pd_135': 'adc:A0'}, OD_MEASUREMENTS={'OD_135': 'pd_135'}))
        self.assertEqual(optics.describe(plan), 'sources: pd_135(adc:A0); OD: OD_135<-pd_135')


# =======================================================================================
# Integration: a real Bioreactor (fake component inits) -> SENSOR_LABELS / fieldnames /
# measure_and_record_sensors rows / measure_od dispatch. Needs numpy, no hardware.
# =======================================================================================
_CREATED_CONFIG = False


def setUpModule():
    global _CREATED_CONFIG
    cfg_path = os.path.join(HERE, 'src', 'config.py')
    if not os.path.exists(cfg_path):
        shutil.copy(os.path.join(HERE, 'src', 'config_default.py'), cfg_path)
        _CREATED_CONFIG = True
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def tearDownModule():
    if _CREATED_CONFIG:
        os.remove(os.path.join(HERE, 'src', 'config.py'))


def _numpy_available():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


class _Chan:
    """Stand-in for an ADS1115 AnalogIn: .voltage returns the configured value."""
    def __init__(self, v): self.v = v
    @property
    def voltage(self): return self.v


class _Led:
    def __init__(self): self.power = None; self.calls = []
    def set_power(self, p): self.power = p; self.calls.append(p); return True
    def off(self): self.power = 0.0; self.calls.append(0.0)


@unittest.skipUnless(_numpy_available(), "numpy not installed (needed by src/utils.py)")
class BioreactorIntegration(unittest.TestCase):

    def setUp(self):
        from src import components
        self.components = components
        self._saved_registry = dict(components.COMPONENT_REGISTRY)
        self.tmp = tempfile.mkdtemp()
        self.ack_boards = None         # None = every configured eyespy board responds
        self.adc_volts = {}            # source name -> volts read by the fake AnalogIn
        test = self

        def fake_od(bio, c):
            plan = bio.optics
            specs = plan.sources_of('adc') if plan.sources else []
            bio.od_channels = {s.name: _Chan(test.adc_volts.get(s.name, 0.0)) for s in specs if s.channel}
            return {'initialized': bool(bio.od_channels)}

        def fake_eyespy(bio, c):
            plan = bio.optics
            specs = plan.sources_of('eyespy') if plan.sources else []
            if test.ack_boards is not None:
                specs = [s for s in specs if s.name in test.ack_boards]
            bio.eyespy_boards = {s.name: s.as_board_config() for s in specs}
            return {'initialized': bool(bio.eyespy_boards)}

        def fake_led(bio, c):
            bio.led_driver = _Led()
            return {'initialized': True}

        def fake_ok(bio, c):
            return {'initialized': True}

        components.COMPONENT_REGISTRY.update({
            'optical_density': fake_od, 'eyespy_adc': fake_eyespy, 'led': fake_led, 'i2c': fake_ok,
        })

    def tearDown(self):
        self.components.COMPONENT_REGISTRY.clear()
        self.components.COMPONENT_REGISTRY.update(self._saved_registry)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _config(self, **overrides):
        from src.config_default import Config

        class C(Config):
            pass
        C.INIT_COMPONENTS = {'i2c': True, 'optical_density': True, 'eyespy_adc': False, 'led': True,
                             'temp_sensor': False, 'peltier_driver': False, 'stirrer': False,
                             'ring_light': False, 'co2_sensor': False, 'o2_sensor': False,
                             'ambient_temp': False, 'peltier_current': False, 'pumps': False,
                             'relays': False}
        C.SENSOR_LABELS = {}
        C.DATA_LOGGING = False
        C.LOG_TO_TERMINAL = False
        C.LOG_FILE = os.path.join(self.tmp, 'test.log')
        C.CLEAR_LOG_ON_START = False
        C.EKF_OD_CHANNEL = '135'       # what old per-rig config files carry
        # None == "not defined" for the resolver -> legacy derivation, unless the test sets them
        for k in ('VOLTAGE_SOURCES', 'OD_MEASUREMENTS'):
            if k not in overrides:
                setattr(C, k, None)
        for k, v in overrides.items():
            setattr(C, k, v)
        return C

    def _make(self, C):
        from src.bioreactor import Bioreactor
        return Bioreactor(C())

    def _record(self, bio, gated, raw=1234, **kwargs):
        """Run measure_and_record_sensors with io stubbed, return (sensor_data, csv_row)."""
        from src import io as bio_io, utils
        rows = []

        class W:
            def writerow(self, r): rows.append(dict(r))

        class F:
            def flush(self): pass
        bio.writer, bio.out_file = W(), F()
        saved = (bio_io.measure_od, bio_io.read_eyespy_adc, bio_io.read_all_voltages)
        if gated is None:
            def _no(*a, **k): raise AssertionError("measure_od must not be called on the cached path")
            bio_io.measure_od = _no
        else:
            bio_io.measure_od = lambda b, led_power, averaging_duration, channel_name='all': dict(gated)
        bio_io.read_eyespy_adc = lambda b, board_name=None: raw
        bio_io.read_all_voltages = lambda b: dict(gated or {})
        try:
            data = utils.measure_and_record_sensors(bio, elapsed=1.0, **kwargs)
        finally:
            bio_io.measure_od, bio_io.read_eyespy_adc, bio_io.read_all_voltages = saved
        self.assertEqual(len(rows), 1)
        return data, rows[0]

    LEGACY_FIELDS = ['time', 'elapsed_time', 'OD_135_V', 'OD_Ref_V', 'OD_90_V',
                     'Eyespy_eyespy1_raw', 'Eyespy_eyespy1_V',
                     'ekf_od_est', 'ekf_growth_rate', 'ekf_doubling_time_s',
                     'ekf_od_std', 'ekf_growth_rate_std', 'ekf_doubling_time_std_s']

    def _legacy_config(self, **extra):
        kw = {'OD_ADC_CHANNELS': {'135': 'A0', 'Ref': 'A1', '90': 'A2'},
              'EYESPY_ADC': {'eyespy1': {'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0}}}
        kw.update(extra)
        C = self._config(**kw)
        C.INIT_COMPONENTS['eyespy_adc'] = True
        return C

    # ---------------------------------------------------------------- legacy parity
    def test_legacy_config_columns_unchanged(self):
        bio = self._make(self._legacy_config())
        self.assertTrue(bio.optics.legacy)
        self.assertEqual(bio.fieldnames, self.LEGACY_FIELDS)
        self.assertEqual(list(bio.od_channels), ['135', 'Ref', '90'])
        self.assertEqual(list(bio.eyespy_boards), ['eyespy1'])
        data, row = self._record(bio, {'135': 0.5, 'Ref': 0.6, '90': 0.7, 'eyespy1': 1.5})
        self.assertEqual({k: row[k] for k in ('OD_135_V', 'OD_Ref_V', 'OD_90_V', 'Eyespy_eyespy1_raw', 'Eyespy_eyespy1_V')},
                         {'OD_135_V': 0.5, 'OD_Ref_V': 0.6, 'OD_90_V': 0.7, 'Eyespy_eyespy1_raw': 1234, 'Eyespy_eyespy1_V': 1.5})
        self.assertEqual(data['od_135'], 0.5)
        self.assertEqual(data['eyespy_eyespy1_voltage'], 1.5)
        # historical rule: with eyespy initialised the EKF read eyespy_135_voltage (absent) and
        # never started — that stays so for legacy configs
        self.assertFalse(getattr(bio, '_ekf_initialized', False))
        self.assertNotIn('ekf_od_est', row)

    def test_legacy_od_only_ekf_runs_on_od_135_as_before(self):
        C = self._config(OD_ADC_CHANNELS={'135': 'A0', 'Ref': 'A1'}, EKF_OD_CHANNEL='bogus')
        bio = self._make(C)
        self._record(bio, {'135': 0.5, 'Ref': 0.6})
        self.assertTrue(getattr(bio, '_ekf_initialized', False))     # EKF_OD_CHANNEL was never honoured
        self.assertTrue(any('is ignored for legacy configs' in w for w in bio.optics.warnings))

    def test_legacy_sensor_labels_override_still_honoured(self):
        C = self._config(OD_ADC_CHANNELS={'135': 'A0', 'Ref': 'A1'}, SENSOR_LABELS={'od_Ref': 'reference_V'})
        bio = self._make(C)
        # a pre-seeded SENSOR_LABELS key keeps its insertion position (as before)
        self.assertEqual(bio.fieldnames[2:4], ['reference_V', 'OD_135_V'])
        _, row = self._record(bio, {'135': 0.5, 'Ref': 0.6})
        self.assertEqual(row['reference_V'], 0.6)

    def test_legacy_absent_board_leaves_empty_cells(self):
        """A configured eyespy board that never ACKs keeps its header columns but writes
        nothing into them and adds no sensor_data keys — as the old driver did."""
        C = self._legacy_config(EYESPY_ADC={'eyespy1': {'i2c_address': 0x49}, 'eyespy2': {'i2c_address': 0x4a}})
        self.ack_boards = {'eyespy1'}
        bio = self._make(C)
        self.assertEqual(bio.fieldnames[5:9], ['Eyespy_eyespy1_raw', 'Eyespy_eyespy1_V',
                                               'Eyespy_eyespy2_raw', 'Eyespy_eyespy2_V'])
        data, row = self._record(bio, {'135': 0.5, 'Ref': 0.6, '90': 0.7, 'eyespy1': 1.5})
        self.assertNotIn('eyespy_eyespy2_voltage', data)
        self.assertNotIn('Eyespy_eyespy2_V', row)
        self.assertEqual(row['Eyespy_eyespy1_V'], 1.5)

    def test_legacy_key_absent_initialises_defaults_but_logs_no_columns(self):
        C = self._config()                     # optical_density on, no OD_ADC_CHANNELS at all
        C.INIT_COMPONENTS['eyespy_adc'] = True  # and no EYESPY_ADC
        bio = self._make(C)
        self.assertEqual(list(bio.od_channels), ['Trx', 'Ref', 'Sct'])
        self.assertEqual(list(bio.eyespy_boards), ['eyespy1'])
        self.assertEqual(bio.fieldnames[2], 'ekf_od_est')          # no optical columns, as before
        data, row = self._record(bio, {'Trx': 0.1, 'Ref': 0.2, 'Sct': 0.3, 'eyespy1': 1.1})
        self.assertEqual(data['od_trx'], 0.1)                      # keys still computed, as before
        self.assertNotIn('OD_Trx_V', row)

    def test_legacy_invalid_pin_keeps_column_with_nan(self):
        C = self._config(OD_ADC_CHANNELS={'135': 'A0', 'Bad': 'A7'})
        bio = self._make(C)
        self.assertEqual(list(bio.od_channels), ['135'])
        self.assertEqual(bio.fieldnames[2:4], ['OD_135_V', 'OD_Bad_V'])
        _, row = self._record(bio, {'135': 0.5})
        self.assertTrue(row['OD_Bad_V'] != row['OD_Bad_V'])         # NaN, as before

    def test_legacy_eyespy_only_rig(self):
        C = self._config(EYESPY_ADC={'ref': {'i2c_address': 0x49}, 'sct1': {'i2c_address': 0x4a}})
        C.INIT_COMPONENTS.update(optical_density=False, eyespy_adc=True)
        bio = self._make(C)
        self.assertEqual(bio.optics.od, {})
        self.assertEqual(bio.fieldnames[2:6], ['Eyespy_ref_raw', 'Eyespy_ref_V', 'Eyespy_sct1_raw', 'Eyespy_sct1_V'])
        data, row = self._record(bio, {'ref': 0.9, 'sct1': 1.1})
        self.assertEqual(row['Eyespy_sct1_V'], 1.1)
        self.assertFalse(getattr(bio, '_ekf_initialized', False))

    def test_use_cached_override_is_keyed_by_source(self):
        """The API run loop passes od_sampler.latest() ({source: V}) as od_override."""
        bio = self._make(self._legacy_config())
        data, row = self._record(bio, None, use_cached=True,
                                 od_override={'135': 0.5, 'Ref': 0.6, '90': 0.7, 'eyespy1': 1.5})
        self.assertEqual((row['OD_Ref_V'], row['Eyespy_eyespy1_V'], row['Eyespy_eyespy1_raw']), (0.6, 1.5, 1234))
        data, row = self._record(bio, None, use_cached=True, od_override=None)      # sampler off
        self.assertTrue(row['OD_135_V'] != row['OD_135_V'])
        self.assertTrue(data['eyespy_eyespy1_voltage'] != data['eyespy_eyespy1_voltage'])

    def test_sensor_labels_idempotent_across_constructions(self):
        """The API constructs Bioreactor() against one class-level SENSOR_LABELS dict."""
        C = self._legacy_config()
        first = self._make(C).fieldnames
        n = len(C.SENSOR_LABELS)
        second = self._make(C).fieldnames
        self.assertEqual(first, second)
        self.assertEqual(len(C.SENSOR_LABELS), n)

    # ---------------------------------------------------------------- new-style
    def test_new_config_columns(self):
        C = self._config(
            VOLTAGE_SOURCES={'pd_135': 'adc:A0', 'pd_ref': 'adc:A1', 'spare': 'adc:A2',
                             'eyespy1': 'eyespy:0x49', 'eyespy2': 'eyespy:0x4a'},
            OD_MEASUREMENTS={'OD_ref': 'pd_ref', 'OD_135': 'pd_135', 'OD_90': 'eyespy1', 'OD_45': False},
            EKF_OD_CHANNEL='OD_90')
        C.INIT_COMPONENTS['eyespy_adc'] = True
        bio = self._make(C)
        self.assertFalse(bio.optics.legacy)
        self.assertEqual(bio.fieldnames[2:7], ['OD_ref_V', 'OD_90_V', 'OD_135_V', 'spare_V', 'eyespy2_V'])
        self.assertEqual(list(bio.eyespy_boards), ['eyespy1', 'eyespy2'])
        data, row = self._record(bio, {'pd_135': 0.1, 'pd_ref': 0.2, 'spare': 0.3, 'eyespy1': 1.1, 'eyespy2': 1.2})
        self.assertEqual({k: row[k] for k in ('OD_ref_V', 'OD_90_V', 'OD_135_V', 'spare_V', 'eyespy2_V')},
                         {'OD_ref_V': 0.2, 'OD_90_V': 1.1, 'OD_135_V': 0.1, 'spare_V': 0.3, 'eyespy2_V': 1.2})
        self.assertNotIn('Eyespy_eyespy1_raw', row)
        self.assertEqual(data['od_90'], 1.1)            # OD from an eyespy board
        self.assertTrue(getattr(bio, '_ekf_initialized', False))   # EKF_OD_CHANNEL='OD_90' honoured

    def test_new_config_component_down_drops_its_columns(self):
        C = self._config(VOLTAGE_SOURCES={'pd_135': 'adc:A0', 'eyespy1': 'eyespy:0x49'},
                         OD_MEASUREMENTS={'OD_135': 'pd_135', 'OD_90': 'eyespy1'})
        bio = self._make(C)                       # eyespy_adc stays disabled
        self.assertEqual(bio.fieldnames[2:3], ['OD_135_V'])
        self.assertNotIn('OD_90_V', bio.fieldnames)
        data, row = self._record(bio, {'pd_135': 0.4})
        self.assertEqual(row['OD_135_V'], 0.4)
        self.assertNotIn('OD_90_V', row)
        self.assertTrue(data['od_90'] != data['od_90'])   # NaN in sensor_data, never written

    def test_new_config_with_no_sources_initialises_nothing(self):
        C = self._config(VOLTAGE_SOURCES={}, OD_MEASUREMENTS={})
        C.INIT_COMPONENTS['eyespy_adc'] = True
        from src import components
        # use the REAL inits' plan logic (they must not fall back to the driver defaults)
        components.COMPONENT_REGISTRY['optical_density'] = self._saved_registry['optical_density']
        components.COMPONENT_REGISTRY['eyespy_adc'] = self._saved_registry['eyespy_adc']
        bio = self._make(C)
        self.assertFalse(bio.is_component_initialized('optical_density'))
        self.assertFalse(bio.is_component_initialized('eyespy_adc'))
        self.assertFalse(hasattr(bio, 'od_channels'))
        self.assertEqual(bio.fieldnames[2], 'ekf_od_est')

    def test_read_voltage_and_read_od_dispatch(self):
        from src import io as bio_io
        C = self._config(VOLTAGE_SOURCES={'pd_135': 'adc:A0', 'eyespy1': 'eyespy:0x49'},
                         OD_MEASUREMENTS={'OD_135': 'eyespy1'})
        C.INIT_COMPONENTS['eyespy_adc'] = True
        self.adc_volts = {'pd_135': 0.75}
        bio = self._make(C)
        with mock.patch.object(bio_io, 'read_eyespy_voltage', lambda b, board_name=None: 2.5):
            self.assertEqual(bio_io.read_voltage(bio, 'eyespy1'), 2.5)
            self.assertEqual(bio_io.read_voltage(bio, 'pd_135'), 0.75)
            self.assertEqual(bio_io.read_od(bio, 'OD_135'), 2.5)
            self.assertIsNone(bio_io.read_od(bio, 'OD_45'))
            self.assertIsNone(bio_io.read_voltage(bio, 'nope', quiet=True))
            self.assertEqual(bio_io.read_all_voltages(bio), {'pd_135': 0.75, 'eyespy1': 2.5})

    def test_measure_od_dispatch(self):
        """The real io.measure_od with the LED, sleeps and reads stubbed."""
        from src import io as bio_io
        # legacy: a single channel name reads that channel, every eyespy board rides along -> dict
        self.adc_volts = {'135': 0.5, 'Ref': 0.6, '90': 0.7}
        bio = self._make(self._legacy_config())
        with mock.patch('time.sleep'), mock.patch.object(bio_io, 'read_eyespy_voltage', lambda b, board_name=None: 1.5):
            res = bio_io.measure_od(bio, 10.0, 0.001, '135')
            self.assertEqual(res, {'135': 0.5, 'eyespy1': 1.5})
            self.assertEqual(bio_io.measure_od(bio, 10.0, 0.001, 'all'), {'135': 0.5, 'Ref': 0.6, '90': 0.7, 'eyespy1': 1.5})
            self.assertEqual(bio_io.measure_od(bio, 10.0, 0.001, 'OD_ref'), {'Ref': 0.6, 'eyespy1': 1.5})  # OD name resolves
        self.assertEqual(bio.led_driver.calls[0], 10.0)     # LED pulsed on ...
        self.assertEqual(bio.led_driver.power, 0.0)         # ... and left off
        # new-style: a single name (source or OD name) returns that source's float
        C = self._config(VOLTAGE_SOURCES={'pd_135': 'adc:A0', 'eyespy1': 'eyespy:0x49'},
                         OD_MEASUREMENTS={'OD_135': 'eyespy1', 'OD_90': 'pd_135'})
        C.INIT_COMPONENTS['eyespy_adc'] = True
        self.adc_volts = {'pd_135': 0.42}
        bio = self._make(C)
        with mock.patch('time.sleep'), mock.patch.object(bio_io, 'read_eyespy_voltage', lambda b, board_name=None: 2.5):
            self.assertEqual(bio_io.measure_od(bio, 10.0, 0.001, 'OD_135'), 2.5)
            self.assertAlmostEqual(bio_io.measure_od(bio, 10.0, 0.001, 'pd_135'), 0.42)
            measured = bio_io.measure_od(bio, 10.0, 0.001, 'all')
            self.assertEqual(set(measured), {'pd_135', 'eyespy1'})
            self.assertAlmostEqual(measured['pd_135'], 0.42)
            self.assertAlmostEqual(measured['eyespy1'], 2.5)
            self.assertIsNone(bio_io.measure_od(bio, 10.0, 0.001, 'nope'))


if __name__ == '__main__':
    unittest.main()
