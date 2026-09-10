"""
Optical configuration: named voltage sources and canonical OD measurements.

Two layers sit between the ADCs and everything that consumes "optical density":

* **Voltage sources** — every photodiode/ADC input the rig has, under a name of your
  choosing. A source is an ADS1115 channel (``kind='adc'``, hardware component
  ``optical_density``) or one ADS1114 eyespy board (``kind='eyespy'``, hardware
  component ``eyespy_adc``). Each source can be read by name (``io.read_voltage``),
  gets its own API endpoint, and its own CSV column ``<name>_V`` when no OD
  measurement consumes it.

* **OD measurements** — exactly four canonical names, ``OD_45``, ``OD_ref``, ``OD_90``,
  ``OD_135``. Each is enabled or not and, when enabled, mapped to ONE voltage source
  of either kind. Enabled measurements are IR-gated, logged as ``OD_<x>_V``, and are
  what the dashboard plots.

Config::

    VOLTAGE_SOURCES = {
        'pd_135':  {'kind': 'adc', 'channel': 'A0'},
        'pd_ref':  {'kind': 'adc', 'channel': 'A1'},
        'eyespy1': {'kind': 'eyespy', 'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0},
    }
    OD_MEASUREMENTS = {
        'OD_45':  {'enabled': False},
        'OD_ref': {'enabled': True, 'source': 'pd_ref'},
        'OD_90':  False,          # shorthand: disabled
        'OD_135': 'eyespy1',      # shorthand: enabled, fed by that source
        # 'OD_135': True          # shorthand: enabled, fed by the source named 'pd_135'
    }

Backward compatibility: a config that defines neither key is resolved from the legacy
``OD_ADC_CHANNELS`` / ``EYESPY_ADC`` keys, and the resulting plan is flagged ``legacy``
so every attribute name, sensor_data key, CSV column and the EKF's channel choice
come out exactly as before (``OD_135_V``, ``OD_Ref_V``, ``Eyespy_<board>_raw``,
``Eyespy_<board>_V`` ...). See ``docs/optics.md``.

This module is pure Python: no hardware libraries, safe to import anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

OD_NAMES: Tuple[str, ...] = ('OD_45', 'OD_ref', 'OD_90', 'OD_135')
"""The four canonical OD measurements, in the order they are logged/served."""

OD_SUFFIX: Dict[str, str] = {'OD_45': '45', 'OD_ref': 'ref', 'OD_90': '90', 'OD_135': '135'}

ADC_PINS: Tuple[str, ...] = ('A0', 'A1', 'A2', 'A3')
SOURCE_KINDS: Tuple[str, ...] = ('adc', 'eyespy')
COMPONENT_OF_KIND: Dict[str, str] = {'adc': 'optical_density', 'eyespy': 'eyespy_adc'}

# Names a voltage source may not take (compared case-insensitively): API path words,
# component names, and the OD measurement names / suffixes, which would otherwise be
# ambiguous in EKF_OD_CHANNEL and in the API.
RESERVED_SOURCE_NAMES = frozenset({
    'state', 'all', 'control', 'od', 'voltage', 'voltages', 'capabilities', 'health',
    'i2c', 'temp_sensor', 'peltier_driver', 'stirrer', 'led', 'ring_light',
    'optical_density', 'eyespy_adc', 'co2_sensor', 'o2_sensor', 'ambient_temp',
    'peltier_current', 'pumps', 'relays', 'run', 'history', 'data', 'camera',
} | {s.lower() for s in OD_SUFFIX.values()} | {n.lower() for n in OD_NAMES})
_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')

# Legacy OD_ADC_CHANNELS names that imply a canonical OD measurement.
_LEGACY_OD_BY_CHANNEL: Dict[str, str] = {'45': 'OD_45', 'ref': 'OD_ref', '90': 'OD_90', '135': 'OD_135'}

# What the pre-optics driver effectively used for the EKF on every legacy rig: the
# config's EKF_OD_CHANNEL was read from an attribute that never existed, so '135' was
# always used, and an initialised eyespy component took precedence over the ADC.
_LEGACY_EKF_CHANNEL = '135'


@dataclass
class SourceSpec:
    name: str
    kind: str                        # 'adc' | 'eyespy'
    channel: Optional[str] = None    # adc: 'A0'..'A3' (None = invalid pin kept for its column)
    i2c_address: Optional[int] = None
    i2c_bus: int = 1
    gain: float = 1.0
    legacy_name: Optional[str] = None  # legacy plans: the original OD_ADC_CHANNELS / EYESPY_ADC key
    logged: bool = True                # False: initialised for reads but never given a CSV column
                                       # (legacy configs that enable a component without its dict key)

    @property
    def component(self) -> str:
        return COMPONENT_OF_KIND[self.kind]

    def as_board_config(self) -> Dict[str, Any]:
        """The dict shape ``components.init_eyespy_adc`` / ``io.read_eyespy_*`` expect."""
        return {'i2c_address': self.i2c_address, 'i2c_bus': self.i2c_bus, 'gain': self.gain}


@dataclass
class OpticalPlan:
    sources: Dict[str, SourceSpec] = field(default_factory=dict)
    od: Dict[str, str] = field(default_factory=dict)      # enabled OD name -> source name (canonical order)
    legacy: bool = False
    errors: List[str] = field(default_factory=list)      # config mistakes (the item was dropped)
    warnings: List[str] = field(default_factory=list)    # things worth knowing that changed nothing

    # ------------------------------------------------------------------ queries
    def sources_of(self, kind: str) -> List[SourceSpec]:
        return [s for s in self.sources.values() if s.kind == kind]

    def source_names(self, kind: Optional[str] = None) -> List[str]:
        return [s.name for s in self.sources.values() if kind is None or s.kind == kind]

    def source_for(self, od_name: str) -> Optional[SourceSpec]:
        src = self.od.get(od_name)
        return self.sources.get(src) if src else None

    def unmapped_sources(self) -> List[str]:
        used = set(self.od.values())
        return [n for n in self.sources if n not in used]

    def od_names_for_source(self, source: str) -> List[str]:
        return [od for od, s in self.od.items() if s == source]

    def has_optics(self) -> bool:
        return bool(self.sources)

    # ------------------------------------------------------------------ keys + labels
    # sensor_data keys are what utils.measure_and_record_sensors and the EKF read;
    # labels are the CSV column names (SENSOR_LABELS values).
    def od_key(self, od_name: str) -> str:
        return f"od_{OD_SUFFIX[od_name]}"

    def od_label(self, od_name: str) -> str:
        if self.legacy:
            src = self.source_for(od_name)
            if src is not None and src.legacy_name is not None:
                return f"OD_{src.legacy_name}_V"        # e.g. OD_Ref_V, exactly as before
        return f"OD_{OD_SUFFIX[od_name]}_V"

    def source_key(self, name: str) -> str:
        src = self.sources[name]
        if self.legacy:
            if src.kind == 'eyespy':
                return f"eyespy_{src.legacy_name or name}_voltage"
            return f"od_{(src.legacy_name or name).lower()}"
        return f"voltage_{name}"

    def source_label(self, name: str) -> str:
        src = self.sources[name]
        if self.legacy:
            if src.kind == 'eyespy':
                return f"Eyespy_{src.legacy_name or name}_V"
            return f"OD_{src.legacy_name or name}_V"
        return f"{name}_V"

    def legacy_raw_key(self, name: str) -> str:
        """Legacy eyespy boards also log a raw ADC count column."""
        src = self.sources[name]
        return f"eyespy_{src.legacy_name or name}_raw"

    def legacy_raw_label(self, name: str) -> str:
        src = self.sources[name]
        return f"Eyespy_{src.legacy_name or name}_raw"

    def logged_columns(self) -> List[Tuple[str, str, str]]:
        """Every optical column this plan logs, as (sensor_data key, CSV label, component).

        Order: OD measurements (canonical order), then sources that no OD measurement
        consumes (config order). Legacy plans reproduce today's columns: every adc channel
        as OD_<chan>_V and every eyespy board as Eyespy_<b>_raw + Eyespy_<b>_V, skipping
        sources synthesised from driver defaults (``logged=False``).
        """
        cols: List[Tuple[str, str, str]] = []
        if self.legacy:
            for name, src in self.sources.items():
                if src.kind == 'adc' and src.logged:
                    cols.append((self.source_key(name), self.source_label(name), src.component))
            for name, src in self.sources.items():
                if src.kind == 'eyespy' and src.logged:
                    cols.append((self.legacy_raw_key(name), self.legacy_raw_label(name), src.component))
                    cols.append((self.source_key(name), self.source_label(name), src.component))
            return cols
        for od_name, src_name in self.od.items():
            cols.append((self.od_key(od_name), self.od_label(od_name), self.sources[src_name].component))
        for name in self.unmapped_sources():
            cols.append((self.source_key(name), self.source_label(name), self.sources[name].component))
        return cols

    def resolve_ekf_channel(self, ekf_channel: Optional[str], *,
                            od_initialized: Optional[bool] = None,
                            eyespy_initialized: Optional[bool] = None) -> Optional[Tuple[str, str]]:
        """Map EKF_OD_CHANNEL to (sensor_data key, CSV label), or None for "no EKF input".

        New-style plans: accepts an OD measurement name ('OD_135'), its suffix ('135',
        'ref'), or a voltage source name (a mapped source resolves to the OD measurement it
        feeds, an unmapped one to its own ``<name>_V`` column).

        Legacy plans reproduce the historical driver EXACTLY, so an existing rig's EKF
        columns do not change when the driver is upgraded: ``EKF_OD_CHANNEL`` is ignored
        (the old code never managed to read it), channel '135' is used, and an initialised
        eyespy component takes precedence over the ADC — which, unless a board is literally
        named '135', means no EKF on rigs with eyespy boards. Pass ``od_initialized`` /
        ``eyespy_initialized`` (the components' runtime state) for an exact match; they
        default to "any source of that kind is configured".
        """
        if self.legacy:
            if eyespy_initialized is None:
                eyespy_initialized = bool(self.sources_of('eyespy'))
            if od_initialized is None:
                od_initialized = bool(self.sources_of('adc'))
            ch = _LEGACY_EKF_CHANNEL
            if eyespy_initialized:
                return f"eyespy_{ch}_voltage", f"Eyespy_{ch}_V"
            if od_initialized:
                return f"od_{ch}", f"OD_{ch}_V"
            return None
        if not ekf_channel:
            return None
        ch = str(ekf_channel)
        for od_name, suffix in OD_SUFFIX.items():          # canonical measurement, by name or suffix
            if ch.lower() in (od_name.lower(), suffix.lower()):
                if od_name in self.od:
                    return self.od_key(od_name), self.od_label(od_name)
                return None                                 # named but disabled -> no EKF
        if ch in self.sources:                              # a voltage source
            mapped = self.od_names_for_source(ch)
            if mapped:
                return self.od_key(mapped[0]), self.od_label(mapped[0])
            return self.source_key(ch), self.source_label(ch)
        return None


# ---------------------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------------------
def _cfg(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def resolve_optical_config(config) -> OpticalPlan:
    """Build the OpticalPlan for a config object (any object with attributes).

    Never raises for config mistakes: problems are appended to ``plan.errors`` and the
    offending source/measurement is dropped; notes go to ``plan.warnings``. Callers log
    both (``Bioreactor.__init__`` and the API do).
    """
    sources_cfg = _cfg(config, 'VOLTAGE_SOURCES')
    od_cfg = _cfg(config, 'OD_MEASUREMENTS')
    plan = OpticalPlan()

    if sources_cfg is None and od_cfg is None:
        return _resolve_legacy(config, plan)
    if sources_cfg is None or od_cfg is None:
        plan.errors.append(
            "VOLTAGE_SOURCES and OD_MEASUREMENTS must be defined together; "
            "falling back to the legacy OD_ADC_CHANNELS / EYESPY_ADC configuration")
        return _resolve_legacy(config, plan)

    _add_sources(plan, sources_cfg)
    _add_measurements(plan, od_cfg)
    if plan.sources and not plan.od:
        plan.warnings.append("VOLTAGE_SOURCES are configured but no OD_MEASUREMENTS is enabled: "
                             "nothing optical will be plotted by the dashboard")
    return plan


def _parse_int(value, what: str, name: str, plan: OpticalPlan) -> Optional[int]:
    """int from an int or a numeric string ('0x49', '73'); None + error otherwise."""
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        plan.errors.append(f"VOLTAGE_SOURCES[{name!r}]: {what} must be an integer (got {value!r})")
        return None


def _add_sources(plan: OpticalPlan, sources_cfg) -> None:
    if not isinstance(sources_cfg, dict):
        plan.errors.append("VOLTAGE_SOURCES must be a dict of name -> {'kind': ...}")
        return
    seen_lower: Dict[str, str] = {}
    used_pins: Dict[str, str] = {}
    used_addr: Dict[Tuple[int, int], str] = {}
    for name, spec in sources_cfg.items():
        name = str(name)
        if not _NAME_RE.match(name):
            plan.errors.append(f"VOLTAGE_SOURCES: invalid source name {name!r} "
                               "(letters, digits, '_' or '-', starting with a letter)")
            continue
        if name.lower() in RESERVED_SOURCE_NAMES:
            plan.errors.append(f"VOLTAGE_SOURCES: {name!r} is a reserved name")
            continue
        if name.lower() in seen_lower:
            plan.errors.append(f"VOLTAGE_SOURCES: {name!r} collides with {seen_lower[name.lower()]!r}")
            continue
        if isinstance(spec, str):
            # shorthand 'adc:A0' / 'eyespy:0x49'
            kind, _, arg = spec.partition(':')
            spec = {'kind': kind.strip()}
            if kind.strip() == 'adc':
                spec['channel'] = arg.strip()
            elif kind.strip() == 'eyespy' and arg.strip():
                spec['i2c_address'] = arg.strip()
        if not isinstance(spec, dict):
            plan.errors.append(f"VOLTAGE_SOURCES[{name!r}] must be a dict or an 'adc:A0' / 'eyespy:0x49' string")
            continue
        kind = str(spec.get('kind', '')).lower()
        if kind not in SOURCE_KINDS:
            plan.errors.append(f"VOLTAGE_SOURCES[{name!r}]: kind must be one of {list(SOURCE_KINDS)}")
            continue
        if kind == 'adc':
            pin = str(spec.get('channel', spec.get('pin', ''))).upper()
            if pin not in ADC_PINS:
                plan.errors.append(f"VOLTAGE_SOURCES[{name!r}]: channel must be one of {list(ADC_PINS)}")
                continue
            if pin in used_pins:
                plan.errors.append(f"VOLTAGE_SOURCES[{name!r}]: channel {pin} already used by {used_pins[pin]!r}")
                continue
            used_pins[pin] = name
            src = SourceSpec(name=name, kind='adc', channel=pin)
        else:
            addr = _parse_int(spec.get('i2c_address', spec.get('address')), 'i2c_address', name, plan)
            bus = _parse_int(spec.get('i2c_bus', 1), 'i2c_bus', name, plan)
            if addr is None or bus is None:
                continue
            try:
                gain = float(spec.get('gain', 1.0))
            except (TypeError, ValueError):
                plan.errors.append(f"VOLTAGE_SOURCES[{name!r}]: gain must be numeric")
                continue
            if (bus, addr) in used_addr:
                plan.errors.append(f"VOLTAGE_SOURCES[{name!r}]: bus {bus} address 0x{addr:02X} "
                                   f"already used by {used_addr[(bus, addr)]!r}")
                continue
            used_addr[(bus, addr)] = name
            src = SourceSpec(name=name, kind='eyespy', i2c_address=addr, i2c_bus=bus, gain=gain)
        seen_lower[name.lower()] = name
        plan.sources[name] = src


def _add_measurements(plan: OpticalPlan, od_cfg) -> None:
    if not isinstance(od_cfg, dict):
        plan.errors.append("OD_MEASUREMENTS must be a dict keyed by OD_45 / OD_ref / OD_90 / OD_135")
        return
    canonical = {n.lower(): n for n in OD_NAMES}
    wanted: Dict[str, str] = {}
    for raw_name, value in od_cfg.items():
        od_name = canonical.get(str(raw_name).lower())
        if od_name is None:
            plan.errors.append(f"OD_MEASUREMENTS: {raw_name!r} is not one of {list(OD_NAMES)}")
            continue
        enabled, source = _parse_measurement(value)
        if not enabled:
            continue
        if source is None:
            # True: the conventionally named source 'pd_<suffix>' (OD names and bare
            # suffixes are reserved, so nothing else could match)
            source = f"pd_{OD_SUFFIX[od_name]}"
            if source not in plan.sources:
                plan.errors.append(f"OD_MEASUREMENTS[{od_name!r}]: True needs a VOLTAGE_SOURCE named "
                                   f"{source!r}; give 'source' explicitly to use another name")
                continue
        source = str(source)
        if source not in plan.sources:
            plan.errors.append(f"OD_MEASUREMENTS[{od_name!r}]: unknown source {source!r} "
                               f"(known: {list(plan.sources)})")
            continue
        wanted[od_name] = source
    for od_name in OD_NAMES:                      # canonical order
        if od_name in wanted:
            plan.od[od_name] = wanted[od_name]


def _parse_measurement(value) -> Tuple[bool, Optional[str]]:
    """-> (enabled, source_name or None)"""
    if value is None or value is False:
        return False, None
    if value is True:
        return True, None
    if isinstance(value, str):
        return (True, value) if value.strip() else (False, None)
    if isinstance(value, dict):
        enabled = value.get('enabled', True)
        src = value.get('source')
        return (bool(enabled), (str(src) if src else None))
    return False, None


def _resolve_legacy(config, plan: OpticalPlan) -> OpticalPlan:
    """Legacy OD_ADC_CHANNELS / EYESPY_ADC configs -> plan, reproducing the pre-optics
    driver: names verbatim, every configured adc channel a column (even an invalid pin:
    it read NaN), every configured eyespy board a raw + voltage column, and components
    enabled WITHOUT their dict key initialised from the driver defaults but logged
    nothing (``logged=False``)."""
    plan.legacy = True
    init = _cfg(config, 'INIT_COMPONENTS', {}) or {}

    if init.get('optical_density', False):
        adc_channels = _cfg(config, 'OD_ADC_CHANNELS')
        logged = adc_channels is not None
        if adc_channels is None:
            adc_channels = {'Trx': 'A0', 'Ref': 'A1', 'Sct': 'A2'}   # components.init_optical_density default
        if isinstance(adc_channels, dict):
            for chan, pin in adc_channels.items():
                chan, pin = str(chan), str(pin).upper()
                if pin not in ADC_PINS:
                    plan.errors.append(f"OD_ADC_CHANNELS[{chan!r}]: invalid pin {pin!r} (channel is skipped at init, "
                                       "its column reads NaN)")
                    pin = None
                plan.sources[chan] = SourceSpec(name=chan, kind='adc', channel=pin, legacy_name=chan, logged=logged)
        else:
            plan.errors.append("OD_ADC_CHANNELS must be a dict of channel name -> 'A0'..'A3'")

    if init.get('eyespy_adc', False):
        boards = _cfg(config, 'EYESPY_ADC')
        logged = bool(boards)          # the old driver only labelled boards when the key was present and non-empty
        if not boards:
            boards = {'eyespy1': {'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0}}  # init_eyespy_adc default
        if isinstance(boards, dict):
            for bname, bcfg in boards.items():
                bname = str(bname)
                if not isinstance(bcfg, dict):
                    plan.errors.append(f"EYESPY_ADC[{bname!r}] must be a dict with i2c_address / i2c_bus / gain")
                    continue
                if bname in plan.sources:
                    plan.errors.append(f"EYESPY_ADC board {bname!r} collides with an OD_ADC_CHANNELS name; "
                                       "the board is left to init_eyespy_adc's config fallback")
                    continue
                try:
                    addr_raw = bcfg.get('i2c_address', 0x49)
                    addr = int(addr_raw, 0) if isinstance(addr_raw, str) else int(addr_raw)
                    bus = int(bcfg.get('i2c_bus', 1))
                    gain = float(bcfg.get('gain', 1.0))
                except (TypeError, ValueError) as e:
                    plan.errors.append(f"EYESPY_ADC[{bname!r}]: bad i2c_address / i2c_bus / gain ({e})")
                    continue
                plan.sources[bname] = SourceSpec(name=bname, kind='eyespy', i2c_address=addr, i2c_bus=bus,
                                                 gain=gain, legacy_name=bname, logged=logged)
        else:
            plan.errors.append("EYESPY_ADC must be a dict of board name -> {i2c_address, i2c_bus, gain}")

    # infer canonical OD measurements from adc channel names only
    for name, src in plan.sources.items():
        if src.kind != 'adc':
            continue
        od_name = _LEGACY_OD_BY_CHANNEL.get(name.lower())
        if od_name and od_name not in plan.od:
            plan.od[od_name] = name
    plan.od = {n: plan.od[n] for n in OD_NAMES if n in plan.od}

    ekf = _cfg(config, 'EKF_OD_CHANNEL')
    if ekf not in (None, _LEGACY_EKF_CHANNEL):
        plan.warnings.append(
            f"EKF_OD_CHANNEL={ekf!r} is ignored for legacy configs (the pre-optics driver never read it and "
            "always used '135'); define VOLTAGE_SOURCES + OD_MEASUREMENTS to make the EKF track it")
    if plan.sources_of('eyespy') and plan.sources_of('adc'):
        plan.warnings.append(
            "legacy config with eyespy boards: as before, the EKF reads 'eyespy_135_voltage' and stays idle "
            "unless a board is literally named '135'; migrate to OD_MEASUREMENTS to run the EKF on an OD measurement")
    return plan


def describe(plan: OpticalPlan) -> str:
    """One-line human summary for logs."""
    srcs = ', '.join(f"{s.name}({s.kind}:{s.channel if s.kind == 'adc' else hex(s.i2c_address or 0)})"
                     + ('' if s.logged else '[unlogged]')
                     for s in plan.sources.values()) or 'none'
    od = ', '.join(f"{k}<-{v}" for k, v in plan.od.items()) or 'none'
    return f"sources: {srcs}; OD: {od}{' [legacy config]' if plan.legacy else ''}"
