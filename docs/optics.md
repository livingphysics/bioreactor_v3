# Optical inputs: voltage sources and OD measurements

Two layers sit between the ADCs and everything that consumes "optical density"
(the CSV, the EKF, the API, the dashboard). Both are resolved once, at
`Bioreactor()` construction, by `src/optics.py` into `bioreactor.optics`.

## 1. Voltage sources

Every photodiode/ADC input the rig has, under a name of your choosing.

| kind     | hardware                          | component that must be enabled | per-source settings                 |
|----------|-----------------------------------|--------------------------------|-------------------------------------|
| `adc`    | one channel of the ADS1115 (A0–A3) | `INIT_COMPONENTS['optical_density']` | `channel`                            |
| `eyespy` | one ADS1114 "eyespy" board         | `INIT_COMPONENTS['eyespy_adc']`      | `i2c_address`, `i2c_bus`, `gain`     |

```python
VOLTAGE_SOURCES = {
    'pd_135':  {'kind': 'adc', 'channel': 'A0'},
    'pd_ref':  {'kind': 'adc', 'channel': 'A1'},
    'pd_90':   'adc:A2',                                  # shorthand
    'eyespy1': {'kind': 'eyespy', 'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0},
    'eyespy2': 'eyespy:0x4a',                             # shorthand (bus 1, gain 1.0)
}
```

Names: letters, digits, `_`, `-`, starting with a letter; unique ignoring case; not a
component or API path word (`led`, `state`, `all`, ...) and not an OD measurement name or
suffix (`OD_135`, `135`, `ref`, ...). Two sources may not share a channel or an I2C address.

What a source gives you:

- `io.read_voltage(bioreactor, name)` — an un-gated reading (volts) of either kind.
- `io.read_all_voltages(bioreactor)` — `{name: volts | None}` for every source.
- `io.measure_od(bioreactor, led_power, duration, channel_name='all')` — IR-gated,
  averaged readings for every source, `{name: volts}`.
- In the API: `GET /api/voltage/<name>` and `GET /api/voltages`.
- A CSV column `<name>_V` **when no OD measurement consumes it** (a mapped source is
  logged under its OD column instead, so nothing is written twice).

## 2. OD measurements

Exactly four canonical measurements: `OD_45`, `OD_ref`, `OD_90`, `OD_135`. Each is
enabled or not and, when enabled, fed by one voltage source of either kind.

```python
OD_MEASUREMENTS = {
    'OD_45':  {'enabled': False},
    'OD_ref': {'enabled': True, 'source': 'pd_ref'},
    'OD_90':  'pd_90',        # shorthand: enabled, that source
    'OD_135': 'eyespy1',      # an eyespy board can feed an OD measurement too
}
```

Value shorthands: a source name (enabled), `False`/`None` (disabled), `True`
(enabled, fed by the source named `pd_<x>`, e.g. `pd_135`; it is an error if no such
source exists). Any other key than the four names is a config error.

What an enabled measurement gives you:

- CSV columns `OD_45_V`, `OD_ref_V`, `OD_90_V`, `OD_135_V` (in that order), from the
  same IR-gated read as every other source.
- `io.read_od(bioreactor, 'OD_135')` — un-gated; `measure_od(..., channel_name='OD_135')`
  — gated float.
- The `sensor_data` keys `od_45`, `od_ref`, `od_90`, `od_135` inside
  `measure_and_record_sensors`, which is what the EKF reads.
- In the API: `GET /api/od/state`, the `od` field of `/api/state`, the history archive,
  and the dashboard's OD plot, which shows OD measurements only.

`EKF_OD_CHANNEL` accepts an OD measurement name (`'OD_135'`), its suffix (`'135'`,
`'ref'`), or a voltage source name (resolved to the OD measurement it feeds, else its own
`<name>_V` column). A disabled measurement or an unknown name means no EKF input.

Config mistakes never abort a run: they are logged as `Optical config: ...` and the
offending source/measurement is dropped. A source whose hardware component is not
initialized reads as `None`/NaN and its columns are omitted from the CSV.

## 3. Legacy configs keep working unchanged

A config that defines **neither** `VOLTAGE_SOURCES` nor `OD_MEASUREMENTS` is resolved
from the historical keys, and the plan is flagged `legacy`:

| legacy config                                   | becomes                                   | CSV columns (unchanged)                       |
|-------------------------------------------------|-------------------------------------------|-----------------------------------------------|
| `OD_ADC_CHANNELS = {'135': 'A0', 'Ref': 'A1', '90': 'A2'}` | adc sources `135`, `Ref`, `90`; `OD_135←135`, `OD_ref←Ref`, `OD_90←90` | `OD_135_V`, `OD_Ref_V`, `OD_90_V` |
| `OD_ADC_CHANNELS = {'Trx': 'A0', 'Sct': 'A2'}`  | plain adc sources (no OD measurement)      | `OD_Trx_V`, `OD_Sct_V`                        |
| `EYESPY_ADC = {'sct': {...}}`                   | eyespy source `sct` (never an OD measurement) | `Eyespy_sct_raw`, `Eyespy_sct_V`           |

Everything else the old driver did is reproduced for legacy plans, deliberately
including its quirks, so an upgraded rig's CSV does not change:

- attribute names (`bioreactor.od_channels`, `bioreactor.eyespy_boards`), the
  `measure_od(...)` return shapes (a single name reads an ADS1115 channel and every eyespy
  board rides along in a dict), `SENSOR_LABELS` overrides including the old `od_Ref` /
  `od_REF` spellings;
- an ADC channel with an invalid pin keeps its column (cells are NaN); an eyespy board
  that does not answer its bus probe keeps its header columns with empty cells; a
  component enabled without its dict key is initialised from the driver defaults
  (`Trx`/`Ref`/`Sct`, `eyespy1`) but logs no columns;
- **the EKF**: the old driver never managed to read `EKF_OD_CHANNEL` (it looked for an
  attribute that did not exist), so it always used `'135'`, and an initialised eyespy
  component took precedence, which left the EKF idle on every rig with eyespy boards.
  Legacy plans keep exactly that; the driver logs a warning saying so. Define
  `VOLTAGE_SOURCES` + `OD_MEASUREMENTS` to make the EKF track a chosen measurement.

Defining only one of the two new keys is reported as an error and treated as legacy.
Config mistakes in either form are logged as `Optical config: ...`, never raised.

## 4. Where the pieces live

| file                         | role                                                            |
|------------------------------|-----------------------------------------------------------------|
| `src/optics.py`              | `resolve_optical_config(config) -> OpticalPlan`; keys, labels, EKF resolution. Pure Python. |
| `src/bioreactor.py`          | resolves the plan, derives `SENSOR_LABELS` / `fieldnames` / `optical_columns` |
| `src/components.py`          | `init_optical_density` / `init_eyespy_adc` build their maps from the plan |
| `src/io.py`                  | `read_voltage` (both kinds), `read_od`, `read_all_voltages`, `measure_od` |
| `src/utils.py`               | `measure_and_record_sensors` writes OD/source columns; EKF channel via the plan |
| `test_optics.py` (repo root) | hardware-free tests: `python3 -m unittest test_optics`           |
