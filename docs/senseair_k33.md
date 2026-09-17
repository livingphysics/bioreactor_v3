# Senseair K33: dedicated I²C bus

For this bioreactor, connect the Senseair K33 to its own I²C bus. Keep Atlas O₂,
Atlas CO₂ (where fitted), optics and the other I²C devices on bus 1.
The standard K33 connection is `/dev/i2c-3`, SDA on GPIO23 and SCL on GPIO24.
Both standalone `Bioreactor` scripts and the API use the same sensor bus setting.

## Wiring and power

| K33 connection | Destination |
|---|---|
| SDA | Pi GPIO23, physical header pin 16 |
| SCL | Pi GPIO24, physical header pin 18 |
| G0 / GND | Main supply negative and Pi ground (for example physical pin 6) |
| G+ / main power input | Separate DC supply rated for the exact K33 variant |
| DVCC | Leave externally unconnected; do not join to Pi 3.3 V |

The main sensor supply and I²C voltage are different. Do **not** power the K33
main input from Pi 3.3 V or put its higher supply voltage on SDA/SCL. The
[K33 ICB specification](https://senseair.jp/wp-content/uploads/2019/06/K33-ICB-Product-specification-PSP141.pdf)
lists 5–14 V supply limits; check the specification for your particular module.
DVCC is an output from the sensor's internal 3.3 V regulator, not its main power
input. It must not be tied directly to another regulator output.

The sensor has internal SDA/SCL pull-ups to DVCC (56 kΩ, per the
[Senseair I²C guide](https://senseair.jp/wp-content/uploads/2019/12/I2C-Communication-guide-TDE4700.pdf)).
No breakout or additional pull-up resistors are part of the direct wiring above.
These internal pull-ups are weak: cable length and capacitance can affect rise
time, so a successful short test does not qualify every installation. If extra
pull-ups are needed, use 3.3 V logic levels and account for all existing resistors.
Connecting a pull-up resistor to a supply is different from directly joining
the two supply outputs.

GPIO23/24 must be unused by other equipment. Do not bridge their SDA/SCL wires
back onto GPIO2/3. Atlas O₂ can retain its existing Pi 3.3 V supply and bus 1.

## Persistent Raspberry Pi setup

Stop the API and any standalone readers before changing wiring or bus setup.
Back up `/boot/firmware/config.txt` (older systems may use `/boot/config.txt`).
Under an `[all]` section, add the following **once**:

```ini
# Dedicated Senseair K33 bus; BCM GPIO numbering
gpio=23,24=ip,pn
dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24,i2c_gpio_delay_us=2
```

`ip,pn` makes the pins inputs without internal pulls so the sensor's pull-ups
set the idle level. The overlay provides software I²C; delay 2 is approximately
100 kHz according to the Pi overlay help. Actual timing depends on the platform.
See [Pi config.txt documentation](https://www.raspberrypi.com/documentation/computers/config_txt.html).

After a coordinated reboot, verify:

```bash
ls -l /dev/i2c-3
pinctrl get 23-24  # Pi 5: both lines should idle high with the powered K33 attached
cat /sys/bus/i2c/devices/i2c-3/name
```

Keep the API stopped during diagnostic reads. Normal reactor initialization can
briefly exercise enabled pumps, so do not instantiate a complete reactor merely
to check this sensor. A runtime `sudo dtoverlay i2c-gpio ...` command alone is
temporary and does not survive reboot. The Python class does not create the bus.

## Python / API configuration

Set the following inside the rig's `Config` class:

```python
CO2_SENSOR_TYPE = 'sensair_k33'
CO2_SENSOR_I2C_ADDRESS = 0x68
CO2_SENSOR_I2C_BUS = 3
O2_SENSOR_I2C_BUS = 1
```

Enable `INIT_COMPONENTS['co2_sensor']` as needed. For standalone use, edit
`bioreactor_v3/src/config.py`. For the API, edit `bioreactor-api/config.py`;
the submodule's `src/config.py` should symlink to it. Do not replace a rig's
whole config with the defaults: that would discard its other calibrations.

`config_default.py` uses `CO2_SENSOR_I2C_BUS = None`: the initializer resolves
this to 3 for `sensair` / `sensair_k33`, or 1 for Atlas. A missing attribute
uses the same rule. Explicit integers remain supported for custom wiring;
an existing K33 config specifying 1 must be migrated manually. There is no
automatic fallback to bus 1 if bus 3 is absent or the K33 does not respond.
The API's K33 template explicitly specifies 3 and also works with older driver
versions that already accept a bus number.

The initializer probes the selected bus and stores it in
`reactor.co2_sensor_config`; `io.read_co2()` uses that bus on every read. No
additional change to the `Bioreactor` class or API endpoints is necessary.
Legacy bench utilities may have their own bus defaults; pass bus 3 explicitly
instead of assuming that they load the rig's config. A generic `i2cdetect`
scan is not a reliable K33 presence test: use its ReadRAM request/response.

## Why separate buses?

The dedicated bus is a deployment requirement for this rig's K33/Atlas
combination, not a claim that every K33 must be isolated in every product.
Shared-bus troubleshooting reproduced missing readings with SDA held low, even
when the K33 was connected but unpolled, and with a replacement Atlas O₂ unit.
Removing the direct DVCC-to-Pi-3.3-V connection alone did not resolve the fault.

Direct wiring to GPIO23/24 with separate K33 main power, DVCC disconnected and
no breakout subsequently passed a five-minute two-sensor check. This supports
the separate-bus arrangement but does not prove long-term reliability or identify
the exact electrical/protocol cause; a reboot also occurred between tests.
Run a longer soak test under the intended workload before relying on automatic
CO₂ dosing. Keep all other controller calibration and validation requirements.
