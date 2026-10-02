# This file is part of ts_eas.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["HvacModel", "HVAC_SLEEP_TIME"]

import asyncio
import logging
import math
from typing import Any, Callable

import yaml
from astropy.time import Time

from lsst.ts import salobj, utils
from lsst.ts.xml.enums.EAS import AHU
from lsst.ts.xml.enums.HVAC import DeviceId

from .cmdwrapper import close_command_tasks, command_wrapper
from .diurnal_timer import DiurnalTimer, get_local_noon_time
from .dome_model import DomeModel
from .weather_model import WeatherModel
from .weatherforecast_model import WeatherForecastModel

HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state (seconds)
STD_TIMEOUT = 5  # seconds
N_CHILLERS = 2  # EAS controls two HVAC glycol chillers.
N_AHUS = 4  # The HVAC system has four dome air handling units (AHUs/UMAs).
# After the dome closes (and EAS enables the AHUs), how long AHUs may report
# off while they start up before that counts toward the AHU-off catch-up.
AHU_STARTUP_TIME = 300.0  # seconds


class HvacModel:
    """A model for HVAC system automation.

    Parameters
    ----------
    log : `~logging.Logger`
        A logger for log messages.
    diurnal_timer : `DiurnalTimer`
        A timer that signals at noon, sunrise, and at the end of twilight.
    dome_model : `DomeModel`
        A model representing the dome state.
    weather_model : `WeatherModel`
        A model representing weather conditions.
    weatherforecast_model : `WeatherForecastModel`
        A model representing forecast conditions used for upcoming twilight.
    ahu_setpoint_delta : `float`
        The offset that will be added to the measured temperature in
        selecting a setpoint for the HVAC air handling units (AHUs/UMAs)
        measured in °C.
    ahu_setpoint_delta_closed_at_night : `float`
        The offset that will be added to the measured temperature in
        selecting a setpoint for the HVAC air handling units (AHUs/UMAs)
        during nighttime closed-dome operation, measured in °C.
    closed_at_night_setpoint_cadence : `float`, optional
        Cadence (s) at which the nighttime closed-dome AHU setpoint is
        reassessed. If ``None``, falls back to the default monitor sleep
        time (``HVAC_SLEEP_TIME``).
    ahu_control : `list`[`int`]
        The AHU numbers that EAS is allowed to control. Values correspond
        to AHUs 1 through 4.
    ahu_off_catchup_deltas : `list`[`float`]
        Four setpoint deltas (°C) applied to the daytime AHU setpoint while
        AHUs are off, indexed by the number of AHUs off minus one. The deepest
        delta reached is held until all four AHUs are back on.
    ahu_off_catchup_rate : `float`
        Once all AHUs are back on after some were off, the daytime AHU
        setpoint is lowered by this amount (°C) for each 1 °C the indoor
        temperature exceeds the base setpoint.
    ahu_off_catchup_poll_interval : `float`
        How often (s) that lowering is reassessed.
    ahu_off_catchup_threshold : `float`
        The excess (°C) of indoor temperature over the base setpoint above
        which that lowering applies.
    setpoint_lower_limit : `float`
        The minimum allowed setpoint for thermal control. If a lower setpoint
        than this is indicated from the ESS temperature readings, this setpoint
        will be used instead.
    wind_threshold : `float`
        Windspeed limit for the VEC-04 fan. (m/s)
    vec04_hold_time : `float`
        Minimum time to wait before changing the state of the VEC-04 fan. This
        value is ignored if the dome is opened or closed. (s)
    vec04_fan_frequency : `float`
        Rotation frequency commanded to the VEC-04 fan when it is enabled (Hz).
    glycol_band_low : `float`
        The lower bound (more negative) of the allowed difference between the
        average glycol setpoint and the ambient temperature (°C). This
        represents how far below ambient the setpoint is permitted to drift
        before being considered out of range. This number is expected (but not
        required) to be negative.
    glycol_band_high : `float`
        Upper bound of allowed glycol setpoint band relative to ambient (°C),
        corresponding to `glycol_band_low`. This number is expected (but not
        required) to be negative.
    glycol_average_offset : `float`
        Nominal average offset for glycol setpoints relative to ambient (°C).
        The average of the two glycol setpoints should differ from the
        ambient temperature reported by the ESS by this amount.
    glycol_dew_point_margin : `float`
        Safety margin (°C) added to the maximum dew point to avoid
        condensation.
    glycol_setpoints_delta : `float`
        Temperature difference (°C) between the two glycol chiller setpoints
        (chiller 1 warmer).
    glycol_absolute_minimum : `float`
        Absolute minimum setpoint (°C) allowed for the colder glycol chiller.
    glycol_absolute_maximum : `float`
        Absolute maximum setpoint (°C) allowed for the warmer glycol chiller.
    disable_features: `list[str]`
        A list of features that should be disabled. The following strings can
        be used:
         * vec04
         * ahu
         * room_setpoint
         * forecast
         * forecast_ahu
         * glycol_chillers
         * ahu_off_catchup
        Any other values are ignored.
    """

    def __init__(
        self,
        *,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        dome_model: DomeModel,
        weather_model: WeatherModel,
        weatherforecast_model: WeatherForecastModel,
        hvac_remote: salobj.Remote,
        ahu_setpoint_delta: float,
        ahu_setpoint_delta_closed_at_night: float,
        ahu_control: list[int],
        ahu_off_catchup_deltas: list[float],
        ahu_off_catchup_rate: float,
        ahu_off_catchup_poll_interval: float,
        ahu_off_catchup_threshold: float,
        setpoint_lower_limit: float,
        wind_threshold: float,
        vec04_hold_time: float,
        vec04_fan_frequency: float,
        glycol_band_low: float,
        glycol_band_high: float,
        glycol_average_offset: float,
        glycol_dew_point_margin: float,
        glycol_setpoints_delta: float,
        glycol_absolute_minimum: float,
        glycol_absolute_maximum: float,
        features_to_disable: list[str],
        forecast_ahu_setpoint_delta: float | None = None,
        closed_at_night_setpoint_cadence: float | None = None,
        allow_send: Callable[[], bool] | None = None,
    ) -> None:
        self.log = log
        self.allow_send = allow_send
        self.diurnal_timer = diurnal_timer

        self.monitor_start_event = asyncio.Event()

        self.last_vec04_time: float = 0  # Last time VEC-04 was changed (UNIX TAI seconds).

        # Latest workingState telemetry for AHUs 1-4 (None: none received).
        self.ahu_working_states: list[bool | None] = [None] * N_AHUS

        # The daytime AHU setpoint before any catch-up offset, set at sunrise
        # and by the forecast; None until known.
        self.base_ahu_setpoint: float | None = None

        # The last daytime AHU setpoint sent by update_ahu_setpoint, or None
        # if the AHUs may not hold it (they were disabled since).
        self.last_ahu_setpoint: float | None = None

        # When the dome was last seen to close (UNIX TAI seconds), or None
        # while it is open or its state is not yet known.
        self.dome_closed_since: float | None = None

        # The current AHU-off catch-up episode (see compute_catchup_offset).
        self.reset_catchup()

        # Configuration parameters:
        self.dome_model = dome_model
        self.weather_model = weather_model
        self.weatherforecast_model = weatherforecast_model
        self.ahu_setpoint_delta = ahu_setpoint_delta
        self.ahu_setpoint_delta_closed_at_night = ahu_setpoint_delta_closed_at_night
        self.ahu_off_catchup_deltas = ahu_off_catchup_deltas
        self.ahu_off_catchup_rate = ahu_off_catchup_rate
        self.ahu_off_catchup_poll_interval = ahu_off_catchup_poll_interval
        self.ahu_off_catchup_threshold = ahu_off_catchup_threshold
        self.closed_at_night_setpoint_cadence = (
            closed_at_night_setpoint_cadence
            if closed_at_night_setpoint_cadence is not None
            else HVAC_SLEEP_TIME
        )
        self.ahu_control = ahu_control
        self.setpoint_lower_limit = setpoint_lower_limit
        self.wind_threshold = wind_threshold
        self.vec04_hold_time = vec04_hold_time
        self.vec04_fan_frequency = vec04_fan_frequency
        self.features_to_disable = features_to_disable

        # Forecast-specific delta overrides
        # (fall back to standard values when absent):
        self.forecast_ahu_setpoint_delta = (
            forecast_ahu_setpoint_delta if forecast_ahu_setpoint_delta is not None else ahu_setpoint_delta
        )

        # Glycol chiller parameters:
        self.glycol_band_low = glycol_band_low
        self.glycol_band_high = glycol_band_high
        self.glycol_average_offset = glycol_average_offset
        self.glycol_dew_point_margin = glycol_dew_point_margin
        self.glycol_setpoints_delta = glycol_setpoints_delta
        self.glycol_absolute_minimum = glycol_absolute_minimum
        self.glycol_absolute_maximum = glycol_absolute_maximum

        # Glycol setpoints
        self.glycol_setpoint1: float | None = None
        self.glycol_setpoint2: float | None = None

        # The remote
        self.hvac_remote = hvac_remote

        self.twilight_forecast_callback_id: int | None = None

    def get_controlled_ahus(self) -> tuple[DeviceId, ...]:
        """Return the AHU device IDs configured for EAS control.

        Returns
        -------
        device_tuple : `tuple`[`DeviceId`, ...]
        """
        return tuple(DeviceId[AHU(ahu).name] for ahu in self.ahu_control)

    async def ahu_working_state_callback(self, ahu: int, data: salobj.BaseMsgType) -> None:
        """Record one AHU's workingState and update the daytime setpoint.

        This single callback is registered for all four
        ``HVAC.airHandlingUnit0<N>Dome`` telemetry topics; ``ahu`` (bound
        with :func:`functools.partial`) is the AHU number, 1-4.

        Parameters
        ----------
        ahu : `int`
            The AHU number (1-4) this telemetry sample is for.
        data : `~lsst.ts.salobj.BaseMsgType`
            A newly received airHandlingUnit telemetry item.
        """
        self.ahu_working_states[ahu - 1] = bool(data.workingState)
        await self.update_ahu_setpoint()

    def reset_catchup(self) -> None:
        """End any AHU-off catch-up episode."""
        # Stage 1: the most AHUs off at once in this episode.
        self.catchup_n_off = 0
        # Stage 2: active, its offset, and when it was last assessed.
        self.catchup_recovering = False
        self.catchup_recovery_offset = 0.0
        self.catchup_assessed_at: float | None = None

    def compute_catchup_offset(self) -> float:
        """Return the AHU-off catch-up offset for the current state.

        The catch-up applies during the day, once the dome has been closed
        for `AHU_STARTUP_TIME` (so AHUs that EAS disabled for an open dome,
        or that are still starting up after it re-closed, don't count as
        off). Outside that, the episode ends and the offset is zero.

        Stage 1: while any AHU is off, the offset is the
        ``ahu_off_catchup_deltas`` entry for the most AHUs off at once in
        this episode. Stage 2: once all are back on, the offset is
        ``-ahu_off_catchup_rate`` times the indoor temperature's excess over
        the base setpoint, reassessed every
        ``ahu_off_catchup_poll_interval``, until that excess is within
        ``ahu_off_catchup_threshold``.

        Returns
        -------
        offset : `float`
            The offset (°C, <= 0) to add to the base AHU setpoint.
        """
        now = utils.current_tai()
        if (
            "ahu_off_catchup" in self.features_to_disable
            or self.dome_closed_since is None
            or now - self.dome_closed_since < AHU_STARTUP_TIME
            or self.diurnal_timer.is_night(Time.now())
        ):
            self.reset_catchup()
            return 0.0

        n_off = sum(1 for state in self.ahu_working_states if state is False)
        if n_off > 0:
            if self.catchup_recovering:
                self.reset_catchup()
            self.catchup_n_off = max(self.catchup_n_off, n_off)
            return self.ahu_off_catchup_deltas[self.catchup_n_off - 1]

        if self.catchup_n_off > 0:
            # All AHUs are back on: start Stage 2, assessed at once.
            self.reset_catchup()
            self.catchup_recovering = True
        if not self.catchup_recovering:
            return 0.0

        if (
            self.catchup_assessed_at is None
            or now - self.catchup_assessed_at >= self.ahu_off_catchup_poll_interval
        ):
            indoor_temperature = self.weather_model.current_indoor_temperature
            if self.base_ahu_setpoint is not None and not math.isnan(indoor_temperature):
                self.catchup_assessed_at = now
                excess = indoor_temperature - max(self.base_ahu_setpoint, self.setpoint_lower_limit)
                if excess <= self.ahu_off_catchup_threshold:
                    self.reset_catchup()
                    return 0.0
                self.catchup_recovery_offset = -self.ahu_off_catchup_rate * excess
        return self.catchup_recovery_offset

    async def update_ahu_setpoint(self, force: bool = False) -> None:
        """Send the daytime AHU setpoint: the base plus the catch-up offset.

        This is the only sender of daytime AHU setpoints. It sends nothing
        until a base setpoint is known. Unless ``force`` is set (sunrise and
        the forecast, which set the base), it sends nothing at night, when
        `apply_setpoint_at_night` owns the setpoint, and only sends when the
        setpoint differs from the last one sent.

        Parameters
        ----------
        force : `bool`
            Send the setpoint even if it is unchanged, or if it is night.
        """
        offset = self.compute_catchup_offset()
        if self.base_ahu_setpoint is None:
            return
        if not force and self.diurnal_timer.is_night(Time.now()):
            return
        setpoint = max(self.base_ahu_setpoint + offset, self.setpoint_lower_limit)
        if not force and setpoint == self.last_ahu_setpoint:
            return
        self.last_ahu_setpoint = setpoint
        self.log.debug("Apply AHU setpoint %.2f (catch-up offset %.2f)", setpoint, offset)
        await self.config_lower_ahu(
            [
                {
                    "device_id": device_id,
                    "workingSetpoint": setpoint,
                    "maxFanSetpoint": math.nan,
                    "minFanSetpoint": math.nan,
                    "antiFreezeTemperature": math.nan,
                }
                for device_id in self.get_controlled_ahus()
            ]
        )

    @classmethod
    def get_config_schema(cls) -> str:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for EAS HVAC configuration.
type: object
properties:
  ahu_setpoint_delta:
    type: number
    default: -1.0
    description: >-
      The offset that will be applied to the measured temperature in
      selecting a setpoint for the HVAC air handling units (AHUs/UMAs)
      measured in °C.
  ahu_setpoint_delta_closed_at_night:
    type: number
    default: -1.0
    description: >-
      The offset that will be applied to the measured temperature in
      selecting a setpoint for the HVAC air handling units (AHUs/UMAs)
      during nighttime closed-dome operation measured in °C.
  ahu_control:
    type: array
    default: [1, 2, 3, 4]
    description: >-
      AHU numbers that EAS is allowed to control. These numbers refer
      to the devices `airHandlingUnit01Dome` - `airHandlingUnit04Dome` in
      :class:`~lsst.ts.xml.enums.HVAC.DeviceId`.
    items:
      type: integer
      enum: [1, 2, 3, 4]
    uniqueItems: true
  ahu_off_catchup_deltas:
    type: array
    default: [0.0, -1.0, -2.0, -3.0]
    description: >-
      Setpoint deltas (°C) applied to the daytime AHU setpoint while AHUs are
      off with the dome closed, to help daytime regulation catch up. The
      elements are used when one, two, three, and four AHUs are off,
      respectively. The deepest delta reached is held until all four AHUs are
      back on.
    items:
      type: number
    minItems: 4
    maxItems: 4
  ahu_off_catchup_rate:
    type: number
    default: 1.0
    description: >-
      Part of the AHU-off daytime catch-up logic. Once all AHUs are back on
      after some were off, the daytime AHU setpoint is lowered by this amount
      (°C) for each 1 °C the indoor (ESS) temperature exceeds the base
      setpoint, down to setpoint_lower_limit.
  ahu_off_catchup_poll_interval:
    type: number
    default: 900.0
    exclusiveMinimum: 0
    description: >-
      Part of the AHU-off daytime catch-up logic. How often (s) the lowering
      applied once all AHUs are back on is reassessed.
  ahu_off_catchup_threshold:
    type: number
    default: 1.0
    description: >-
      Part of the AHU-off daytime catch-up logic. Excess (°C) of the indoor
      temperature over the base setpoint above which the lowering applied
      once all AHUs are back on continues.
  setpoint_lower_limit:
    type: number
    default: 6.0
    description: >-
      The minimum allowed setpoint for thermal control. If a lower setpoint
      than this is indicated from the ESS temperature readings, this setpoint
      will be used instead.
  wind_threshold:
    type: number
    default: 10.0
    description: Windspeed limit for the VEC-04 fan (m/s).
  vec04_hold_time:
    type: number
    default: 300.0
    description: >-
      Minimum time to wait before changing the state of the VEC-04 fan. This
      value is ignored if the dome is opened or closed (s).
  vec04_fan_frequency:
    type: number
    default: 55.0
    description: >-
      Rotation frequency commanded to the VEC-04 fan when it is enabled (Hz).
  glycol_band_low:
    type: number
    default: -10.0
    description: Lower bound of allowed glycol setpoint band relative to ambient (°C).
  glycol_band_high:
    type: number
    default: -5.0
    description: Upper bound of allowed glycol setpoint band relative to ambient (°C).
  glycol_average_offset:
    type: number
    default: -7.5
    description: Nominal average offset for glycol setpoints relative to ambient (°C).
  glycol_dew_point_margin:
    type: number
    default: 1.0
    description: Safety margin (°C) added to the maximum dew point to avoid condensation.
  glycol_setpoints_delta:
    type: number
    default: 1.0
    description: Temperature difference (°C) between the two glycol chiller setpoints (chiller 1 warmer).
  glycol_absolute_minimum:
    type: number
    default: -10.0
    description: Absolute minimum setpoint (°C) allowed for the colder glycol chiller.
  glycol_absolute_maximum:
    type: number
    default: 10.0
    description: Absolute maximum setpoint (°C) allowed for the warmer glycol chiller.
  forecast_ahu_setpoint_delta:
    type: [number, "null"]
    default: null
    description: >-
      AHU setpoint offset (°C) used when driven by forecast. If absent,
      ahu_setpoint_delta is used.
  closed_at_night_setpoint_cadence:
    type: [number, "null"]
    default: null
    exclusiveMinimum: 0
    description: >-
      Cadence (s) at which the nighttime closed-dome AHU setpoint is
      reassessed. If absent, the default monitor sleep time is used.
required:
  - ahu_setpoint_delta
  - setpoint_lower_limit
  - wind_threshold
  - vec04_hold_time
  - glycol_band_low
  - glycol_band_high
  - glycol_average_offset
  - glycol_dew_point_margin
  - glycol_setpoints_delta
  - glycol_absolute_minimum
  - glycol_absolute_maximum
additionalProperties: false
"""
        )

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_enableDevice")
    async def enable_devices(self, device_ids: list[DeviceId]) -> list[dict[str, Any]] | None:
        if not device_ids:
            return None
        return [{"device_id": device_id} for device_id in device_ids]

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_disableDevice")
    async def disable_devices(self, device_ids: list[DeviceId]) -> list[dict[str, Any]] | None:
        if not device_ids:
            return None
        return [{"device_id": device_id} for device_id in device_ids]

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_configLowerAhu")
    async def config_lower_ahu(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not commands:
            return None
        return commands

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_configChiller")
    async def config_chiller(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not commands:
            return None
        return commands

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_configFan")
    async def config_fan(self, frequency: float) -> dict[str, Any]:
        return {"device_id": DeviceId.airExtractionFan04Dome, "frequency": frequency}

    async def monitor(self) -> None:
        """Monitor the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        self.log.debug("HvacModel.monitor")

        # Give the dome model an opportunity to collect some telemetry...
        await asyncio.sleep(STD_TIMEOUT)

        tasks = [
            self.control_ahus_and_vec04(),
            self.wait_for_sunrise(),
            self.monitor_twilight_forecast(),
            self.apply_setpoint_at_night(),
            self.adjust_glycol_chillers_at_noon(),
            self.monitor_glycol_chillers(),
        ]

        hvac_future = asyncio.gather(*tasks)
        self.monitor_start_event.set()

        try:
            await hvac_future
        except asyncio.CancelledError:
            hvac_future.cancel()
            await asyncio.gather(hvac_future, return_exceptions=True)
            raise
        finally:
            self.monitor_start_event.clear()

    async def close(self) -> None:
        """Cancel any in-flight command tasks."""
        self.clear_twilight_forecast_callback()
        await close_command_tasks(self)

    async def control_ahus_and_vec04(self) -> None:
        cached_shutter_closed = None
        cached_wind_threshold = None

        while True:
            # Check the aperture state
            shutter_closed = self.dome_model.is_closed
            if shutter_closed is None:
                await asyncio.sleep(0.1)
                continue
            elif shutter_closed and not cached_shutter_closed:
                # Reset the cache and time of previous VEC-04 operation
                # so that action will be taken on dome re-open.
                cached_wind_threshold = None
                self.last_vec04_time = 0

            enable_device_list = []
            disable_device_list = []

            if (
                "vec04" not in self.features_to_disable
                and not shutter_closed
                and (utils.current_tai() - self.last_vec04_time > self.vec04_hold_time)
            ):
                # Check windspeed threshold
                average_windspeed = self.weather_model.average_windspeed
                wind_threshold = average_windspeed < self.wind_threshold
                if wind_threshold != cached_wind_threshold:
                    change_message = f"VEC-04 operation demanded: {average_windspeed} m/s -> {wind_threshold}"

                    cached_wind_threshold = wind_threshold
                    self.last_vec04_time = utils.current_tai()
                    if wind_threshold:
                        self.log.info(f"Turning on VEC-04 fan! {change_message}")
                        await self.config_fan(self.vec04_fan_frequency)
                        enable_device_list.append(DeviceId.airExtractionFan04Dome)
                    else:
                        self.log.info(f"Turning off VEC-04 fan! {change_message}")
                        disable_device_list.append(DeviceId.airExtractionFan04Dome)

            if shutter_closed != cached_shutter_closed:
                cached_shutter_closed = shutter_closed
                ahus = self.get_controlled_ahus()
                if shutter_closed:
                    self.dome_closed_since = utils.current_tai()
                    # The AHUs were disabled while the dome was open and may
                    # not hold the last setpoint sent, so send it again below.
                    self.last_ahu_setpoint = None
                    if "ahu" not in self.features_to_disable:
                        # Enable the configured AHUs
                        self.log.info("Enabling HVAC AHUs!")
                        enable_device_list.extend(ahus)

                    if "vec04" not in self.features_to_disable:
                        # Disable the VEC-04 fan
                        self.log.info("Turning off VEC-04 fan!")
                        disable_device_list.append(DeviceId.airExtractionFan04Dome)
                        self.last_vec04_time = utils.current_tai()
                else:
                    # With the dome open the catch-up doesn't apply: send the
                    # base setpoint before the AHUs are disabled below.
                    self.dome_closed_since = None
                    await self.update_ahu_setpoint()
                    if "ahu" not in self.features_to_disable:
                        self.log.info("Disabling HVAC AHUs!")
                        disable_device_list.extend(ahus)

            if disable_device_list:
                await self.disable_devices(disable_device_list)
            if enable_device_list:
                await self.enable_devices(enable_device_list)
            # After any enable, and every pass (which keeps the catch-up's
            # periodic reassessment going without AHU telemetry).
            await self.update_ahu_setpoint()
            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def wait_for_sunrise(self) -> None:
        """Wait for sunrise and then set the room temperature.

        Wait for the timer to signal sunrise, and then obtain the
        temperature that was reported last night at the end
        of twilight, and then apply that temperature as at AHU
        setpoint.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.sunrise_condition:
                await self.diurnal_timer.sunrise_condition.wait()
                last_twilight_temperature = await self.weather_model.get_last_twilight_temperature()
                if self.diurnal_timer.is_running and last_twilight_temperature is not None:
                    if "room_setpoint" not in self.features_to_disable:
                        # Time to set the room setpoint based on last twilight
                        self.base_ahu_setpoint = last_twilight_temperature + self.ahu_setpoint_delta
                        await self.update_ahu_setpoint(force=True)

    def clear_twilight_forecast_callback(self) -> None:
        if self.twilight_forecast_callback_id is None:
            return
        self.weatherforecast_model.remove_callback(self.twilight_forecast_callback_id)
        self.twilight_forecast_callback_id = None

    def set_twilight_forecast_callback(self) -> None:
        self.clear_twilight_forecast_callback()
        twilight_time = self.diurnal_timer.get_twilight_time(after=Time.now())
        callback_id = self.weatherforecast_model.add_callback(
            twilight_time.tai.unix,
            self.handle_twilight_forecast,
        )
        self.twilight_forecast_callback_id = callback_id

    def handle_twilight_forecast(self, predicted_temperature: float) -> None:
        if not self.diurnal_timer.is_running:
            return
        if "room_setpoint" in self.features_to_disable or "forecast" in self.features_to_disable:
            return
        self.log.info(
            f"Applying HVAC setpoints based on forecast twilight temperature: {predicted_temperature:.2f}°C"
        )
        asyncio.create_task(self.apply_forecast_setpoints(predicted_temperature))

    async def apply_forecast_setpoints(self, predicted_temperature: float) -> None:
        if "room_setpoint" not in self.features_to_disable and "forecast_ahu" not in self.features_to_disable:
            self.base_ahu_setpoint = predicted_temperature + self.forecast_ahu_setpoint_delta
            await self.update_ahu_setpoint(force=True)

    async def monitor_twilight_forecast(self) -> None:
        """Run forecast callback between noon and evening twilight."""
        # If started between noon and twilight, begin operating immediately
        # without waiting for noon.
        next_noon = get_local_noon_time()
        if (
            self.diurnal_timer.is_running
            and self.diurnal_timer.twilight_time is not None
            and self.diurnal_timer.twilight_time < next_noon
        ):
            self.set_twilight_forecast_callback()
            async with self.diurnal_timer.twilight_condition:
                await self.diurnal_timer.twilight_condition.wait()
            self.clear_twilight_forecast_callback()

        # Subsequently, wait for noon and start, then wait for
        # twilight and stop, in a loop.
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                await self.diurnal_timer.noon_condition.wait()
            if not self.diurnal_timer.is_running:
                break
            self.set_twilight_forecast_callback()
            async with self.diurnal_timer.twilight_condition:
                await self.diurnal_timer.twilight_condition.wait()
            self.clear_twilight_forecast_callback()

    def compute_glycol_setpoints(
        self, ambient_temperature: float, average_offset: float | None = None
    ) -> tuple[float, float]:
        """Compute staggered glycol chiller setpoints.

        Compute staggered glycol chiller setpoints based on ambient
        temperature, configured band limits, dew point, and absolute
        minimum and maximum constraint.

        The algorithm enforces:
          * Average setpoint nominally `ambient + glycol_average_offset`
            (float, with `glycol_average_offset` being negative).
          * Raised if necessary to exceed the nightly maximum indoor dew
            point plus a safety margin (`dew_point_margin`: `float`).
          * Split into two staggered setpoints
            (`setpoint1`: `float`, `setpoint2`: `float`)
            separated by `glycol_setpoints_delta`: `float`, **with chiller 1
            warmer**.
          * Absolute minimum enforced on the colder chiller
            (`setpoint2` : `float` >= `glycol_absolute_minimum`: `float`),
            adjusting both setpoints to preserve the delta.
          * Similar constraints for the absolute maximum temperature.

        Parameters
        ----------
        ambient_temperature : `float`
            Ambient temperature in degrees Celsius. This value
            is used to determine the nominal target band for the glycol
            loop average.

        Returns
        -------
        setpoint1 : `float`
            Active setpoint for chiller 1 (°C), the warmer of the two.
        setpoint2 : `float`
            Active setpoint for chiller 2 (°C), the colder of the two.
        """
        # Compute a target average setpoint
        if average_offset is None:
            average_offset = self.glycol_average_offset
        target_average = ambient_temperature + average_offset

        # Incorporate dew point into the calculation - setpoint
        # average should not be lower than the dew point (with margin)
        nightly_maximum_dew_point = self.weather_model.nightly_maximum_indoor_dew_point
        if nightly_maximum_dew_point is not None:
            target_average = max(target_average, nightly_maximum_dew_point + self.glycol_dew_point_margin)

        # Break average and delta into individual setpoints. The two
        # setpoints should have the computed average and differ
        # with each other by `glycol_setpoints_delta`.
        setpoint1, setpoint2 = target_average, target_average
        setpoint1 += self.glycol_setpoints_delta / N_CHILLERS
        setpoint2 -= self.glycol_setpoints_delta / N_CHILLERS

        # Enforce the absolute minimum temperature
        if setpoint2 < self.glycol_absolute_minimum:
            setpoint2 = self.glycol_absolute_minimum
            setpoint1 = setpoint2 + self.glycol_setpoints_delta

        # Enforce the absolute maximum temperature
        if setpoint1 > self.glycol_absolute_maximum:
            setpoint1 = self.glycol_absolute_maximum
            setpoint2 = setpoint1 - self.glycol_setpoints_delta

        return setpoint1, setpoint2

    def check_glycol_setpoint(self, ambient_temperature: float) -> bool:
        """Verify whether the chiller setpoints are within the allowed band.

        Parameters
        ----------
        ambient_temperature : `float`
            Ambient temperature (°C)

        Returns
        -------
        bool
            True if the current setpoints are acceptable, or False otherwise.
        """
        # This test makes mypy happy:
        if self.glycol_setpoint1 is None or self.glycol_setpoint2 is None:
            return False

        # Find the average of the two glycol setpoints.
        average_setpoint = (self.glycol_setpoint1 + self.glycol_setpoint2) / N_CHILLERS

        # The difference between the average and the current ambient
        # reading must not fall below `glycol_band_low` or above
        # `glycol_band_high`.
        return self.glycol_band_low <= average_setpoint - ambient_temperature <= self.glycol_band_high

    async def monitor_glycol_chillers(self) -> None:
        """Continuously monitor and enforce glycol chiller setpoints.

        This coroutine runs while the diurnal timer is active. On each cycle it
        checks whether the current chiller setpoints are within the allowed
        band relative to the ambient indoor temperature. If the setpoints are
        outside of the band, new setpoints are computed and applied to the
        HVAC CSC.
        """
        self.log.debug("monitor_glycol_chillers")
        while self.diurnal_timer.is_running:
            try:
                if "glycol_chillers" in self.features_to_disable:
                    await asyncio.sleep(HVAC_SLEEP_TIME)
                    continue

                # After the setpoints are chosen at noon, monitor
                # the system and adjust setpoints if needed.
                ambient_temperature = self.weather_model.current_indoor_temperature
                if ambient_temperature is not None and not self.check_glycol_setpoint(ambient_temperature):
                    self.log.debug("Recomputing glycol setpoints.")
                    glycol_setpoint1, glycol_setpoint2 = self.compute_glycol_setpoints(ambient_temperature)

                    if all(
                        (
                            glycol_setpoint1 is not None,
                            not math.isnan(glycol_setpoint1),
                            glycol_setpoint2 is not None,
                            not math.isnan(glycol_setpoint2),
                        )
                    ):
                        self.glycol_setpoint1 = glycol_setpoint1
                        self.glycol_setpoint2 = glycol_setpoint2

                chiller_commands = []
                if self.glycol_setpoint1 is not None:
                    chiller_commands.append(
                        {
                            "device_id": DeviceId.coldGlycolChiller01,
                            "activeSetpoint": self.glycol_setpoint1,
                        }
                    )
                if self.glycol_setpoint2 is not None:
                    chiller_commands.append(
                        {
                            "device_id": DeviceId.coldGlycolChiller02,
                            "activeSetpoint": self.glycol_setpoint2,
                        }
                    )
                if chiller_commands:
                    await self.config_chiller(chiller_commands)
            except Exception:
                self.log.exception("In HVAC glycol control loop")

            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def adjust_glycol_chillers_at_noon(self) -> None:
        """Wait for noon and then sets the glycol chillers.

        Wait for the timer to signal noon, and then obtain the minimum
        temperature that was reported last night, and then apply an
        appropriate temperature as the glycol setpoint.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                await self.diurnal_timer.noon_condition.wait()
                if not self.diurnal_timer.is_running:
                    return
                if "glycol_chillers" in self.features_to_disable:
                    continue

                nightly_minimum_temperature = self.weather_model.nightly_minimum_temperature
                if math.isnan(nightly_minimum_temperature):
                    self.log.error("Nightly minimum temperature was not available.")
                    continue

                self.glycol_setpoint1, self.glycol_setpoint2 = self.compute_glycol_setpoints(
                    nightly_minimum_temperature
                )

                if self.glycol_setpoint1 is None or self.glycol_setpoint2 is None:
                    self.log.error("Failed to calculate noon glycol setpoints.")
                    continue

                await self.config_chiller(
                    [
                        {
                            "device_id": DeviceId.coldGlycolChiller01,
                            "activeSetpoint": self.glycol_setpoint1,
                        },
                        {
                            "device_id": DeviceId.coldGlycolChiller02,
                            "activeSetpoint": self.glycol_setpoint2,
                        },
                    ]
                )

    async def apply_setpoint_at_night(self) -> None:
        """Control the HVAC setpoint during the night.

        At night time (defined by `DiurnalTimer.is_night`) the HVAC
        AHU setpoint should be applied based on the outside temperature,
        if the dome is closed. If the dome is open, the HVAC AHUs
        should not be enabled, and the setpoint should not matter.
        """
        warned_no_temperature = False

        while self.diurnal_timer.is_running:
            if "closed_at_night" in self.features_to_disable:
                await asyncio.sleep(self.closed_at_night_setpoint_cadence)
                continue

            if self.diurnal_timer.is_night(Time.now()) and self.dome_model.is_closed:
                if "room_setpoint" in self.features_to_disable:
                    await asyncio.sleep(self.closed_at_night_setpoint_cadence)
                    continue

                setpoint = max(
                    self.weather_model.current_temperature + self.ahu_setpoint_delta_closed_at_night,
                    self.setpoint_lower_limit,
                )
                if math.isnan(setpoint):
                    if not warned_no_temperature:
                        self.log.warning("Failed to collect a temperature sample for HVAC setpoint.")
                        warned_no_temperature = True

                else:
                    # Apply setpoint for each configured AHU
                    await self.config_lower_ahu(
                        [
                            {
                                "device_id": device_id,
                                "workingSetpoint": setpoint,
                                "maxFanSetpoint": math.nan,
                                "minFanSetpoint": math.nan,
                                "antiFreezeTemperature": math.nan,
                            }
                            for device_id in self.get_controlled_ahus()
                        ]
                    )

            await asyncio.sleep(self.closed_at_night_setpoint_cadence)
