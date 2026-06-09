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
        Four setpoint deltas (°C) applied to the AHU working setpoint while
        AHUs are off, indexed by the number of AHUs off minus one (one off uses
        the first element, ... four off the fourth). The deepest delta reached
        is held until all four AHUs are back on, then released to zero.
    ahu_off_catchup_rate : `float`
        Part of the AHU-off daytime catch-up logic: once all AHUs are back on
        after some were turned off, this is the amount (°C) by which the
        daytime AHU setpoint is lowered for each 1 °C the ambient (indoor ESS)
        temperature exceeds the base setpoint, helping the dome catch up after
        falling behind while the AHUs were off.
    ahu_off_catchup_poll_interval : `float`
        Part of the AHU-off daytime catch-up logic: the cadence (s) at which
        the ambient-overshoot catch-up (applied once all AHUs are back on) is
        reassessed.
    ahu_off_catchup_threshold : `float`
        Part of the AHU-off daytime catch-up logic: the ambient excess (°C)
        above the base setpoint above which the ambient-overshoot catch-up
        (applied once all AHUs are back on) begins.
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

        # Most recent workingState (on/off) telemetry for AHUs 1-4, indexed by
        # AHU number minus one. None means no telemetry has been received yet
        # for that AHU.
        self.ahu_working_states: list[bool | None] = [None] * N_AHUS

        # Cached AHU setpoint, which can be used if catchup is required
        # because one or more AHUs has been disabled.
        self.cached_ahu_setpoint: float | None = None

        # Catch-up offset (°C, <= 0) currently applied to the AHU setpoint. It
        # carries the first stage's hold while AHUs are off (latched to the
        # deepest level reached, see ahu_working_state_callback) and, once all
        # AHUs are back on, the second stage's ambient-overshoot lowering (see
        # apply_ahu_off_catchup). Only one stage owns it at a time. Every
        # setpoint writer (forecast, sunrise) adds it, so a periodic forecast
        # refresh re-applies the current catch-up setpoint instead of
        # overwriting it.
        self.catchup_delta: float = 0.0

        # The ambient-overshoot catch-up coroutine. It is spawned by
        # ahu_working_state_callback when all AHUs return to on after one or
        # more were off, and cancelled when any AHU goes off again.
        self.ahu_off_catchup_task: asyncio.Task | None = None

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
        """Record the latest workingState for one AHU.

        This single callback is registered for all four
        ``HVAC.airHandlingUnit0<N>Dome`` telemetry topics. The ``ahu`` argument
        (bound via :func:`functools.partial` at registration time) identifies
        which AHU the sample describes, since the telemetry itself carries no
        unit identifier.

        Parameters
        ----------
        ahu : `int`
            The AHU number (1-4) this telemetry sample is for.
        data : `~lsst.ts.salobj.BaseMsgType`
            A newly received airHandlingUnit telemetry item.
        """
        # Off-count before recording this sample. Only this AHU's entry changes
        # per call, so the array still holds the previous state here; this lets
        # the recovery transition (any number off -> all on) be detected
        # without caching state across calls.
        previous_n_off = sum(1 for state in self.ahu_working_states if state is False)
        self.ahu_working_states[ahu - 1] = bool(data.workingState)

        if "ahu_off_catchup" in self.features_to_disable:
            return

        # Count AHUs that are explicitly off (None means no telemetry yet, so a
        # lack of data never triggers a catch-up offset). All four AHUs count,
        # regardless of whether EAS is configured to control them.
        n_off = sum(1 for state in self.ahu_working_states if state is False)
        old_catchup_delta = self.catchup_delta

        if n_off > 0:
            # One or more AHUs off: the first stage owns catchup_delta. Stop
            # the second-stage loop and (re)establish the first-stage hold. A
            # fresh episode starts from the evening target (dropping any
            # second-stage overshoot); going deeper holds the deepest level
            # reached until all AHUs are back on.
            self.cancel_ahu_off_catchup()
            if previous_n_off == 0:
                self.catchup_delta = self.ahu_off_catchup_deltas[n_off - 1]
            else:
                self.catchup_delta = min(self.catchup_delta, self.ahu_off_catchup_deltas[n_off - 1])
        elif previous_n_off > 0:
            # All AHUs just came back on: release the first-stage hold to the
            # evening target and hand catchup_delta to the second-stage
            # ambient-overshoot loop.
            self.catchup_delta = 0.0
            self.start_ahu_off_catchup()
        # Otherwise all AHUs remain on and the second-stage loop owns
        # catchup_delta; leave it untouched so this per-sample callback does
        # not clobber the value apply_ahu_off_catchup is maintaining.

        # If the offset changed, re-apply the cached setpoint with the new
        # offset: this lowers the setpoint as AHUs go off and restores it to
        # the target once they are all back on. Skipped until a base setpoint
        # is known (e.g. while the dome is open, see control_ahus_and_vec04).
        if self.catchup_delta != old_catchup_delta and self.cached_ahu_setpoint is not None:
            self.log.debug(
                "Apply AHU setpoints [6] catchup_delta=%.2f => %.2f",
                self.catchup_delta,
                self.cached_ahu_setpoint + self.catchup_delta,
            )
            await self.apply_ahu_setpoints(self.cached_ahu_setpoint + self.catchup_delta)

    def start_ahu_off_catchup(self) -> None:
        """Spawn the ambient-overshoot catch-up loop if not already running."""
        if self.ahu_off_catchup_task is None or self.ahu_off_catchup_task.done():
            self.ahu_off_catchup_task = asyncio.create_task(self.run_ahu_off_catchup())

    def cancel_ahu_off_catchup(self) -> None:
        """Cancel the ambient-overshoot catch-up loop if it is running."""
        if self.ahu_off_catchup_task is not None:
            self.ahu_off_catchup_task.cancel()
            self.ahu_off_catchup_task = None

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
      Setpoint deltas (°C) applied to the AHU working setpoint while AHUs are
      off, to help daytime regulation catch up. The elements are used when one,
      two, three, and four AHUs are off, respectively. The deepest delta reached
      is held until all four AHUs are back on, then released to zero. (While all
      four are off no setpoint is sent, but the fourth element is still held and
      applied as the AHUs come back.)
    items:
      type: number
    minItems: 4
    maxItems: 4
  ahu_off_catchup_rate:
    type: number
    default: 1.0
    description: >-
      Part of the AHU-off daytime catch-up logic. Once all AHUs are back on
      after some were turned off, the daytime AHU setpoint is lowered by this
      amount (°C) for each 1 °C the ambient (indoor ESS) temperature exceeds the
      base setpoint, down to setpoint_lower_limit.
  ahu_off_catchup_poll_interval:
    type: number
    default: 900.0
    exclusiveMinimum: 0
    description: >-
      Part of the AHU-off daytime catch-up logic. Cadence (s) at which the
      ambient-overshoot catch-up (applied once all AHUs are back on) is
      reassessed.
  ahu_off_catchup_threshold:
    type: number
    default: 1.0
    description: >-
      Part of the AHU-off daytime catch-up logic. Ambient excess (°C) above the
      base setpoint above which the ambient-overshoot catch-up (applied once all
      AHUs are back on) begins.
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

    async def apply_ahu_setpoints(self, setpoint: float, respect_lower_limit: bool = True) -> None:
        """Send a new setpoint to all of the controlled AHUs.

        The configured list of AHUs to be controlled is respected so that
        only those AHUs are commanded. Additionally, the limit imposed by
        `setpoint_lower_limit` is enforced.

        Parameters
        ----------
        `setpoint` : `float`
            The desired setpoint. The greater of this temperature or the
            configured lower limit will be applied to all currently controlled
            AHUs.
        """
        if respect_lower_limit:
            setpoint = max(setpoint, self.setpoint_lower_limit)
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
        self.cancel_ahu_off_catchup()
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

                # Clear the AHU-off catch-up hold on every dome transition:
                # opening the dome intentionally disables the AHUs, which would
                # otherwise pollute the latched offset. The cached base
                # setpoint is kept so it can be re-commanded to the AHUs on a
                # re-close instead of leaving them at the stale catch-up value.
                self.catchup_delta = 0.0

                ahus = self.get_controlled_ahus()
                if shutter_closed:
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
                    if "ahu" not in self.features_to_disable:
                        self.log.info("Disabling HVAC AHUs!")
                        disable_device_list.extend(ahus)

            if disable_device_list:
                await self.disable_devices(disable_device_list)
            if enable_device_list:
                await self.enable_devices(enable_device_list)
            # On a dome re-close the AHUs are re-enabled above; re-command the
            # current base setpoint now (after the enable, so it is not a no-op
            # on a still-disabled AHU) so they do not run at the stale catch-up
            # value they held at dome open until the next forecast refresh.
            if shutter_closed and enable_device_list and self.cached_ahu_setpoint is not None:
                self.log.debug("Apply AHU setpoints [1] %.2f", self.cached_ahu_setpoint)
                await self.apply_ahu_setpoints(self.cached_ahu_setpoint)
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
                        self.cached_ahu_setpoint = last_twilight_temperature + self.ahu_setpoint_delta
                        self.log.debug(
                            "Apply AHU setpoints [2] "
                            "last_twilight_temperature=%.2f ahu_setpoint_delta=%.2f => %.2f",
                            last_twilight_temperature,
                            self.ahu_setpoint_delta,
                            self.cached_ahu_setpoint,
                        )
                        await self.apply_ahu_setpoints(
                            last_twilight_temperature + self.ahu_setpoint_delta + self.catchup_delta,
                        )

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
            self.cached_ahu_setpoint = predicted_temperature + self.forecast_ahu_setpoint_delta
            await self.apply_ahu_setpoints(
                predicted_temperature + self.forecast_ahu_setpoint_delta + self.catchup_delta
            )

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
                    # Apply setpoint for each configured AHU. For nighttime
                    # setpoints, the lower limit is not enforced.
                    self.log.debug("Apply AHU setpoints [3] %.2f", setpoint)
                    await self.apply_ahu_setpoints(setpoint, respect_lower_limit=False)

            await asyncio.sleep(self.closed_at_night_setpoint_cadence)

    async def apply_ahu_off_catchup(self) -> bool:
        """Apply one cycle of the ambient-overshoot AHU catch-up.

        This is the second stage of the AHU-off daytime catch-up logic. The
        first stage (see `ahu_working_state_callback`) lowers the setpoint
        while AHUs are off. Once all AHUs are back on, this stage keeps
        lowering the daytime setpoint while the ambient (indoor ESS)
        temperature is running hotter than the base setpoint, so the dome can
        recover the heat it gained while the AHUs were down.

        Returns
        -------
        should_stop : `bool`
            True if the catch-up loop should stop: either the overshoot has
            been corrected, or the regulating context no longer holds (the
            feature was disabled, it is night, the dome opened, or an AHU went
            off). False if the caller should keep polling (the base setpoint or
            the ambient temperature is not yet available).
        """
        if (
            "ahu_off_catchup" in self.features_to_disable
            or "room_setpoint" in self.features_to_disable
            or "ahu" in self.features_to_disable
        ):
            return True

        # If an AHU is off the first stage owns catchup_delta.
        if not all(self.ahu_working_states):
            return True

        # All AHUs are on, so the second stage owns catchup_delta. The catch-up
        # only runs during the day with the dome closed; once that context is
        # gone there is nothing to recover, so release the offset and stop.
        if self.diurnal_timer.is_night(Time.now()) or not self.dome_model.is_closed:
            self.catchup_delta = 0.0
            return True

        # Keep polling until the data needed to regulate becomes available.
        if self.cached_ahu_setpoint is None:
            return False
        ambient_temperature = self.weather_model.current_indoor_temperature
        if math.isnan(ambient_temperature):
            return False

        base_setpoint = max(self.cached_ahu_setpoint, self.setpoint_lower_limit)
        excess = ambient_temperature - base_setpoint
        if excess <= self.ahu_off_catchup_threshold:
            # Overshoot corrected: release the offset, restore the base
            # setpoint, and finish.
            self.catchup_delta = 0.0
            self.log.debug("Apply AHU setpoints [4] %.2f", base_setpoint)
            await self.apply_ahu_setpoints(base_setpoint)
            return True

        # Lower the setpoint and record the offset in catchup_delta, so the
        # forecast/sunrise setpoint writers re-apply this lowered value on
        # their next refresh.
        self.catchup_delta = -self.ahu_off_catchup_rate * excess
        self.log.debug(
            "Apply AHU setpoints [5] catchup_delta=%.2f => %.2f", self.catchup_delta, base_setpoint
        )
        await self.apply_ahu_setpoints(base_setpoint + self.catchup_delta)
        return False

    async def run_ahu_off_catchup(self) -> None:
        """Reassess the ambient-overshoot catch-up.

        Spawned by `ahu_working_state_callback` when all AHUs return to on
        after one or more were off. Applies :meth:`apply_ahu_off_catchup` once
        immediately and then every ``ahu_off_catchup_poll_interval``
        seconds, exiting when that method reports the catch-up is complete or
        its context is gone (the callback also cancels it directly when an AHU
        goes off).
        """
        self.log.debug("run_ahu_off_catchup")
        while self.diurnal_timer.is_running:
            try:
                if await self.apply_ahu_off_catchup():
                    break
            except Exception:
                self.log.exception("In AHU-off daytime catch-up loop")
            await asyncio.sleep(self.ahu_off_catchup_poll_interval)
