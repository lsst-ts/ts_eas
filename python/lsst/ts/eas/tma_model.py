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

__all__ = ["TmaModel"]

import asyncio
import logging
import math
import time
from typing import Any, Callable

import yaml
from astropy.time import Time

from lsst.ts import salobj, utils
from lsst.ts.xml.enums.MTMount import ThermalCommandState

from .cmdwrapper import close_command_tasks, command_wrapper
from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .glass_temperature_model import GlassTemperatureModel
from .weather_model import WeatherModel

STD_TIMEOUT = 10  # seconds
DORMANT_TIME = 100  # Time to wait while sleeping, seconds

MAX_TEMPERATURE_FAILURES = 10

SECONDS_PER_HOUR = 3600.0  # Number of seconds in one hour
SLOW_COOLING_START_TIME = 2  # Time (in hours) before twilight to begin using the slow cooling rate.


class TmaModel:
    """A model for the MTMount and M1M3TS system automation.

    Parameters
    ----------
    log : `~logging.Logger`
        A logger for log messages.
    diurnal_timer : `DiurnalTimer`
        A timer that signals every day at noon, at sunrise, and at the
        end of evening twilight.
    weather_model : `WeatherModel`
        A model for the outdoor weather station, which records the last
        twilight temperature observed while the dome was opened.
    m1m3ts_remote : `~salobj.Remote`
        The Remote for the MTM1M3TS interface.
    mtmount_remote : `~salobj.Remote`
        The Remote for the MTMount interface.
    glycol_setpoint_delta : `float`
        The difference between the twilight ambient temperature and the
        setpoint to apply for the glycol, e.g., -2 if the glycol should
        be two degrees cooler than ambient.
    heater_setpoint_delta : `float`
        The difference between the twilight ambient temperature and the
        setpoint to apply for the FCU heaters, e.g., -1 if the FCU heaters
        should be one degree cooler than ambient.
    top_end_setpoint_delta : `float`
        The difference between the indoor (ESS:112) temperature and the
        setpoint to apply for the top end, via MTMount.setThermal.
    m1m3_setpoint_cadence : `float`
        The cadence at which applySetpoints commands should be sent to
        MTM1M3TS (seconds).
    after_open_delay : `float`
        Time (s) after opening to wait before applying setpoints.
    setpoint_deadband_heating : `float`
        Deadband for M1M3TS heating. If the the new setpoint exceeds the
        previous setpoint by less than this amount, no new command is sent.
        (°C)
    setpoint_deadband_cooling : `float`
        Deadband for M1M3TS cooling. If the new setpoint is lower than the
        previous setpoint by less than this amount, no new command is sent.
        (°C)
    maximum_heating_rate : `float`
        Maximum allowed rate of increase in the M1M3TS setpoint temperature.
        Limits how quickly the setpoint can rise, in degrees Celsius per hour.
        (°C/hr)
    slow_cooling_rate : `float`
        Cooling rate to be used shortly before and during the night.
        Limits how quickly the setpoint can fall, in degrees Celsius per hour.
    fast_cooling_rate : `float`
        Cooling rate to be used during the day. Limits how quickly the setpoint
        can fall, in degrees Celsius per hour.
    features_to_disable : `list[str]`
        A list of features that should be disabled. The following strings can
        be used:
         * m1m3ts
         * top_end
        Any other values are ignored.
    """

    def __init__(
        self,
        *,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        dome_model: DomeModel,
        weather_model: WeatherModel,
        glass_temperature_model: GlassTemperatureModel,
        m1m3ts_remote: salobj.Remote,
        mtmount_remote: salobj.Remote,
        glycol_setpoint_delta: float,
        heater_setpoint_delta: float,
        top_end_setpoint_delta: float,
        m1m3_setpoint_cadence: float,
        after_open_delay: float,
        setpoint_deadband_heating: float,
        setpoint_deadband_cooling: float,
        maximum_heating_rate: float,
        slow_cooling_rate: float,
        fast_cooling_rate: float,
        fan_speed: dict[str, float],
        features_to_disable: list[str],
        allow_send: Callable[[], bool] | None = None,
    ) -> None:
        self.log = log
        self.allow_send = allow_send

        self.monitor_start_event = asyncio.Event()

        self.diurnal_timer = diurnal_timer
        self.dome_model = dome_model
        self.weather_model = weather_model
        self.glass_temperature_model = glass_temperature_model

        # Remotes used by command decorators.
        self.m1m3ts_remote = m1m3ts_remote
        self.mtmount_remote = mtmount_remote

        # Configuration parameters:
        self.glycol_setpoint_delta = glycol_setpoint_delta
        self.heater_setpoint_delta = heater_setpoint_delta
        self.top_end_setpoint_delta = top_end_setpoint_delta
        self.m1m3_setpoint_cadence = m1m3_setpoint_cadence
        self.after_open_delay = after_open_delay
        self.setpoint_deadband_heating = setpoint_deadband_heating
        self.setpoint_deadband_cooling = setpoint_deadband_cooling
        self.maximum_heating_rate = maximum_heating_rate
        self.slow_cooling_rate = slow_cooling_rate
        self.fast_cooling_rate = fast_cooling_rate
        self.features_to_disable = features_to_disable

        # If true, applySetpoints needs to be called to
        # refresh the setpoints on M1M3
        self.m1m3_setpoints_are_stale: bool = True

        self.top_end_task = utils.make_done_future()
        self.top_end_task_warned: bool = False

        # Fan speed configuration
        self.fan_speed_min: float = fan_speed["fan_speed_min"]
        self.fan_speed_max: float = fan_speed["fan_speed_max"]
        self.fan_glycol_heater_offset_min: float = fan_speed["fan_glycol_heater_offset_min"]
        self.fan_glycol_heater_offset_max: float = fan_speed["fan_glycol_heater_offset_max"]
        self.fan_throttle_turn_on_temp_diff: float = fan_speed["fan_throttle_turn_on_temp_diff"]
        self.fan_throttle_max_temp_diff: float = fan_speed["fan_throttle_max_temp_diff"]

    @classmethod
    def get_config_schema(cls) -> str:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for TMA EAS configuration.
type: object
properties:
  glycol_setpoint_delta:
    description: Difference (°C) between the ambient temperature and M1M3TS glycol setpoint.
    type: number
    default: -2.0
  heater_setpoint_delta:
    description: Difference (°C) between the ambient temperature and M1M3TS heater setpoint.
    type: number
    default: -1.0
  top_end_setpoint_delta:
    description: Difference (°C) between the ambient temperature and MTMount thermal setpoint.
    type: number
    default: -1.0
  m1m3_setpoint_cadence:
    description: Time (s) between successive assessments of the TMA setpoint.
    type: number
    default: 300.0
  after_open_delay:
    description: Time (s) after opening to wait before applying setpoints.
    type: number
    default: 1800.0
  setpoint_deadband_heating:
    description: Allowed upward deviation (°C) of MTM1M3TS setpoint before it is re-applied.
    type: number
    default: 0.1
  setpoint_deadband_cooling:
    description: Allowed downward deviation (°C) of MTM1M3TS setpoint before it is re-applied.
    type: number
    default: 0.1
  maximum_heating_rate:
    description: Maximum allowed rate (°C/hour) at which the MTM1M3 setpoint is allowed to increase.
    type: number
    default: 1.0
  slow_cooling_rate:
    description: >-
        Maximum allowed rate (°C/hour) at which the MTM1M3 setpoint is allowed to decrease
        while observing.
    type: number
    default: 1.0
  fast_cooling_rate:
    description: >-
        Maximum allowed rate (°C/hour) at which the MTM1M3 setpoint is allowed to decrease
        while observing.
    type: number
    default: 10.0
  fan_speed:
    description: Parameters controlling the M1M3TS fan speed response.
    type: object
    properties:
      fan_speed_min:
        description: Minimum allowed M1M3TS fan speed (RPM).
        type: number
        default: 700.0
      fan_speed_max:
        description: Maximum allowed M1M3TS fan speed (RPM).
        type: number
        default: 2000.0
      fan_glycol_heater_offset_min:
        description: Glycol–heater temperature offset (°C) at fan_speed_min.
        type: number
        default: -1.0
      fan_glycol_heater_offset_max:
        description: Glycol–heater temperature offset (°C) at fan_speed_max.
        type: number
        default: -4.0
      fan_throttle_turn_on_temp_diff:
        description: Temperature difference (°C) at which fan speed begins to increase.
        type: number
        default: 0.0
      fan_throttle_max_temp_diff:
        description: Temperature difference (°C) at which fan_speed_max is commanded.
        type: number
        default: 1.0
    required:
      - fan_speed_min
      - fan_speed_max
      - fan_glycol_heater_offset_min
      - fan_glycol_heater_offset_max
      - fan_throttle_turn_on_temp_diff
      - fan_throttle_max_temp_diff
    additionalProperties: false
required:
  - glycol_setpoint_delta
  - heater_setpoint_delta
  - top_end_setpoint_delta
  - m1m3_setpoint_cadence
  - after_open_delay
  - setpoint_deadband_heating
  - setpoint_deadband_cooling
  - maximum_heating_rate
  - slow_cooling_rate
  - fast_cooling_rate
  - fan_speed
additionalProperties: false
"""
        )

    @command_wrapper(remote_attr="m1m3ts_remote", command_attr="cmd_applySetpoints")
    async def send_apply_setpoints(
        self,
        *,
        glycol_setpoint: float,
        heaters_setpoint: float,
    ) -> dict[str, Any]:
        return {
            "glycolSetpoint": glycol_setpoint,
            "heatersSetpoint": heaters_setpoint,
        }

    @command_wrapper(remote_attr="m1m3ts_remote", command_attr="cmd_heaterFanDemand")
    async def send_heater_fan_demand(
        self,
        *,
        heater_pwm: list[int],
        fan_rpm: list[int],
    ) -> dict[str, Any]:
        return {"heaterPWM": heater_pwm, "fanRPM": fan_rpm}

    @command_wrapper(remote_attr="mtmount_remote", command_attr="cmd_setThermal")
    async def send_set_thermal(
        self,
        *,
        top_end_chiller_setpoint: float,
        top_end_chiller_state: ThermalCommandState,
    ) -> dict[str, Any]:
        return {
            "topEndChillerSetpoint": top_end_chiller_setpoint,
            "topEndChillerState": top_end_chiller_state,
        }

    async def monitor(self) -> None:
        self.log.debug("TmaModel.monitor")

        if "m1m3ts" not in self.features_to_disable or "top_end" not in self.features_to_disable:
            self.log.debug("TmaModel.monitor started")

            try:
                await asyncio.gather(
                    self.follow_ess_indoor(),
                    self.wait_for_sunrise(),
                )
            finally:
                if not self.top_end_task.done():
                    self.top_end_task.cancel()
                    try:
                        await self.top_end_task
                    except asyncio.CancelledError:
                        pass  # Expected

                self.monitor_start_event.clear()

            self.log.debug("TmaModel.monitor closing...")

        else:
            # If m1m3ts is disabled, just sleep.
            while True:
                try:
                    await asyncio.sleep(DORMANT_TIME)
                except asyncio.CancelledError:
                    self.log.debug("monitor cancelled")
                    raise

    async def apply_setpoints(self, setpoint: float) -> None:
        if "m1m3ts" not in self.features_to_disable:
            glycol_setpoint = setpoint + self.glycol_setpoint_delta
            heaters_setpoint = setpoint + self.heater_setpoint_delta
            self.log.debug(f"Setting MTM1MTS: {glycol_setpoint=:.2f}°C {heaters_setpoint=:.2f}°C")
            await self.send_apply_setpoints(
                glycol_setpoint=glycol_setpoint,
                heaters_setpoint=heaters_setpoint,
            )

        self.m1m3_setpoints_are_stale = False

    async def set_fan_speed(self, *, setpoint: float) -> None:
        """Compute and send the FCU fan speed based on glass temperature.

        The commanded fan speed is determined by the absolute difference
        between the median bulk glass temperature and the target setpoint
        (including the heater setpoint delta). The speed scales linearly
        from `self.fan_speed_min` at or below
        `self.fan_throttle_turn_on_temp_diff` to `self.fan_speed_max` at
        `self.fan_throttle_max_temp_diff` degrees difference, with any
        larger differences clamped at maximum.

        The computed speed is scaled as required by the
        heaterFanDemand command and applied to all 96 fans.

        Parameters
        ----------
        setpoint : `float`
            Demand temperature (°C) before applying any offsets.
        """
        glass_temperature = self.glass_temperature_model.median_temperature

        if (
            glass_temperature is None
            or not math.isfinite(glass_temperature)
            or not -100 < glass_temperature < 100
        ):
            return

        setpoint += self.heater_setpoint_delta
        slope = (self.fan_speed_max - self.fan_speed_min) / (
            self.fan_throttle_max_temp_diff - self.fan_throttle_turn_on_temp_diff
        )
        temperature_difference = abs(glass_temperature - setpoint)

        fan_speed = self.fan_speed_min + slope * (
            temperature_difference - self.fan_throttle_turn_on_temp_diff
        )
        fan_speed = max(min(fan_speed, self.fan_speed_max), self.fan_speed_min)

        fan_rpm = int(round(0.1 * fan_speed))
        await self.send_heater_fan_demand(
            heater_pwm=[-1] * 96,
            fan_rpm=[fan_rpm] * 96,
        )

        if glass_temperature > setpoint:
            # Adjust glycol offset based on fan speed:
            #   Fan speed of 700 (minimum) -> glycol offset of -1 from heater
            #   Fan speed of 2000 (maximum) -> glycol offset of -4 from heater
            glycol_offset_slope = (self.fan_glycol_heater_offset_max - self.fan_glycol_heater_offset_min) / (
                self.fan_speed_max - self.fan_speed_min
            )
            glycol_offset = glycol_offset_slope * (fan_speed - self.fan_speed_min)
            glycol_offset += self.fan_glycol_heater_offset_min
            self.glycol_setpoint_delta = self.heater_setpoint_delta + glycol_offset
        else:
            self.glycol_setpoint_delta = self.heater_setpoint_delta + self.fan_glycol_heater_offset_min

        self.m1m3_setpoints_are_stale = True

    async def follow_ess_indoor(self) -> None:
        self.log.debug("follow_ess_indoor")

        n_failures = 0  # Number of failures to read temperature

        self.monitor_start_event.set()

        while self.diurnal_timer.is_running:
            if "require_dome_open" not in self.features_to_disable:
                # Wait for some dome telemetry to arrive to avoid
                # an unnecessary wait of `after_open_delay` seconds.
                while self.dome_model.is_closed is None:
                    await asyncio.sleep(STD_TIMEOUT)

                while self.dome_model.is_closed is not False:
                    event = asyncio.Event()
                    self.dome_model.on_open.append((event, self.after_open_delay))
                    self.log.debug("Waiting for dome open.")
                    await event.wait()
                    self.log.debug("Dome has been opened.")

            indoor_temperature = self.weather_model.current_indoor_temperature

            if indoor_temperature is None or math.isnan(indoor_temperature):
                self.log.warning(
                    f"Failed to collect an indoor temperature measurement! ({indoor_temperature=})"
                )
                n_failures += 1
                if n_failures == MAX_TEMPERATURE_FAILURES:
                    self.log.error("No temperature samples were collected. CSC will fault.")
                    raise RuntimeError("No temperature samples were collected.")
                await asyncio.sleep(self.m1m3_setpoint_cadence)
                continue
            else:
                n_failures = 0

            if "top_end" not in self.features_to_disable:
                await self.send_set_thermal(
                    top_end_chiller_setpoint=indoor_temperature + self.top_end_setpoint_delta,
                    top_end_chiller_state=ThermalCommandState.ON,
                )

            msg = self.m1m3ts_remote.evt_appliedSetpoints.get()
            if msg is None:
                last_m1m3ts_setpoint = None
            else:
                last_m1m3ts_setpoint = msg.heatersSetpoint - self.heater_setpoint_delta

            # Apply the new setpoint to change fan speed.
            if "fanspeed" not in self.features_to_disable and not math.isnan(indoor_temperature):
                await self.set_fan_speed(setpoint=indoor_temperature)

            # Maximum cooling rate depends on environmental conditions:
            #  * If the dome is open or
            #    sunrise time > current time > twilight-2hours
            #    then we use the slow cooling rate
            #    (currently 1 degree per hour)
            #  * Otherwise we use the fast cooling rate
            #    (currently 10 degrees per hour)
            # We assume that twilight is always shorter than two hours, as
            # is the case at Cerro Pachón.
            current_time = time.time()
            use_slow_cooling_rate = (
                (not self.dome_model.is_closed)
                or (self.diurnal_timer.sun_altitude_at(current_time) < 0)
                or (
                    self.diurnal_timer.seconds_until_twilight(Time.now())
                    < SLOW_COOLING_START_TIME * SECONDS_PER_HOUR
                )
            )
            cooling_rate = self.slow_cooling_rate if use_slow_cooling_rate else self.fast_cooling_rate

            if last_m1m3ts_setpoint is None:
                # No previous setpoint = apply it regardless
                await self.apply_setpoints(indoor_temperature)

            elif indoor_temperature > last_m1m3ts_setpoint:
                # Warm the mirror if the setpoint is past the heating deadband.
                new_setpoint = indoor_temperature
                delta = new_setpoint - last_m1m3ts_setpoint

                if delta > self.setpoint_deadband_heating:
                    # Apply the setpoint, limited by maximum_heating_rate
                    maximum_heating_step = (
                        self.maximum_heating_rate * self.m1m3_setpoint_cadence / SECONDS_PER_HOUR
                    )
                    new_setpoint = min(
                        new_setpoint,
                        last_m1m3ts_setpoint + maximum_heating_step,
                    )
                    await self.apply_setpoints(new_setpoint)
                else:
                    self.log.debug("Heating deadband criterion not met. No M1M3TS setpoint update.")

            elif indoor_temperature < last_m1m3ts_setpoint:
                # Cool the mirror if the setpoint is past the cooling deadband.
                new_setpoint = indoor_temperature
                delta = last_m1m3ts_setpoint - new_setpoint

                if delta > self.setpoint_deadband_cooling:
                    # Apply the setpoint, limited by maximum_cooling_rate
                    maximum_cooling_step = cooling_rate * self.m1m3_setpoint_cadence / SECONDS_PER_HOUR
                    new_setpoint = max(
                        new_setpoint,
                        last_m1m3ts_setpoint - maximum_cooling_step,
                    )
                    await self.apply_setpoints(new_setpoint)
                else:
                    self.log.debug("Cooling deadband criterion not met. No M1M3TS setpoint update.")

            else:
                # indoor_temperature == last_m1m3ts_setpoint
                self.log.debug("No M1M3TS setpoint update required.")

            if self.m1m3_setpoints_are_stale and last_m1m3ts_setpoint is not None:
                await self.apply_setpoints(indoor_temperature)

            await asyncio.sleep(self.m1m3_setpoint_cadence)

    async def wait_for_sunrise(self) -> None:
        """Wait for sunrise and then sets the room temperature.

        Wait for the timer to signal sunrise, and then obtain the
        temperature that was reported last night at the end
        of twilight, and then apply that temperature as the
        M1M3TS for the day.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.sunrise_condition:
                await self.diurnal_timer.sunrise_condition.wait()
                last_twilight_temperature = await self.weather_model.get_last_twilight_temperature()
                if self.diurnal_timer.is_running and last_twilight_temperature is not None:
                    self.log.info(
                        "Sunrise M1M3TS and top end is set based on twilight temperature: "
                        f"{last_twilight_temperature:.2f}°C"
                    )
                    await self.apply_setpoints(last_twilight_temperature)

                    if "top_end" not in self.features_to_disable:
                        chiller_setpoint = last_twilight_temperature + self.top_end_setpoint_delta
                        await self.send_set_thermal(
                            top_end_chiller_setpoint=chiller_setpoint,
                            top_end_chiller_state=ThermalCommandState.ON,
                        )

    async def close(self) -> None:
        """Cancel any in-flight command tasks."""
        await close_command_tasks(self)
