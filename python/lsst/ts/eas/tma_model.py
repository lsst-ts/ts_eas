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

__all__ = [
    "TmaModel",
    "MIN_FAN_RPM",
    "MAX_FAN_RPM",
    "OFFSET_AT_MIN_RPM",
    "OFFSET_AT_MAX_RPM",
    "FAN_SCALE_DT",
]

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Callable

import yaml
from astropy.time import Time
from lsst.ts import salobj, utils
from lsst.ts.xml.enums.MTMount import ThermalCommandState
from lsst.ts.xml.sal_enums import State

from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .glass_temperature_model import GlassTemperatureModel
from .weather_model import WeatherModel

STD_TIMEOUT = 10  # seconds
DORMANT_TIME = 100  # Time to wait while sleeping, seconds

MIN_FAN_RPM = 700  # Minimum allowed M1M3TS fan speed
MAX_FAN_RPM = 2000  # Maximum allowed fan speed
OFFSET_AT_MIN_RPM = -1.0  # at 700 rpm
OFFSET_AT_MAX_RPM = -4.0  # at 2000 rpm
FAN_SCALE_DT = 1.0  # Temperature difference at which we command MAX_RPM

MAX_TEMPERATURE_FAILURES = 10

SECONDS_PER_HOUR = 3600.0  # Number of seconds in one hour
SLOW_COOLING_START_TIME = (
    2  # Time (in hours) before twilight to begin using the slow cooling rate.
)

LastSetpointGetter = Callable[[], float | None]


class TmaModel:
    """A model for the MTMount and M1M3TS system automation.

    Parameters
    ----------
    domain : `~lsst.ts.salobj.Domain`
        A SAL domain object for obtaining remotes.
    log : `~logging.Logger`
        A logger for log messages.
    diurnal_timer : `DiurnalTimer`
        A timer that signals every day at noon, at sunrise, and at the
        end of evening twilight.
    weather_model : `WeatherModel`
        A model for the outdoor weather station, which records the last
        twilight temperature observed while the dome was opened.
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
    m1m3_setpoint_cadence : float
        The cadence at which applySetpoints commands should be sent to
        MTM1M3TS (seconds).
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
        domain: salobj.Domain,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        dome_model: DomeModel,
        weather_model: WeatherModel,
        glass_temperature_model: GlassTemperatureModel,
        glycol_setpoint_delta: float,
        heater_setpoint_delta: float,
        top_end_setpoint_delta: float,
        m1m3_setpoint_cadence: float,
        setpoint_deadband_heating: float,
        setpoint_deadband_cooling: float,
        maximum_heating_rate: float,
        slow_cooling_rate: float,
        fast_cooling_rate: float,
        features_to_disable: list[str],
    ) -> None:
        self.domain = domain
        self.log = log

        self.monitor_start_event = asyncio.Event()

        self.diurnal_timer = diurnal_timer
        self.dome_model = dome_model
        self.weather_model = weather_model
        self.glass_temperature_model = glass_temperature_model

        # Configuration parameters:
        self.glycol_setpoint_delta = glycol_setpoint_delta
        self.heater_setpoint_delta = heater_setpoint_delta
        self.top_end_setpoint_delta = top_end_setpoint_delta
        self.m1m3_setpoint_cadence = m1m3_setpoint_cadence
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

    @asynccontextmanager
    async def last_setpoint_getter(
        self, mtm1m3ts_remote: salobj.Remote
    ) -> AsyncIterator[LastSetpointGetter]:
        """Handle context for a function getting the last MTM1M3TS setpoint.

        Parameters
        ----------
        mtm1m3ts_remote: salobj.Remote
            The MTM1M3TS remote

        Returns
        -------
        AsyncIterator[LastSetpointGetter]
            An asynchronous iterator that yields a function. The yielded
            function, when called, gets the most recently received MTM1M3TS
            setpoint value as a float, or `None` if no setpoint is available.
        """
        salinfo_copy = salobj.SalInfo(self.domain, mtm1m3ts_remote.salinfo.name)
        topic = salobj.topics.ReadTopic(
            salinfo=salinfo_copy, attr_name="cmd_applySetpoints", max_history=1
        )
        await salinfo_copy.start()

        def get_last_setpoint() -> float | None:
            msg = topic.get()
            if msg is None:
                return None
            return msg.heatersSetpoint - self.heater_setpoint_delta

        try:
            yield get_last_setpoint
        finally:
            await salinfo_copy.close()

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
required:
  - glycol_setpoint_delta
  - heater_setpoint_delta
  - top_end_setpoint_delta
  - m1m3_setpoint_cadence
  - setpoint_deadband_heating
  - setpoint_deadband_cooling
  - maximum_heating_rate
  - slow_cooling_rate
  - fast_cooling_rate
additionalProperties: false
"""
        )

    async def monitor(self) -> None:
        self.log.debug("TmaModel.monitor")

        async with (
            salobj.Remote(
                domain=self.domain,
                name="MTM1M3TS",
            ) as m1m3ts_remote,
            salobj.Remote(
                domain=self.domain,
                name="MTMount",
            ) as mtmount_remote,
            self.last_setpoint_getter(m1m3ts_remote) as get_last_setpoint,
        ):
            if (
                "m1m3ts" not in self.features_to_disable
                or "top_end" not in self.features_to_disable
            ):
                ready_futures: list[asyncio.Future] = [
                    asyncio.Future() for _ in range(2)
                ]
                m1m3ts_future = asyncio.gather(
                    self.follow_ess_indoor(
                        m1m3ts_remote=m1m3ts_remote,
                        mtmount_remote=mtmount_remote,
                        get_last_setpoint=get_last_setpoint,
                        future=ready_futures[0],
                    ),
                    self.wait_for_sunrise(
                        m1m3ts_remote=m1m3ts_remote,
                        mtmount_remote=mtmount_remote,
                        future=ready_futures[1],
                    ),
                )
                await asyncio.gather(*ready_futures)
                self.log.debug("TmaModel.monitor started")
                self.monitor_start_event.set()

                try:
                    await m1m3ts_future
                except asyncio.CancelledError:
                    m1m3ts_future.cancel()
                    await asyncio.gather(m1m3ts_future, return_exceptions=True)
                    raise
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

    async def apply_setpoints(
        self,
        *,
        m1m3ts_remote: salobj.Remote,
        setpoint: float,
    ) -> None:
        if "m1m3ts" not in self.features_to_disable:
            glycol_setpoint = setpoint + self.glycol_setpoint_delta
            heaters_setpoint = setpoint + self.heater_setpoint_delta
            self.log.info(
                f"Setting MTM1MTS: {glycol_setpoint=}°C {heaters_setpoint=}°C"
            )
            await m1m3ts_remote.cmd_applySetpoints.set_start(
                glycolSetpoint=glycol_setpoint,
                heatersSetpoint=heaters_setpoint,
            )

        self.m1m3_setpoints_are_stale = False

    async def set_fan_speed(
        self,
        *,
        m1m3ts_remote: salobj.Remote,
        setpoint: float,
    ) -> None:
        """Compute and send the FCU fan speed based on glass temperature.

        The commanded fan speed is determined by the absolute difference
        between the median bulk glass temperature and the target setpoint
        (including the heater setpoint delta). The speed scales linearly
        from `MIN_FAN_RPM` at zero temperature difference to `MAX_FAN_RPM`
        at `FAN_SCALE_DT` degrees difference, with any larger differences
        clamped at maximum.

        The computed speed is scaled as required by the
        heaterFanDemand command and applied to all 96 fans.

        Parameters
        ----------
        m1m3ts_remote : `~lsst.ts.salobj.Remote`
            SAL remote for the M1M3 thermal system CSC.
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
        slope = (MAX_FAN_RPM - MIN_FAN_RPM) / FAN_SCALE_DT
        temperature_difference = abs(glass_temperature - setpoint)

        fan_speed = MIN_FAN_RPM + slope * temperature_difference
        fan_speed = min(fan_speed, MAX_FAN_RPM)
        fan_rpm = int(round(0.1 * fan_speed))

        await m1m3ts_remote.cmd_heaterFanDemand.set_start(
            heaterPWM=[-1] * 96,
            fanRPM=[fan_rpm] * 96,
        )

        if glass_temperature > setpoint:
            # Adjust glycol offset based on fan speed:
            #   Fan speed of 700 (minimum) -> glycol offset of -1 from heater
            #   Fan speed of 2000 (maximum) -> glycol offset of -5 from heater
            glycol_offset_slope = (OFFSET_AT_MAX_RPM - OFFSET_AT_MIN_RPM) / (
                MAX_FAN_RPM - MIN_FAN_RPM
            )
            glycol_offset = glycol_offset_slope * (fan_speed - MIN_FAN_RPM)
            glycol_offset += OFFSET_AT_MIN_RPM
            self.glycol_setpoint_delta = self.heater_setpoint_delta + glycol_offset
        else:
            self.glycol_setpoint_delta = self.heater_setpoint_delta + OFFSET_AT_MIN_RPM

        self.m1m3_setpoints_are_stale = True

    async def start_top_end_task(
        self, mtmount_remote: salobj.Remote, setpoint: float
    ) -> None:
        """Schedule self.apply_top_end_setpoint for asynchronous execution.

        This method handles canceling an outstanding call if needed and will
        raise if the previous call failed. If the previous call was still
        running (presumably because the MTMount CSC was not enabled) then a
        warning is logged.

        Parameters
        ----------
        mtmount_remote: `~lsst.ts.salobj.Remote`
            The remote object for MTMount CSC commands.

        setpoint: float
            The setpoint (without delta applied) for the mirror system.
        """
        if "top_end" in self.features_to_disable:
            return

        if self.top_end_task.done():
            self.top_end_task_warned = False
        else:
            self.top_end_task.cancel()
        try:
            await self.top_end_task
        except asyncio.CancelledError:
            if not self.top_end_task_warned:
                self.log.warning(
                    "Previous attempt to apply MTMount setpoint was cancelled."
                )
                self.top_end_task_warned = True

        self.top_end_task = asyncio.create_task(
            self.apply_top_end_setpoint(mtmount_remote, setpoint)
        )

    async def apply_top_end_setpoint(
        self, mtmount_remote: salobj.Remote, setpoint: float
    ) -> None:
        """Apply the average temperature plus offset as the top end setpoint.

        This is a very simple approach to start with, but more complexity may
        be added later. That might merit other methods, or a separate model
        for the top end, but for now we just use the simplest approach
        possible.

        Parameters
        ----------
        mtmount_remote: `~lsst.ts.salobj.Remote`
            The remote object for MTMount CSC commands.

        setpoint: float
            The setpoint (without delta applied) for the mirror system.
        """
        while (
            await mtmount_remote.evt_summaryState.aget(timeout=STD_TIMEOUT)
        ).summaryState != State.ENABLED:
            await asyncio.sleep(DORMANT_TIME)

        top_end_setpoint = setpoint + self.top_end_setpoint_delta
        try:
            await asyncio.wait_for(
                mtmount_remote.cmd_setThermal.set_start(
                    topEndChillerSetpoint=top_end_setpoint,
                    topEndChillerState=ThermalCommandState.ON,
                ),
                timeout=STD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self.log.exception("Apply setpoint to top end timed out!")

    async def follow_ess_indoor(
        self,
        *,
        m1m3ts_remote: salobj.Remote,
        mtmount_remote: salobj.Remote,
        get_last_setpoint: LastSetpointGetter,
        future: asyncio.Future,
    ) -> None:
        self.log.debug("follow_ess_indoor")

        n_failures = 0  # Number of failures to read temperature

        if not future.done():
            future.set_result(None)

        while self.diurnal_timer.is_running:
            if "require_dome_open" not in self.features_to_disable:
                if self.dome_model.is_closed is not False:
                    event = asyncio.Event()
                    self.dome_model.on_open.append(event)
                    await event.wait()

            indoor_temperature = self.weather_model.current_indoor_temperature

            if indoor_temperature is None or math.isnan(indoor_temperature):
                self.log.warning(
                    f"Failed to collect an indoor temperature measurement! ({indoor_temperature=})"
                )
                n_failures += 1
                if n_failures == MAX_TEMPERATURE_FAILURES:
                    self.log.error(
                        "No temperature samples were collected. CSC will fault."
                    )
                    raise RuntimeError("No temperature samples were collected.")
                await asyncio.sleep(self.m1m3_setpoint_cadence)
                continue
            else:
                n_failures = 0

            await self.start_top_end_task(mtmount_remote, indoor_temperature)

            last_m1m3ts_setpoint = get_last_setpoint()

            # Apply the new setpoint to change fan speed.
            if "fanspeed" not in self.features_to_disable and not math.isnan(
                indoor_temperature
            ):
                await self.set_fan_speed(
                    m1m3ts_remote=m1m3ts_remote,
                    setpoint=indoor_temperature,
                )

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
            cooling_rate = (
                self.slow_cooling_rate
                if use_slow_cooling_rate
                else self.fast_cooling_rate
            )

            if last_m1m3ts_setpoint is None:
                # No previous setpoint = apply it regardless
                await self.apply_setpoints(
                    m1m3ts_remote=m1m3ts_remote,
                    setpoint=indoor_temperature,
                )
            elif indoor_temperature > last_m1m3ts_setpoint:
                # Warm the mirror if the setpoint is past the deadband
                new_setpoint = indoor_temperature
                if new_setpoint - last_m1m3ts_setpoint > self.setpoint_deadband_heating:
                    # Apply the setpoint, limited by maximum_heating_rate
                    maximum_heating_step = (
                        self.maximum_heating_rate
                        * self.m1m3_setpoint_cadence
                        / SECONDS_PER_HOUR
                    )
                    new_setpoint = min(
                        indoor_temperature,
                        last_m1m3ts_setpoint + maximum_heating_step,
                    )
                    await self.apply_setpoints(
                        m1m3ts_remote=m1m3ts_remote,
                        setpoint=new_setpoint,
                    )
                else:
                    # Cool the mirror if the setpoint is past the deadband
                    new_setpoint = indoor_temperature
                    if (
                        last_m1m3ts_setpoint - new_setpoint
                        > self.setpoint_deadband_cooling
                    ):
                        # Apply the setpoint, limited by maximum_cooling_rate
                        maximum_cooling_step = (
                            cooling_rate * self.m1m3_setpoint_cadence / SECONDS_PER_HOUR
                        )
                        new_setpoint = max(
                            indoor_temperature,
                            last_m1m3ts_setpoint - maximum_cooling_step,
                        )
                        await self.apply_setpoints(
                            m1m3ts_remote=m1m3ts_remote,
                            setpoint=new_setpoint,
                        )
            else:
                # Cool the mirror if the setpoint is past the deadband
                new_setpoint = indoor_temperature
                if last_m1m3ts_setpoint - new_setpoint > self.setpoint_deadband_cooling:
                    # Apply the setpoint, limited by maximum_cooling_rate
                    maximum_cooling_step = (
                        cooling_rate * self.m1m3_setpoint_cadence / SECONDS_PER_HOUR
                    )
                    new_setpoint = max(
                        indoor_temperature,
                        last_m1m3ts_setpoint - maximum_cooling_step,
                    )
                    await self.apply_setpoints(
                        m1m3ts_remote=m1m3ts_remote,
                        setpoint=new_setpoint,
                    )

            if self.m1m3_setpoints_are_stale and last_m1m3ts_setpoint is not None:
                await self.apply_setpoints(
                    m1m3ts_remote=m1m3ts_remote,
                    setpoint=indoor_temperature,
                )

            await asyncio.sleep(self.m1m3_setpoint_cadence)

    async def wait_for_sunrise(
        self,
        *,
        m1m3ts_remote: salobj.Remote,
        mtmount_remote: salobj.Remote,
        future: asyncio.Future,
    ) -> None:
        """Wait for sunrise and then sets the room temperature.

        Wait for the timer to signal sunrise, and then obtain the
        temperature that was reported last night at the end
        of twilight, and then apply that temperature as the
        M1M3TS for the day.

        Parameters
        ----------
        m1m3ts_remote : `~lsst.ts.salobj.Remote`
            A SALobj remote representing the MTM1M3TS controller.
        mtmount_remote : `~lsst.ts.salobj.Remote`
            A SALobj remote representing the MTMount controller.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.sunrise_condition:
                if not future.done():
                    future.set_result(None)

                await self.diurnal_timer.sunrise_condition.wait()
                last_twilight_temperature = (
                    await self.weather_model.get_last_twilight_temperature()
                )
                if (
                    self.diurnal_timer.is_running
                    and last_twilight_temperature is not None
                ):
                    self.log.info(
                        "Sunrise M1M3TS and top end is set based on twilight temperature: "
                        f"{last_twilight_temperature:.2f}°C"
                    )
                    await self.apply_setpoints(
                        m1m3ts_remote=m1m3ts_remote,
                        setpoint=last_twilight_temperature,
                    )
                    await self.start_top_end_task(
                        mtmount_remote=mtmount_remote,
                        setpoint=last_twilight_temperature,
                    )
