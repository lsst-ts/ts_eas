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

from lsst.ts import salobj, utils
from lsst.ts.xml.sal_enums import State

from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .weather_model import WeatherModel

STD_TIMEOUT = 10  # seconds
DORMANT_TIME = 100  # Time to wait while sleeping, seconds


class TmaModel:
    """A model for the MTMount and M1M3TS system automation.

    Parameters
    ----------
    domain : salobj.Domain
        A SAL domain object for obtaining remotes.
    log : logging.Logger
        A logger for log messages.
    diurnal_timer : DiurnalTimer
        A timer that signals every day at noon and at the end of evening
        twilight.
    dome_model : DomeModel
        A model for the MTDome remote, indicating whether it is closed.
    weather_model : WeatherModel
        A model for the outdoor weather station, which records the last
        twilight temperature observed while the dome was opened.
    indoor_ess_index : int
        The SAL index for the indoor ESS meter.
    ess_timeout : float
        The amount of time (seconds) of no ESS measurements after which
        the CSC should fault.
    glycol_setpoint_delta : float
        The difference between the twilight ambient temperature and the
        setpoint to apply for the glycol, e.g., -2 if the glycol should
        be two degrees cooler than ambient.
    heater_setpoint_delta : float
        The difference between the twilight ambient temperature and the
        setpoint to apply for the FCU heaters, e.g., -1 if the FCU heaters
        should be one degree cooler than ambient.
    top_end_setpoint_delta : float
        The difference between the indoor (ESS:112) temperature and the
        setpoint to apply for the top end, via MTMount.setThermal.
    m1m3_setpoint_cadence : float
        The cadence at which applySetpoints commands should be sent to
        MTM1M3TS (seconds).
    setpoint_deadband_heating : float
        Deadband for M1M3TS heating. If the the new setpoint exceeds the
        previous setpoint by less than this amount, no new command is sent.
        (°C)
    setpoint_deadband_cooling : float
        Deadband for M1M3TS cooling. If the new setpoint is lower than the
        previous setpoint by less than this amount, no new command is sent.
        (°C)
    maximum_heating_rate : float
        Maximum allowed rate of increase in the M1M3TS setpoint temperature.
        Limits how quickly the setpoint can rise, in degrees Celsius per hour.
        (°C/hr)
    features_to_disable : list[str]
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
        indoor_ess_index: int,
        ess_timeout: float,
        glycol_setpoint_delta: float,
        heater_setpoint_delta: float,
        top_end_setpoint_delta: float,
        m1m3_setpoint_cadence: float,
        setpoint_deadband_heating: float,
        setpoint_deadband_cooling: float,
        maximum_heating_rate: float,
        features_to_disable: list[str] = [],
    ) -> None:
        self.domain = domain
        self.log = log

        self.monitor_start_event = asyncio.Event()

        self.diurnal_timer = diurnal_timer
        self.dome_model = dome_model
        self.weather_model = weather_model

        # Configuration parameters:
        self.indoor_ess_index = indoor_ess_index
        self.ess_timeout = ess_timeout
        self.glycol_setpoint_delta = glycol_setpoint_delta
        self.heater_setpoint_delta = heater_setpoint_delta
        self.top_end_setpoint_delta = top_end_setpoint_delta
        self.m1m3_setpoint_cadence = m1m3_setpoint_cadence
        self.setpoint_deadband_heating = setpoint_deadband_heating
        self.setpoint_deadband_cooling = setpoint_deadband_cooling
        self.maximum_heating_rate = maximum_heating_rate
        self.features_to_disable = features_to_disable

        # Last setpoint, for deadband purposes
        self.last_m1m3ts_setpoint: float | None = None

        self.top_end_task = utils.make_done_future()
        self.top_end_task_warned: bool = False

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
                        future=ready_futures[0],
                    ),
                    self.wait_for_noon(
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
        glycol_setpoint = setpoint + self.glycol_setpoint_delta
        heaters_setpoint = setpoint + self.heater_setpoint_delta
        self.log.info(f"Setting MTM1MTS: {glycol_setpoint=}°C {heaters_setpoint=}°C")
        if hasattr(m1m3ts_remote, "cmd_applySetpoints"):
            await m1m3ts_remote.cmd_applySetpoints.set_start(
                glycolSetpoint=glycol_setpoint,
                heatersSetpoint=heaters_setpoint,
            )
        else:
            await m1m3ts_remote.cmd_applySetpoint.set_start(
                glycolSetpoint=glycol_setpoint,
                heatersSetpoint=heaters_setpoint,
            )

        # Record the last setpoint used.
        self.last_m1m3ts_setpoint = setpoint

    async def start_top_end_task(
        self, mtmount_remote: salobj.Remote, setpoint: float
    ) -> None:
        """Schedules self.apply_top_end_setpoint for asynchronous execution.

        This method handles canceling an outstanding call if needed and will
        raise if the previous call failed. If the previous call was still
        running (presumably because the MTMount CSC was not enabled) then a
        warning is logged.

        Parameters
        ----------
        mtmount_remote: salobj.Remote
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
        be added later. That might merit other methods, or a seperate model
        for the top end, but for now we just use the simplest approach
        possible.

        Parameters
        ----------
        mtmount_remote: salobj.Remote
            The remote object for MTMount CSC commands.

        setpoint: float
            The setpoint (without delta applied) for the mirror system.
        """
        while (
            await mtmount_remote.evt_summaryState.aget(timeout=STD_TIMEOUT)
        ).summaryState != State.ENABLED:
            await asyncio.sleep(DORMANT_TIME)

        top_end_setpoint = setpoint + self.top_end_setpoint_delta
        await mtmount_remote.cmd_setThermal.set_start(
            topEndChillerSetpoint=top_end_setpoint
        )

    async def collect_temperature_samples(
        self, ess_remote: salobj.Remote
    ) -> float | None:
        """Gather temperature samples over the configured cadence period.

        If a single `asyncio.TimeoutError` occurs, the contiguous-time clock is
        restarted (``end_time = now + cadence``).  Any other exception is
        propagated so that the caller can decide how to handle it.

        Parameters
        ----------
        ess_remote : salobj.Remote
            A remote endpoint for the ESS to collect from.

        Returns
        -------
        float | None
            Average temperature readings, or None if a valid average
            could not be collected.

        Raises
        ------
        asyncio.CancelledError
            Propagated immediately so outer tasks can shut down cleanly.
        Exception
            Anything other than ``asyncio.TimeoutError`` bubbles up to the
            caller.
        """
        sum_temperatures = 0
        count_temperatures = 0
        current_time = time.monotonic()
        end_time = current_time + self.m1m3_setpoint_cadence
        timeout_time = current_time + self.ess_timeout
        warned_nan = False
        warned_timeout = False

        # Collect average indoor ESS temperature for
        # m1m3_setpoint_cadence seconds.
        while (current_time := time.monotonic()) < end_time:
            if current_time > timeout_time:
                self.log.error("No temperature samples were collected. CSC will fault.")
                raise RuntimeError("No temperature samples were collected.")

            try:
                sample = await ess_remote.tel_temperature.next(timeout=10, flush=True)
                await asyncio.sleep(0)  # Make sure we get CancelledError

                new_temperature = sample.temperatureItem[0]

                # Check for NaN, warn, and reset the timer if needed.
                if math.isnan(new_temperature):
                    end_time = time.monotonic() + self.m1m3_setpoint_cadence
                    if not warned_nan:
                        self.log.warning(
                            f"Received temperature NaN from ESS:{self.indoor_ess_index}"
                        )
                        warned_nan = True

                else:
                    sum_temperatures += new_temperature
                    count_temperatures += 1

            except asyncio.TimeoutError:
                end_time = time.monotonic() + self.m1m3_setpoint_cadence
                if not warned_timeout:
                    self.log.warning(
                        f"Timed out while getting ESS:{self.indoor_ess_index} temperature"
                    )
                    warned_timeout = True

        average_temperature = (
            sum_temperatures / count_temperatures if count_temperatures > 0 else None
        )
        self.log.debug(
            f"Collected {count_temperatures} ESS:{self.indoor_ess_index}"
            f" samples with {average_temperature=}."
        )
        return average_temperature

    async def follow_ess_indoor(
        self,
        *,
        m1m3ts_remote: salobj.Remote,
        mtmount_remote: salobj.Remote,
        future: asyncio.Future,
    ) -> None:
        self.log.debug("follow_ess_indoor")

        async with salobj.Remote(
            domain=self.domain,
            name="ESS",
            index=self.indoor_ess_index,
            include=("temperature",),
        ) as ess_remote:
            if not future.done():
                future.set_result(None)

            while True:
                if "require_dome_open" not in self.features_to_disable:
                    if self.dome_model.is_closed:
                        event = asyncio.Event()
                        self.dome_model.on_open.append(event)
                        await event.wait()

                average_temperature = await self.collect_temperature_samples(ess_remote)

                if average_temperature is None:
                    self.log.error(
                        "No temperature samples were collected. CSC will fault."
                    )
                    raise RuntimeError("No temperature samples were collected.")

                await self.start_top_end_task(mtmount_remote, average_temperature)

                if "m1m3ts" in self.features_to_disable:
                    continue

                if self.last_m1m3ts_setpoint is None:
                    # No previous setpoint = apply it regardless
                    await self.apply_setpoints(
                        m1m3ts_remote=m1m3ts_remote,
                        setpoint=average_temperature,
                    )
                elif average_temperature > self.last_m1m3ts_setpoint:
                    # Warm the mirror if the setpoint is past the deadband
                    new_setpoint = average_temperature
                    if (
                        new_setpoint - self.last_m1m3ts_setpoint
                        > self.setpoint_deadband_heating
                    ):
                        # Apply the setpoint, limited by maximum_heating_rate
                        maximum_heating_step = (
                            self.maximum_heating_rate
                            * self.m1m3_setpoint_cadence
                            / 3600.0
                        )
                        new_setpoint = min(
                            average_temperature,
                            self.last_m1m3ts_setpoint + maximum_heating_step,
                        )
                        await self.apply_setpoints(
                            m1m3ts_remote=m1m3ts_remote,
                            setpoint=new_setpoint,
                        )
                else:
                    # Cool the mirror if the setpoint is past the deadband
                    new_setpoint = average_temperature
                    if (
                        self.last_m1m3ts_setpoint - new_setpoint
                        > self.setpoint_deadband_heating
                    ):
                        # Apply the setpoint, no maximum cooling rate.
                        await self.apply_setpoints(
                            m1m3ts_remote=m1m3ts_remote,
                            setpoint=new_setpoint,
                        )

    async def wait_for_noon(
        self,
        *,
        m1m3ts_remote: salobj.Remote,
        mtmount_remote: salobj.Remote,
        future: asyncio.Future,
    ) -> None:
        """Waits for noon and then sets the room temperature.

        Waits for the timer to signal noon, and then obtains the
        temperature that was reported last night at the end
        of twilight, and then applies that temperature as the
        M1M3TS for the afternoon.

        Parameters
        ----------
        m1m3ts_remote : salobj.Remote
            A SALobj remote representing the MTM1M3TS controller.
        mtmount_remote : salobj.Remote
            A SALobj remote representing the MTMount controller.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                if not future.done():
                    future.set_result(None)

                await self.diurnal_timer.noon_condition.wait()
                if (
                    self.diurnal_timer.is_running
                    and self.weather_model.last_twilight_temperature is not None
                ):
                    self.log.info(
                        "Noon M1M3TS and top end is set based on twilight temperature: "
                        f"{self.weather_model.last_twilight_temperature:.2f}°C"
                    )
                    await self.apply_setpoints(
                        m1m3ts_remote=m1m3ts_remote,
                        setpoint=self.weather_model.last_twilight_temperature,
                    )
                    await self.start_top_end_task(
                        mtmount_remote=mtmount_remote,
                        setpoint=self.weather_model.last_twilight_temperature,
                    )
