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

__all__ = ["WeatherModel"]

import asyncio
import logging
import math
import statistics
from collections import deque

import astropy.units as u
import lsst_efd_client
from astropy.time import Time
from lsst.ts import salobj, utils

from .diurnal_timer import DiurnalTimer

SAL_TIMEOUT = 60  # SAL timeout time. (seconds)


class WeatherModel:
    """A model for the weather station ESS.

    Parameters
    ----------
    domain : `~lsst.ts.salobj.Domain`
        A SAL domain object for obtaining remotes.
    log : `~logging.Logger`
        A logger for log messages.
    wind_average_window : float
        Time over which to average windspeed for threshold determination. (s)
    wind_minimum_window : float
        Minimum amount of time to collect wind data before acting on it. (s)
    """

    def __init__(
        self,
        *,
        domain: salobj.Domain,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        efd_name: str,
        ess_index: int = 301,
        wind_average_window: float = 30 * 60,
        wind_minimum_window: float = 10 * 60,
    ) -> None:
        self.domain = domain
        self.log = log
        self.diurnal_timer = diurnal_timer

        self.efd_name = efd_name
        self.ess_index = ess_index

        self.monitor_start_event = asyncio.Event()

        self.wind_average_window = wind_average_window
        self.wind_minimum_window = wind_minimum_window

        # A deque containing tuples of timestamp, windspeed
        self.wind_history: deque = deque()

        # A deque containing tuples of timestamp, temperature
        self.temperature_history: deque = deque(maxlen=20)

        # The last observed temperature at the end of twilight
        self.last_twilight_temperature: float | None = None

    async def get_last_twilight_temperature(self) -> float | None:
        """Retrieve the most recent twilight temperature from the EFD.

        This method searches for the temperature at the twilight time of
        interest, going back up to 10 days from the current time. It queries
        the EFD for a one-minute interval starting at the twilight time for
        each day until a valid median temperature is found. The result is
        cached in `self.last_twilight_temperature` for future calls.

        If no valid data is found within the last 10 days, returns None.

        Returns
        -------
        float | None
            The median temperature (°C) at the most recent twilight time found,
            or None if no usable result is available.
        """
        if self.last_twilight_temperature is not None:
            return self.last_twilight_temperature

        of_date = Time.now()
        efd_client = lsst_efd_client.EfdClient(self.efd_name)
        for days_ago in range(10):
            # Get time of twilight of interest.
            of_date -= 86400 * u.s

            twilight_time = self.diurnal_timer.get_twilight_time(of_date)

            time_series = await efd_client.select_time_series(
                "lsst.sal.ESS.temperature",
                ["temperatureItem0"],
                twilight_time,
                twilight_time + 60 * u.s,
                index=self.ess_index,
            )
            if len(time_series) == 0:
                continue

            self.last_twilight_temperature = float(
                time_series["temperatureItem0"].median()
            )
            if math.isnan(self.last_twilight_temperature):
                self.last_twilight_temperature = None
                continue

            self.log.info(
                "Obtained twilight temperature from EFD for "
                f"{twilight_time.isot}: {self.last_twilight_temperature}"
            )
            return self.last_twilight_temperature
        else:
            # Unable to find a usable result.
            return None

    async def air_flow_callback(self, air_flow: salobj.BaseMsgType) -> None:
        """Callback for ESS.tel_airFlow.

        This function appends new airflow data to the existing table.

        Parameters
        ----------
        air_flow : `~lsst.ts.salobj.BaseMsgType`
           A newly received air_flow telemetry item.
        """
        now = air_flow.private_sndStamp
        self.wind_history.append((air_flow.speed, now))

        # Prune old data
        time_horizon = now - self.wind_average_window
        while self.wind_history and self.wind_history[0][1] < time_horizon:
            self.wind_history.popleft()

    async def temperature_callback(self, temperature: salobj.BaseMsgType) -> None:
        """Callback for ESS.tel_temperature.

        This function appends new temperature data to the existing table.

        Parameters
        ----------
        temperature : `~lsst.ts.salobj.BaseMsgType`
           A newly received temperature telemetry item.
        """
        now = temperature.private_sndStamp
        self.temperature_history.append((temperature.temperatureItem[0], now))

    @property
    def average_windspeed(self) -> float:
        """Average windspeed in m/s.

        The measurement returned by this function uses ESS CSC telemetry
        (collected while the monitor loop runs).

        Returns
        -------
        float
            The average of all wind speed samples collected
            in the past `self.wind_average_window` seconds.
            If the oldest sample is newer than
            `config.wind_minimum_window` seconds old, then
            NaN is returned. Units are m/s.
        """
        if not self.wind_history:
            return math.nan

        current_time = utils.current_tai()
        time_horizon = current_time - self.wind_average_window

        # Remove old entries first
        while self.wind_history and self.wind_history[0][1] < time_horizon:
            self.wind_history.popleft()

        # If not enough data remains, return NaN
        if (
            not self.wind_history
            or current_time - self.wind_history[0][1] < self.wind_minimum_window
        ):
            return math.nan

        # Compute average directly
        speeds = [s for s, _ in self.wind_history]
        return sum(speeds) / len(speeds)

    @property
    def current_temperature(self) -> float:
        """Current temperature in °C.

        Returns
        -------
        float
            The average of the last 10 temperature samples from
            the ESS, or NaN if there are no temperature samples
            in the last 5 minutes.
        """
        cutoff = utils.current_tai() - 5 * 60
        recent_samples = [
            temperature
            for temperature, time in self.temperature_history
            if time >= cutoff
        ]

        if not recent_samples:
            return math.nan

        return statistics.median(recent_samples)

    async def monitor(self) -> None:
        """Monitors the temperature and windspeed to control the HVAC.

        This monitor does the following:
         * Sets a callback for ESS.airFlow to collect average windspeed.
         * Waits for evening twilight and then gets a temperature measurement.
        """
        self.log.debug("WeatherModel.monitor")

        async with salobj.Remote(
            domain=self.domain,
            name="ESS",
            index=self.ess_index,
            include=("airFlow", "temperature"),
        ) as weather_remote:
            weather_remote.tel_airFlow.callback = self.air_flow_callback
            weather_remote.tel_temperature.callback = self.temperature_callback

            while self.diurnal_timer.is_running:
                async with self.diurnal_timer.twilight_condition:
                    self.monitor_start_event.set()

                    await self.diurnal_timer.twilight_condition.wait()
                    if self.diurnal_timer.is_running:
                        try:
                            self.last_twilight_temperature = (
                                await self.measure_twilight_temperature()
                            )

                        except Exception:
                            self.log.exception("Failed to read temperature from ESS")
                            self.last_twilight_temperature = None
                            self.monitor_start_event.clear()
                            raise

        self.monitor_start_event.clear()

    async def measure_twilight_temperature(self) -> float:
        """Refreshes the twilight temperature with a new measurement."""
        # Store the current temperature for future use.
        last_twilight_temperature = self.current_temperature
        self.log.info(f"Collected twilight temperature: {last_twilight_temperature}°C")
        return last_twilight_temperature
