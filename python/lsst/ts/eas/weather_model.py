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
from collections import deque

from lsst.ts import salobj, utils


class WeatherModel:
    """A model for the weather station ESS.

    Parameters
    ----------
    domain : salobj.Domain
        A SAL domain object for obtaining remotes.
    log : logging.Logger
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
        wind_average_window: float = 30 * 60,
        wind_minimum_window: float = 10 * 60,
    ) -> None:
        self.domain = domain
        self.log = log

        self.wind_average_window = wind_average_window
        self.wind_minimum_window = wind_minimum_window

        # A deque containing tuples of timestamp, windspeed
        self.wind_history: deque = deque()

    async def air_flow_callback(self, air_flow: salobj.BaseMsgType) -> None:
        """Callback for ESS.tel_airFlow.

        This function appends new airflow data to the existing table.

        Parameters
        ----------
        air_flow : salobj.BaseMsgType
           A newly received air_flow telemetry item.

        """
        self.log.debug(
            f"air_flow_callback: {air_flow.private_sndStamp=} {air_flow.speed=}"
        )
        now = air_flow.private_sndStamp
        self.wind_history.append((air_flow.speed, now))

        # Prune old data
        time_horizon = now - self.wind_average_window
        while self.wind_history and self.wind_history[0][1] < time_horizon:
            self.wind_history.popleft()

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
            return float("nan")

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
            return float("nan")

        # Compute average directly
        speeds = [s for s, _ in self.wind_history]
        return sum(speeds) / len(speeds)

    async def monitor(self) -> None:
        """Monitors the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        self.log.debug("WeatherModel.monitor")

        async with salobj.Remote(
            domain=self.domain, name="ESS", index=301
        ) as weather_remote:
            weather_remote.tel_airFlow.callback = self.air_flow_callback

            while True:
                # TODO: add measurement of temperature at end of twilight
                await asyncio.sleep(3600)
