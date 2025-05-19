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

__all__ = ["HvacModel"]

import asyncio
import logging
from collections import deque

from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state (seconds)

STD_TIMEOUT = 10  # seconds


class HvacModel:
    """A model for HVAC system automation.

    Parameters
    ----------
    domain : salobj.Domain
        A SAL domain object for obtaining remotes.
    log : logging.Logger
        A logger for log messages.
    wind_threshold : float
        Windspeed limit for the VEC-04 fan. (m/s)
    wind_average_window : float
        Time over which to average windspeed for threshold determination. (s)
    wind_minimum_window : float
        Minimum amount of time to collect wind data before acting on it. (s)
    vec04_hold_time : float
        Minimum time to wait before changing the state of the VEC-04 fan. This
        value is ignored if the dome is opened or closed. (s)
    """

    def __init__(
        self,
        *,
        domain: salobj.Domain,
        log: logging.Logger,
        wind_threshold: float = 5,
        wind_average_window: float = 30 * 60,
        wind_minimum_window: float = 10 * 60,
        vec04_hold_time: float = 5 * 60,
    ) -> None:
        self.domain = domain
        self.log = log

        self.last_vec04_time: float = (
            0  # Last time VEC-04 was changed (UNIX TAI seconds).
        )
        # A deque containing tuples of timestamp, windspeed
        self.wind_history: deque = deque()

        # Configuration parameters:
        self.wind_threshold = wind_threshold
        self.wind_average_window = wind_average_window
        self.wind_minimum_window = wind_minimum_window
        self.vec04_hold_time = vec04_hold_time

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

    async def monitor_dome_shutter(self) -> None:
        """Monitors the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        self.log.debug("monitor_dome_shutter")

        cached_shutter_closed = None
        cached_wind_threshold = None

        async with salobj.Remote(
            domain=self.domain, name="MTDome"
        ) as dome_remote, salobj.Remote(
            domain=self.domain, name="HVAC"
        ) as hvac_remote, salobj.Remote(
            domain=self.domain, name="ESS", index=301
        ) as weather_remote:
            weather_remote.tel_airFlow.callback = self.air_flow_callback

            while True:
                # Check the aperture state
                try:
                    aperture_shutter = await dome_remote.tel_apertureShutter.aget(
                        timeout=STD_TIMEOUT
                    )
                except TimeoutError:
                    self.log.error(
                        "Timeout error while trying to read apertureShutter telemetry."
                    )
                    continue
                shutter_closed = (
                    aperture_shutter.positionActual[0] < 0.1
                    and aperture_shutter.positionActual[1] < 0.1
                )

                if not shutter_closed and (
                    utils.current_tai() - self.last_vec04_time > self.vec04_hold_time
                ):
                    # Check windspeed threshold
                    average_windspeed = self.average_windspeed
                    wind_threshold = average_windspeed < self.wind_threshold
                    self.log.debug(
                        f"VEC-04 operation demanded: {average_windspeed} -> {wind_threshold}"
                    )
                    if wind_threshold != cached_wind_threshold:
                        cached_wind_threshold = wind_threshold
                        self.last_vec04_time = utils.current_tai()
                        if wind_threshold:
                            self.log.info("Turning on VEC-04 fan!")
                            await hvac_remote.cmd_enableDevice.set_start(
                                device_id=DeviceId.lowerDamperFan03P04
                            )
                        else:
                            self.log.info("Turning off VEC-04 fan!")
                            await hvac_remote.cmd_disableDevice.set_start(
                                device_id=DeviceId.lowerDamperFan03P04
                            )

                if shutter_closed != cached_shutter_closed:
                    cached_shutter_closed = shutter_closed
                    ahus = (
                        DeviceId.lowerAHU01P05,
                        DeviceId.lowerAHU02P05,
                        DeviceId.lowerAHU03P05,
                        DeviceId.lowerAHU04P05,
                    )
                    if shutter_closed:
                        # Enable the four AHUs
                        self.log.info("Enabling HVAC AHUs!")
                        for device in ahus:
                            await hvac_remote.cmd_enableDevice.set_start(
                                device_id=device
                            )

                        # Disable the VEC-04 fan
                        self.log.info("Turning off VEC-04 fan!")
                        await hvac_remote.cmd_disableDevice.set_start(
                            device_id=DeviceId.lowerDamperFan03P04
                        )
                        self.last_vec04_time = utils.current_tai()
                    else:
                        self.log.info("Disabling HVAC AHUs!")
                        for device in ahus:
                            await hvac_remote.cmd_disableDevice.set_start(
                                device_id=device
                            )

                await asyncio.sleep(HVAC_SLEEP_TIME)
