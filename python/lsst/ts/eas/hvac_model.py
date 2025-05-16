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

from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from .weather_model import WeatherModel

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
    vec04_hold_time : float
        Minimum time to wait before changing the state of the VEC-04 fan. This
        value is ignored if the dome is opened or closed. (s)
    """

    def __init__(
        self,
        *,
        domain: salobj.Domain,
        log: logging.Logger,
        weather_model: WeatherModel,
        wind_threshold: float = 5,
        vec04_hold_time: float = 5 * 60,
    ) -> None:
        self.domain = domain
        self.log = log

        self.last_vec04_time: float = (
            0  # Last time VEC-04 was changed (UNIX TAI seconds).
        )

        # Configuration parameters:
        self.weather_model = weather_model
        self.wind_threshold = wind_threshold
        self.vec04_hold_time = vec04_hold_time

    async def monitor(self) -> None:
        """Monitors the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        self.log.debug("HvacModel.monitor")

        cached_shutter_closed = None
        cached_wind_threshold = None

        async with salobj.Remote(
            domain=self.domain, name="MTDome"
        ) as dome_remote, salobj.Remote(domain=self.domain, name="HVAC") as hvac_remote:
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
                    average_windspeed = self.weather_model.average_windspeed
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
