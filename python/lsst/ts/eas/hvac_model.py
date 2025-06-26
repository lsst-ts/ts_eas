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

from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .weather_model import WeatherModel

HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state (seconds)
STD_TIMEOUT = 5  # seconds


class HvacModel:
    """A model for HVAC system automation.

    Parameters
    ----------
    domain : salobj.Domain
        A SAL domain object for obtaining remotes.
    log : logging.Logger
        A logger for log messages.
    diurnal_timer : DiurnalTimer
        A timer that signals at noon and at the end of evening twilight.
    dome_model : DomeModel
        A model representing the dome state.
    wind_threshold : float
        Windspeed limit for the VEC-04 fan. (m/s)
    vec04_hold_time : float
        Minimum time to wait before changing the state of the VEC-04 fan. This
        value is ignored if the dome is opened or closed. (s)
    disable_features: list[str]
        A list of features that should be disabled. The following strings can
        be used:
         * vec04
         * ahu
         * room_setpoint
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
        wind_threshold: float = 5,
        vec04_hold_time: float = 5 * 60,
        features_to_disable: list[str] = [],
    ) -> None:
        self.domain = domain
        self.log = log
        self.diurnal_timer = diurnal_timer

        self.monitor_start_event = asyncio.Event()

        self.last_vec04_time: float = (
            0  # Last time VEC-04 was changed (UNIX TAI seconds).
        )

        # Configuration parameters:
        self.dome_model = dome_model
        self.weather_model = weather_model
        self.wind_threshold = wind_threshold
        self.vec04_hold_time = vec04_hold_time
        self.features_to_disable = features_to_disable

    async def monitor(self) -> None:
        """Monitors the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        self.log.debug("HvacModel.monitor")

        # Give the dome model an opportunity to collect some telemetry...
        await asyncio.sleep(STD_TIMEOUT)

        async with salobj.Remote(
            domain=self.domain,
            name="HVAC",
            include=("enableDevice", "disableDevice", "configAhu"),
        ) as hvac_remote:
            hvac_future = asyncio.gather(
                self.control_ahus_and_vec04(hvac_remote=hvac_remote),
                self.wait_for_noon(hvac_remote=hvac_remote),
            )
            self.monitor_start_event.clear()

            try:
                await hvac_future
            except asyncio.CancelledError:
                hvac_future.cancel()
                await asyncio.gather(hvac_future, return_exceptions=True)
                raise

    async def control_ahus_and_vec04(self, *, hvac_remote: salobj.Remote) -> None:
        cached_shutter_closed = None
        cached_wind_threshold = None

        while True:
            # Check the aperture state
            shutter_closed = self.dome_model.is_closed
            if shutter_closed is None:
                await asyncio.sleep(0.1)
                continue

            if (
                "vec04" not in self.features_to_disable
                and not shutter_closed
                and (utils.current_tai() - self.last_vec04_time > self.vec04_hold_time)
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
                    if "ahu" not in self.features_to_disable:
                        # Enable the four AHUs
                        self.log.info("Enabling HVAC AHUs!")
                        for device in ahus:
                            await hvac_remote.cmd_enableDevice.set_start(
                                device_id=device
                            )

                    if "vec04" not in self.features_to_disable:
                        # Disable the VEC-04 fan
                        self.log.info("Turning off VEC-04 fan!")
                        await hvac_remote.cmd_disableDevice.set_start(
                            device_id=DeviceId.lowerDamperFan03P04
                        )
                        self.last_vec04_time = utils.current_tai()
                else:
                    if "ahu" not in self.features_to_disable:
                        self.log.info("Disabling HVAC AHUs!")
                        for device in ahus:
                            await hvac_remote.cmd_disableDevice.set_start(
                                device_id=device
                            )

            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def wait_for_noon(self, *, hvac_remote: salobj.Remote) -> None:
        """Waits for noon and then sets the room temperature.

        Waits for the timer to signal noon, and then obtains the
        temperature that was reported last night at the end
        of twilight, and then applies that temperature as at AHU
        setpoint.

        Parameters
        ----------
        hvac_remote : salobj.Remote
            A SALobj remote representing the HVAC.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                self.monitor_start_event.set()

                await self.diurnal_timer.noon_condition.wait()
                if (
                    self.diurnal_timer.is_running
                    and self.weather_model.last_twilight_temperature is not None
                ):
                    if "room_setpoint" not in self.features_to_disable:
                        # Time to set the room setpoint based on last twilight
                        for device_id in (
                            DeviceId.lowerAHU01P05,
                            DeviceId.lowerAHU02P05,
                            DeviceId.lowerAHU03P05,
                            DeviceId.lowerAHU04P05,
                        ):
                            await hvac_remote.cmd_configLowerAhu.set_start(
                                device_id=device_id,
                                workingSetpoint=self.weather_model.last_twilight_temperature,
                                maxFanSetpoint=math.nan,
                                minFanSetpoint=math.nan,
                                antiFreezeTemperature=math.nan,
                            )
