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

__all__ = ["M1M3TSModel"]

import logging

from lsst.ts import salobj

from .diurnal_timer import DiurnalTimer
from .weather_model import WeatherModel

STD_TIMEOUT = 10  # seconds


class M1M3TSModel:
    """A model for the M1M3TS system automation.

    Parameters
    ----------
    domain : salobj.Domain
        A SAL domain object for obtaining remotes.
    log : logging.Logger
        A logger for log messages.
    diurnal_timer : DiurnalTimer
        A timer that signals at noon and at the end of evening twilight.
    weather_model : WeatherModel
        Source for twilight ambient temperature.
    glycol_setpoint_delta : float
        The difference between the twilight ambient temperature and the
        setpoint to apply for the glycol, e.g., -2 if the glycol should
        be two degrees cooler than ambient.
    heater_setpoint_delta : float
        The difference between the twilight ambient temperature and the
        setpoint to apply for the FCU heaters, e.g., -1 if the FCU heaters
        should be one degree cooler than ambient.
    features_to_disable : list[str]
        A list of features that should be disabled. The following strings can
        be used:
         * m1m3ts
        Any other values are ignored.
    """

    def __init__(
        self,
        *,
        domain: salobj.Domain,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        weather_model: WeatherModel,
        glycol_setpoint_delta: float,
        heater_setpoint_delta: float,
        features_to_disable: list[str] = [],
    ) -> None:
        self.domain = domain
        self.log = log
        self.diurnal_timer = diurnal_timer

        self.last_vec04_time: float = (
            0  # Last time VEC-04 was changed (UNIX TAI seconds).
        )

        # Configuration parameters:
        self.weather_model = weather_model
        self.glycol_setpoint_delta = glycol_setpoint_delta
        self.heater_setpoint_delta = heater_setpoint_delta
        self.features_to_disable = features_to_disable

    async def monitor(self) -> None:
        self.log.debug("HvacModel.monitor")

        async with salobj.Remote(domain=self.domain, name="MTM1M3TS") as m1m3ts_remote:
            await self.wait_for_noon(m1m3ts_remote=m1m3ts_remote)

    async def wait_for_noon(self, m1m3ts_remote: salobj.Remote) -> None:
        """Waits for noon and then sets the M1M3TS setpoint.

        Waits for the timer to signal noon, and then obtains the
        temperature that was reported last night at the end
        of twilight, and then applies that temperature as the
        M1M3TS setpoints.

        Parameters
        ----------
        m1m3ts_remote : salobj.Remote
            A SALobj remote representing the M1M3TS.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                await self.diurnal_timer.noon_condition.wait()
                if (
                    self.diurnal_timer.is_running
                    and self.weather_model.last_twilight_temperature is not None
                ):
                    if "m1m3ts" not in self.features_to_disable:
                        twilight_temperature = (
                            self.weather_model.last_twilight_temperature
                        )
                        glycol_setpoint = (
                            twilight_temperature + self.glycol_setpoint_delta
                        )
                        heaters_setpoint = (
                            twilight_temperature + self.heater_setpoint_delta
                        )
                        self.log.info(
                            f"Setting MTM1MTS: {glycol_setpoint=} {heaters_setpoint=}"
                        )
                        await m1m3ts_remote.cmd_applySetpoints.set_start(
                            glycolSetpoint=glycol_setpoint,
                            heatersSetpoint=heaters_setpoint,
                        )
