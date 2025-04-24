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

__all__ = ["EasCsc", "run_eas"]

import asyncio
import typing
from types import SimpleNamespace

import astropy.units as u
import lsst_efd_client
import pandas as pd
from astropy.time import Time
from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from . import __version__
from .config_schema import CONFIG_SCHEMA

SAL_TIMEOUT = 5.0  # SAL telemetry/command timeout
HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state
WINDSPEED_WINDOW = 30 * 60  # Maximum age of windspeed data to consider


def run_eas() -> None:
    asyncio.run(EasCsc.amain(index=None))


class EasCsc(salobj.ConfigurableCsc):
    """Commandable SAL Component for the EAS.

    Parameters
    ----------
    config_dir : `string`
        The configuration directory
    initial_state : `salobj.State`
        The initial state of the CSC
    simulation_mode : `int`
        Simulation mode (1) or not (0)
    override : `str`, optional
        Override of settings if ``initial_state`` is `State.DISABLED`
        or `State.ENABLED`.
    """

    valid_simulation_modes = (0, 1)
    version = __version__

    def __init__(
        self,
        config_dir: typing.Optional[str] = None,
        initial_state: salobj.State = salobj.State.STANDBY,
        simulation_mode: int = 0,
        override: str = "",
    ) -> None:
        self.config: typing.Optional[SimpleNamespace] = None
        self._config_dir = config_dir
        super().__init__(
            name="EAS",
            index=0,
            config_schema=CONFIG_SCHEMA,
            config_dir=config_dir,
            initial_state=initial_state,
            simulation_mode=simulation_mode,
            override=override,
        )

    async def handle_summary_state(self) -> None:
        """Override of the handle_summary_state function to
        set up the control loop.
        """
        self.log.debug(f"handle_summary_state {salobj.State(self.summary_state).name}")

        if self.disabled_or_enabled:
            pass

        else:
            pass

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config

    async def get_wind_history(self) -> None:
        """Retrieves windspeed history from the EFD.

        The last 30 minutes of airFlow telemetry are queried from
        the EFD and stored in a table for use in `monitor_dome_shutter`.
        """
        if self.config is None:
            raise RuntimeError("Not yet configured")

        topic = "lsst.sal.ESS.airFlow"
        fields = ["speed", "private_sndStamp"]
        sal_index = 301
        end_date = Time.now()
        start_date = end_date - WINDSPEED_WINDOW * u.s
        client = lsst_efd_client.EfdClient(self.config.efd_instance)
        self.wind_history = await client.select_time_series(
            topic, fields, start_date, end_date, index=sal_index
        )

    async def air_flow_callback(self, air_flow: salobj.BaseMsgType) -> None:
        """Callback for ESS.tel_airFlow.

        This function appends new airflow data to the existing table.
        Note that `get_wind_history` must be called first.

        Parameters
        ----------
        air_flow : salobj.BaseMsgType
           A newly received air_flow telemetry item.

        """
        new_row = pd.Datagram(
            [
                {
                    "speed": air_flow.speed,
                    "private_sndStamp": air_flow.private_sndStamp,
                }
            ]
        )
        self.wind_history = pd.concat([self.wind_history, new_row], ignore_index=True)

    async def monitor_dome_shutter(self) -> None:
        """Monitors the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        if self.config is None:
            raise RuntimeError("Not yet configured")

        cached_shutter_closed = None
        cached_wind_threshold = None

        async with (
            salobj.Remote(domain=self.domain, name="MTDome") as dome_remote,
            salobj.Remote(domain=self.domain, name="HVAC") as hvac_remote,
            salobj.Remote(domain=self.domain, name="ESS", index=301) as weather_remote,
        ):
            await self.get_wind_history()
            weather_remote.tel_airFlow.callback = self.air_flow_callback

            while True:
                await asyncio.sleep(HVAC_SLEEP_TIME)

                # Check the aperture state
                aperture_shutter = dome_remote.tel_apertureShutter.get()
                if not aperture_shutter:
                    continue
                shutter_closed = (
                    aperture_shutter.positionAcutal[0] < 0.1
                    and aperture_shutter.positionActual[1] < 0.1
                )

                if not shutter_closed:
                    # Remove old wind history
                    time_horizon = utils.current_tai() - WINDSPEED_WINDOW
                    self.wind_history = self.wind_history[
                        self.wind_history["private_sndStamp"] >= time_horizon
                    ]

                    # Check windspeed threshold
                    wind_threshold = (
                        self.wind_history["speed"].mean() < self.config.wind_threshold
                    )
                    if wind_threshold != cached_wind_threshold:
                        cached_wind_threshold = wind_threshold
                        if wind_threshold:
                            # Turn on VEC-04 fan
                            await hvac_remote.cmd_enableDevice(
                                device_id=DeviceId.lowerDamperFan03P04
                            )
                        else:
                            # Turn off VEC-04 fan
                            await hvac_remote.cmd_enableDevice(
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
                        for device in ahus:
                            await hvac_remote.cmd_enableDevice(device_id=device)

                        # Disable the VEC-04 fan
                        await hvac_remote.cmd_disableDevice(
                            device_id=DeviceId.lowerDamperFan03P04
                        )
                    else:
                        for device in ahus:
                            hvac_remote.disableDevice(device_id=device)

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_ocs"
