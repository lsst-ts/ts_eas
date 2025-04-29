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

import numpy as np
import pandas as pd
from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from . import __version__
from .config_schema import CONFIG_SCHEMA

HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state (seconds)

# Constants for the monitor_dome_shutter health monitor:
MAX_FAILURES = 15  # Maximum number of failures allowed before the CSC will fault.
FAILURE_TIMEOUT = 600  # Failure count will reset after monitor_dome_shutter has run for this time (seconds).
INITIAL_BACKOFF = 1  # monitor_dome_shutter initial retry delay (seconds)
MAX_BACKOFF = 60  # monitor_dome_shutter maximum retry delay (seconds)

STD_TIMEOUT = 10  # seconds

# Error codes
DOME_MONITOR_FAILED = 101


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

        self.last_vec04_time: float = (
            0  # Last time VEC-04 was changed (UNIX TAI seconds).
        )
        self.health_monitor_task = utils.make_done_future()
        self.wind_history = pd.DataFrame(
            {
                "speed": pd.Series(dtype=float),
                "private_sndStamp": pd.Series(dtype=float),
            }
        )

    async def handle_summary_state(self) -> None:
        """Override of the handle_summary_state function to
        set up the control loop.
        """
        self.log.debug(f"handle_summary_state {salobj.State(self.summary_state).name}")

        if self.disabled_or_enabled:
            if self.health_monitor_task.done():
                self.health_monitor_task = asyncio.create_task(self.health_monitor())

        else:
            await self.health_monitor_shutdown()

    async def health_monitor_shutdown(self) -> None:
        self.health_monitor_task.cancel()
        try:
            await self.health_monitor_task
        except asyncio.CancelledError:
            pass

    async def close_tasks(self) -> None:
        """Stop active tasks."""
        await self.health_monitor_shutdown()
        await super().close_tasks()

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config

    async def health_monitor(self) -> None:
        """Manages the `monitor_dome_shutter` control loop.

        Manages the `monitor_dome_shutter` control loop with backoff and
        time-based failure reset, which is (hopefully) robust against
        disconnects of the remotes, as well as any other form of
        recoverable failure.
        """
        self.log.debug("health_monitor")
        failure_count = 0
        last_attempt_time = utils.current_tai()

        while True:
            now = utils.current_tai()

            # Reset failure count if it's been too long since last attempt
            if now - last_attempt_time > FAILURE_TIMEOUT:
                self.log.info(
                    f"Resetting monitor failure count: {now - last_attempt_time:.1f}s since last attempt"
                )
                failure_count = 0

            last_attempt_time = now

            try:
                await self.monitor_dome_shutter()
            except Exception as ex:
                self.log.exception("monitor_dome_shutter threw exception")

                failure_count += 1
                if failure_count >= MAX_FAILURES:
                    self.log.error("Too many failures in succession. CSC will fault.")
                    await self.fault(code=DOME_MONITOR_FAILED, report=str(ex))
                    return

                # Exponential backoff with cap
                backoff = min(INITIAL_BACKOFF * 2 ** (failure_count - 1), MAX_BACKOFF)
                self.log.info(f"Backing off for {backoff:.1f} seconds")
                await asyncio.sleep(backoff)

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
        new_row = pd.DataFrame(
            [
                {
                    "speed": air_flow.speed,
                    "private_sndStamp": air_flow.private_sndStamp,
                }
            ]
        )
        self.wind_history = pd.concat([self.wind_history, new_row], ignore_index=True)

    @property
    def average_windspeed(self) -> float:
        """Average windspeed in m/s.

        The measurement returned by this function uses ESS CSC telemetry
        (collected while the monitor loop runs).

        Returns
        -------
        float
            The average of all wind speed samples collected
            in the past `self.config.wind_average_window` seconds.
            If the oldest sample is newer than
            `config.wind_minimum_window` seconds old, then
            NaN is returned. Units are m/s.
        """
        if self.config is None:
            raise RuntimeError("Not yet configured")

        if len(self.wind_history) < 1:
            return np.nan

        # Remove old wind history
        current_time = utils.current_tai()
        time_horizon = current_time - self.config.wind_average_window
        oldest_sample_age = current_time - self.wind_history["private_sndStamp"].min()

        if oldest_sample_age < self.config.wind_minimum_window:
            return np.nan

        self.wind_history = self.wind_history[
            self.wind_history["private_sndStamp"] >= time_horizon
        ]

        return self.wind_history["speed"].mean()

    async def monitor_dome_shutter(self) -> None:
        """Monitors the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        if self.config is None:
            raise RuntimeError("Not yet configured")

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
                    utils.current_tai() - self.last_vec04_time
                    > self.config.vec04_hold_time
                ):
                    # Check windspeed threshold
                    wind_threshold = self.average_windspeed < self.config.wind_threshold
                    self.log.debug(f"VEC-04 operation demanded: {wind_threshold}")
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

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_ocs"
