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

from lsst.ts import salobj, utils

from . import __version__
from .config_schema import CONFIG_SCHEMA
from .hvac_model import HvacModel

# Constants for the health monitor:
MAX_FAILURES = 15  # Maximum number of failures allowed before the CSC will fault.
FAILURE_TIMEOUT = (
    600  # Failure count will reset after monitor has run for this time (seconds).
)
INITIAL_BACKOFF = 1  # monitor initial retry delay (seconds)
MAX_BACKOFF = 60  # monitor maximum retry delay (seconds)

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

        self.health_monitor_task = utils.make_done_future()
        self.hvac_model: HvacModel | None = None

    async def handle_summary_state(self) -> None:
        """Override of the handle_summary_state function to
        set up the control loop.
        """
        self.log.debug(f"handle_summary_state {salobj.State(self.summary_state).name}")

        if self.disabled_or_enabled:
            if self.health_monitor_task.done():
                self.health_monitor_task = asyncio.create_task(self.monitor_health())

        else:
            await self.shutdown_health_monitor()

    async def shutdown_health_monitor(self) -> None:
        """Cancels the health monitor task and waits for completion."""
        self.health_monitor_task.cancel()
        try:
            await self.health_monitor_task
        except asyncio.CancelledError:
            pass

    async def close_tasks(self) -> None:
        """Stop active tasks."""
        await self.shutdown_health_monitor()
        await super().close_tasks()

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config
        self.hvac_model = HvacModel(
            domain=self.domain,
            log=self.log,
            wind_threshold=self.config.wind_threshold,
            wind_average_window=self.config.wind_average_window,
            wind_minimum_window=self.config.wind_minimum_window,
            vec04_hold_time=self.config.vec04_hold_time,
        )

    async def monitor_health(self) -> None:
        """Manages the `monitor_dome_shutter` control loop.

        Manages the `monitor_dome_shutter` control loop with backoff and
        time-based failure reset, which is (hopefully) robust against
        disconnects of the remotes, as well as any other form of
        recoverable failure.
        """
        assert self.hvac_model is not None, "HVAC Model not initialized."

        self.log.debug("monitor_health")
        failure_count = 0
        last_attempt_time = utils.current_tai()

        while self.disabled_or_enabled:
            now = utils.current_tai()

            # Reset failure count if it's been too long since last attempt
            if now - last_attempt_time > FAILURE_TIMEOUT:
                self.log.info(
                    f"Resetting monitor failure count: {now - last_attempt_time:.1f}s since last attempt"
                )
                failure_count = 0

            last_attempt_time = now

            try:
                await self.hvac_model.monitor_dome_shutter()
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
        return (
            self.hvac_model.average_windspeed
            if self.hvac_model is not None
            else float("nan")
        )

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_ocs"
