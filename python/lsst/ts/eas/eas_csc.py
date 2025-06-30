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
import math
import typing
from types import SimpleNamespace

from lsst.ts import salobj, utils

from . import __version__
from .config_schema import CONFIG_SCHEMA
from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .hvac_model import HvacModel
from .m1m3ts_model import M1M3TSModel
from .weather_model import WeatherModel

# Constants for the health monitor:
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

        self.monitor_start_event = asyncio.Event()

        self.health_monitor_task = utils.make_done_future()
        self.subtasks: list[asyncio.Task[None]] = []
        self.diurnal_timer: DiurnalTimer | None = None
        self.hvac_model: HvacModel | None = None
        self.m1m3ts_model: M1M3TSModel | None = None

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
        tasks_to_cancel = self.subtasks
        self.subtasks = []
        tasks_to_cancel.append(self.health_monitor_task)
        for task in tasks_to_cancel:
            task.cancel()

        for task in tasks_to_cancel:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def close_tasks(self) -> None:
        """Stop active tasks."""
        await self.shutdown_health_monitor()
        if self.diurnal_timer is not None:
            await self.diurnal_timer.stop()
        await super().close_tasks()

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config
        self.diurnal_timer = DiurnalTimer(sun_altitude=self.config.twilight_definition)
        self.dome_model = DomeModel(domain=self.domain)
        self.weather_model = WeatherModel(
            domain=self.domain,
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            dome_model=self.dome_model,
            ess_index=self.config.weather_ess_index,
            wind_average_window=self.config.wind_average_window,
            wind_minimum_window=self.config.wind_minimum_window,
        )
        self.hvac_model = HvacModel(
            domain=self.domain,
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            dome_model=self.dome_model,
            weather_model=self.weather_model,
            setpoint_lower_limit=self.config.setpoint_lower_limit,
            wind_threshold=self.config.wind_threshold,
            vec04_hold_time=self.config.vec04_hold_time,
            features_to_disable=self.config.features_to_disable,
        )
        self.m1m3ts_model = M1M3TSModel(
            domain=self.domain,
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            dome_model=self.dome_model,
            weather_model=self.weather_model,
            indoor_ess_index=self.config.indoor_ess_index,
            ess_timeout=self.config.ess_timeout,
            setpoint_lower_limit=self.config.setpoint_lower_limit,
            glycol_setpoint_delta=self.config.glycol_setpoint_delta,
            heater_setpoint_delta=self.config.heater_setpoint_delta,
            top_end_setpoint_delta=self.config.top_end_setpoint_delta,
            m1m3_setpoint_cadence=self.config.m1m3_setpoint_cadence,
            setpoint_deadband_heating=self.config.setpoint_deadband_heating,
            setpoint_deadband_cooling=self.config.setpoint_deadband_cooling,
            maximum_heating_rate=self.config.maximum_heating_rate,
            features_to_disable=self.config.features_to_disable,
        )

    async def monitor_health(self) -> None:
        """Manages the `monitor_dome_shutter` control loop.

        Manages the `monitor_dome_shutter` control loop with backoff and
        time-based failure reset, which is (hopefully) robust against
        disconnects of the remotes, as well as any other form of
        recoverable failure.
        """
        assert self.hvac_model is not None, "HVAC Model not initialized."
        assert self.m1m3ts_model is not None, "M1M3TS Model not initialized."
        assert self.diurnal_timer is not None, "Timer not initialized."

        self.log.debug("monitor_health")

        while self.disabled_or_enabled:
            self.diurnal_timer.start()

            self.subtasks = [
                asyncio.create_task(coro())
                for coro in (
                    self.dome_model.monitor,
                    self.weather_model.monitor,
                    self.hvac_model.monitor,
                    self.m1m3ts_model.monitor,
                )
            ]

            # Wait for start and then signal
            await asyncio.gather(
                self.dome_model.monitor_start_event.wait(),
                self.weather_model.monitor_start_event.wait(),
                self.hvac_model.monitor_start_event.wait(),
                self.m1m3ts_model.monitor_start_event.wait(),
            )
            self.log.debug("Monitors started.")
            self.monitor_start_event.set()

            done, pending = await asyncio.wait(
                self.subtasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.subtasks = []
            self.monitor_start_event.clear()

            self.log.debug("At least one monitor task ended.")

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                try:
                    task.result()

                except Exception as ex:
                    self.log.exception("A monitor threw exception")
                    await self.fault(code=DOME_MONITOR_FAILED, report=str(ex))
                    self.monitor_start_event.clear()
                    return

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
            self.weather_model.average_windspeed
            if self.weather_model is not None
            else math.nan
        )

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_ocs"
