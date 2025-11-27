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
from .glass_temperature_model import GlassTemperatureModel
from .hvac_model import HvacModel
from .tma_model import TmaModel
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

THERMAL_SCANNER_1_INDEX = 114
THERMAL_SCANNER_2_INDEX = 115
THERMAL_SCANNER_3_INDEX = 116
THERMAL_SCANNER_4_INDEX = 117


def run_eas() -> None:
    asyncio.run(EasCsc.amain(index=None))


class EasCsc(salobj.ConfigurableCsc):
    """Commandable SAL Component for the EAS.

    Parameters
    ----------
    config_dir : `str`
        The configuration directory
    initial_state : `~lsst.ts.salobj.State`
        The initial state of the CSC
    simulation_mode : `int`
        Simulation mode (1) or not (0)
    override : `str`, optional
        Override of settings if `initial_state` is `State.DISABLED`
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
        self.tma_model: TmaModel | None = None

        self.dome_model: DomeModel | None = None
        self.glass_temperature_model: GlassTemperatureModel | None = None
        self.weather_model: WeatherModel | None = None

        self.dome_remote = salobj.Remote(
            domain=self.domain,
            name="MTDome",
            readonly=True,
            include=["apertureShutter", "louvers"],
        )
        self.ess_ts1_remote = salobj.Remote(
            domain=self.domain,
            name="ESS",
            index=THERMAL_SCANNER_1_INDEX,
            readonly=True,
            include=["temperature"],
        )
        self.ess_ts2_remote = salobj.Remote(
            domain=self.domain,
            name="ESS",
            index=THERMAL_SCANNER_2_INDEX,
            include=["temperature"],
        )
        self.ess_ts3_remote = salobj.Remote(
            domain=self.domain,
            name="ESS",
            index=THERMAL_SCANNER_3_INDEX,
            include=["temperature"],
        )
        self.ess_ts4_remote = salobj.Remote(
            domain=self.domain,
            name="ESS",
            index=THERMAL_SCANNER_4_INDEX,
            include=["temperature"],
        )
        self.mtm1m3ts_remote = salobj.Remote(
            domain=self.domain,
            name="MTM1M3TS",
            include=["appliedSetpoints"],
        )
        self.mtmount_remote = salobj.Remote(
            domain=self.domain,
            name="MTMount",
            include=["summaryState"],
        )
        self.hvac_remote = salobj.Remote(
            domain=self.domain,
            name="HVAC",
            include=[],
        )

        self.ess_indoor_remote: salobj.Remote | None = None
        self.ess_outdoor_remote: salobj.Remote | None = None
        self.ess_indoor_remote_index: int | None = None
        self.ess_outdoor_remote_index: int | None = None

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
        """Cancel the health monitor task and waits for completion."""
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

    async def construct_ess_remotes(
        self, *, indoor_ess_index: int, outdoor_ess_index: int
    ) -> None:
        if (
            self.ess_indoor_remote is None
            or self.ess_indoor_remote_index != indoor_ess_index
        ):
            if self.ess_indoor_remote is not None:
                await self.ess_indoor_remote.close()
            self.ess_indoor_remote = salobj.Remote(
                domain=self.domain,
                name="ESS",
                index=indoor_ess_index,
                readonly=True,
                include=["temperature", "dewPoint"],
            )
            self.ess_indoor_remote_index = indoor_ess_index

        if (
            self.ess_outdoor_remote is None
            or self.ess_outdoor_remote_index != outdoor_ess_index
        ):
            if self.ess_outdoor_remote is not None:
                await self.ess_outdoor_remote.close()
            self.ess_outdoor_remote = salobj.Remote(
                domain=self.domain,
                name="ESS",
                index=outdoor_ess_index,
                readonly=True,
                include=["temperature", "airFlow"],
            )
            self.ess_outdoor_remote_index = outdoor_ess_index

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config

        await self.construct_ess_remotes(
            indoor_ess_index=config.weather["indoor_ess_index"],
            outdoor_ess_index=config.weather["ess_index"],
        )

        if self.ess_indoor_remote is None or self.ess_outdoor_remote is None:
            raise RuntimeError(
                "The ESS indoor and outdoor temperature remotes did not "
                "initialize. This is likely caused by an incorrectly "
                "specified SAL index in the configuration file. Check "
                "the configuration file and try again."
            )

        # Make sure temperature probe Remotes have started.
        await asyncio.gather(
            self.ess_indoor_remote.start_task,
            self.ess_outdoor_remote.start_task,
        )

        if self.diurnal_timer is not None:
            await self.diurnal_timer.stop()
        self.diurnal_timer = DiurnalTimer(sun_altitude=self.config.twilight_definition)
        await self.diurnal_timer.start()

        if self.dome_model is not None:
            self.dome_model.cancel_pending_events()

        self.glass_temperature_model = GlassTemperatureModel(log=self.log)

        # Validate the sub-schemas and update with defaults.
        for object_type, attr in (
            (WeatherModel, "weather"),
            (HvacModel, "hvac"),
            (TmaModel, "tma"),
            (DomeModel, "dome"),
        ):
            schema = object_type.get_config_schema()
            validator = salobj.DefaultingValidator(schema)
            setattr(config, attr, validator.validate(getattr(config, attr)))

        self.dome_model = DomeModel(log=self.log, **self.config.dome)

        self.weather_model = WeatherModel(
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            **self.config.weather,
        )
        self.hvac_model = HvacModel(
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            dome_model=self.dome_model,
            weather_model=self.weather_model,
            hvac_remote=self.hvac_remote,
            features_to_disable=self.config.features_to_disable,
            **self.config.hvac,
        )
        self.tma_model = TmaModel(
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            dome_model=self.dome_model,
            glass_temperature_model=self.glass_temperature_model,
            weather_model=self.weather_model,
            m1m3ts_remote=self.mtm1m3ts_remote,
            mtmount_remote=self.mtmount_remote,
            features_to_disable=self.config.features_to_disable,
            **self.config.tma,
        )

    def connect_callbacks(self) -> None:
        """Connects callbacks to their remotes."""

        # Models should be initialized before this method is called.
        assert self.dome_model is not None, "Dome model not initialized."
        assert self.glass_temperature_model is not None, "Glass model not initialized."
        assert self.weather_model is not None, "Weather Model not initialized."

        if self.ess_indoor_remote is None or self.ess_outdoor_remote is None:
            raise RuntimeError(
                "The ESS indoor and outdoor temperature remotes did not "
                "initialize. This is likely caused by an incorrectly "
                "specified SAL index in the configuration file. Check "
                "the configuration file and try again."
            )

        self.dome_remote.tel_apertureShutter.callback = (
            self.dome_model.aperture_shutter_callback
        )
        self.dome_remote.tel_louvers.callback = self.dome_model.louvers_callback
        self.ess_ts1_remote.tel_temperature.callback = (
            self.glass_temperature_model.temperature_callback
        )
        self.ess_ts2_remote.tel_temperature.callback = (
            self.glass_temperature_model.temperature_callback
        )
        self.ess_ts3_remote.tel_temperature.callback = (
            self.glass_temperature_model.temperature_callback
        )
        self.ess_ts4_remote.tel_temperature.callback = (
            self.glass_temperature_model.temperature_callback
        )
        self.ess_outdoor_remote.tel_airFlow.callback = (
            self.weather_model.air_flow_callback
        )
        self.ess_outdoor_remote.tel_temperature.callback = (
            self.weather_model.temperature_callback
        )
        self.ess_indoor_remote.tel_dewPoint.callback = (
            self.weather_model.indoor_dew_point_callback
        )
        self.ess_indoor_remote.tel_temperature.callback = (
            self.weather_model.indoor_temperature_callback
        )

    def disconnect_callbacks(self) -> None:
        """Disconnects callbacks from their remotes."""
        if self.ess_indoor_remote is None or self.ess_outdoor_remote is None:
            raise RuntimeError(
                "The ESS indoor and outdoor temperature remotes did not "
                "initialize. This is likely caused by an incorrectly "
                "specified SAL index in the configuration file. Check "
                "the configuration file and try again."
            )

        self.dome_remote.tel_apertureShutter.callback = None
        self.ess_ts1_remote.tel_temperature.callback = None
        self.ess_ts2_remote.tel_temperature.callback = None
        self.ess_ts3_remote.tel_temperature.callback = None
        self.ess_ts4_remote.tel_temperature.callback = None
        self.ess_outdoor_remote.tel_airFlow.callback = None
        self.ess_outdoor_remote.tel_temperature.callback = None
        self.ess_indoor_remote.tel_dewPoint.callback = None
        self.ess_indoor_remote.tel_temperature.callback = None

    async def monitor_health(self) -> None:
        """Manage the `monitor_dome_shutter` control loop.

        Manage the `monitor_dome_shutter` control loop with backoff and
        time-based failure reset, which is (hopefully) robust against
        disconnects of the remotes, as well as any other form of
        recoverable failure.
        """
        assert self.hvac_model is not None, "HVAC Model not initialized."
        assert self.tma_model is not None, "TMA Model not initialized."
        assert self.diurnal_timer is not None, "Timer not initialized."
        assert self.glass_temperature_model is not None, "Glass model not initialized."
        assert self.weather_model is not None, "Weather Model not initialized."

        self.log.debug("monitor_health")

        while self.disabled_or_enabled:
            self.subtasks = [
                asyncio.create_task(coro())
                for coro in (
                    self.weather_model.monitor,
                    self.hvac_model.monitor,
                    self.tma_model.monitor,
                )
            ]

            # Wait for start and then signal
            self.connect_callbacks()
            await asyncio.gather(
                self.weather_model.monitor_start_event.wait(),
                self.hvac_model.monitor_start_event.wait(),
                self.tma_model.monitor_start_event.wait(),
            )
            self.log.debug("Monitors started.")
            self.monitor_start_event.set()

            done, pending = await asyncio.wait(
                self.subtasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.subtasks = []
            self.disconnect_callbacks()
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
