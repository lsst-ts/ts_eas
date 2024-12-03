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
import traceback
import typing
from math import isnan
from types import SimpleNamespace

from lsst.ts import salobj, utils

from . import __version__
from .config_schema import CONFIG_SCHEMA

SAL_TIMEOUT = 5.0  # SAL telemetry/command timeout
REMOTE_STARTUP_TIME = 5.0  # Time for remotes to get set up
SUMMARY_STATE_TIME = 5.0  # Wait time for a summary state change
FAN_SLEEP_TIME = 30.0  # Time to wait after changing the fans
VALVE_SLEEP_TIME = 60.0  # Time to wait after changing the valve

THERMAL_LOOP_ERROR = 100


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
        self.eas = None
        self.m1m3ts = salobj.Remote(self.domain, "MTM1M3TS")
        self.ess = salobj.Remote(self.domain, "ESS", index=112)

        # Variables for the m1m3ts loop
        self.m1m3_thermal_task = utils.make_done_future()

        self.heaterdemand: list[int] = [0] * 96
        self.fandemand: list[int] = [30] * 96

        self.oldvalveposition: float = float("nan")

        self.log.info("__init__")

    async def connect(self) -> None:
        """Connect the EAS CSC or start the mock client, if in
        simulation mode.
        """
        self.log.info("Connecting")
        self.log.info(self.config)
        self.log.info(f"self.simulation_mode = {self.simulation_mode}")
        if self.config is None:
            raise RuntimeError("Not yet configured")
        if self.connected:
            raise RuntimeError("Already connected")
        if self.simulation_mode == 1:
            # TODO Add code for simulation case, see DM-26440
            pass
        else:
            # TODO Add code for non-simulation case
            pass

        if self.eas:
            self.eas.connect()

    async def disconnect(self) -> None:
        """Disconnect the EAS CSC, if connected."""
        self.log.info("Disconnecting")

        if self.eas:
            self.eas.disconnect()

    async def handle_summary_state(self) -> None:
        """Override of the handle_summary_state function to connect or
        disconnect to the EAS CSC (or the mock client) when needed.
        """
        self.log.info(f"handle_summary_state {salobj.State(self.summary_state).name}")
        if self.disabled_or_enabled:
            if self.m1m3_thermal_task.done():
                self.m1m3_thermal_task = asyncio.create_task(self.run_control())

            if not self.connected:
                await self.connect()
        else:
            if not self.m1m3_thermal_task.done():
                self.m1m3_thermal_task.cancel()
                try:
                    await self.m1m3_thermal_task
                except asyncio.CancelledError:
                    pass

            await self.disconnect()

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config

    @property
    def connected(self) -> bool:
        # TODO Add code to determine if the CSC is connected or not.
        return True

    async def run_loop(self) -> None:
        """The core loop that regulates the M1M3 temperature."""

        assert not isnan(self.oldvalveposition)

        glycol = await self.m1m3ts.tel_glycolLoopTemperature.next(
            flush=True, timeout=SAL_TIMEOUT
        )
        mixing = await self.m1m3ts.tel_mixingValve.next(flush=True, timeout=SAL_TIMEOUT)
        fcu = await self.m1m3ts.tel_thermalData.next(flush=True, timeout=SAL_TIMEOUT)
        currenttemp = (
            glycol.insideCellTemperature1
            + glycol.insideCellTemperature2
            + glycol.insideCellTemperature3
        ) / 3
        currentvalveposition = mixing.valvePosition

        fcu = await self.m1m3ts.tel_thermalData.next(flush=True, timeout=SAL_TIMEOUT)
        fanspeed = fcu.fanRPM
        fcutemp = fcu.absoluteTemperature

        airtemp = await self.ess.tel_temperature.next(flush=True, timeout=SAL_TIMEOUT)
        targettemp = airtemp.temperatureItem[0]

        self.log.info(
            f"""
            target cell temp (above air temp): {targettemp}
            current cell temp: {currenttemp}
            current valve position: {currentvalveposition}
            current fan speed: {fanspeed[50]}
            current FCU temp: {fcutemp[50]}
            """
        )

        # if the FCUs are off, try to turn them on
        if fanspeed[50] > 60000:
            self.log.info(
                f"fans off, turning them on and waiting {FAN_SLEEP_TIME} seconds..."
            )
            await salobj.set_summary_state(
                self.m1m3ts,
                salobj.State.STANDBY,
                timeout=SAL_TIMEOUT,
            )
            await asyncio.sleep(SUMMARY_STATE_TIME)
            await salobj.set_summary_state(
                self.m1m3ts,
                salobj.State.ENABLED,
                timeout=SAL_TIMEOUT,
            )
            await asyncio.sleep(SUMMARY_STATE_TIME)
            await self.m1m3ts.cmd_setEngineeringMode.set_start(
                enableEngineeringMode=True,
                timeout=SAL_TIMEOUT,
            )
            await self.m1m3ts.cmd_heaterFanDemand.set_start(
                heaterPWM=self.heaterdemand,
                fanRPM=self.fandemand,
                timeout=SAL_TIMEOUT,
            )
            await asyncio.sleep(FAN_SLEEP_TIME)
        elif fanspeed[50] < 50:
            self.log.info(
                "fans rpms too low, turning them back up and waiting {FAN_SLEEP_TIME} seconds..."
            )
            await salobj.set_summary_state(self.m1m3ts, salobj.State.STANDBY)
            await asyncio.sleep(SUMMARY_STATE_TIME)
            await salobj.set_summary_state(self.m1m3ts, salobj.State.ENABLED)
            await asyncio.sleep(SUMMARY_STATE_TIME)
            await self.m1m3ts.cmd_setEngineeringMode.set_start(
                enableEngineeringMode=True,
                timeout=SAL_TIMEOUT,
            )
            await self.m1m3ts.cmd_heaterFanDemand.set_start(
                heaterPWM=self.heaterdemand,
                fanRPM=self.fandemand,
                timeout=SAL_TIMEOUT,
            )
            await asyncio.sleep(FAN_SLEEP_TIME)

        if currenttemp - targettemp >= 0.05:
            newvalveposition = min(10.0, self.oldvalveposition + 5.0)
            self.log.info(f"temp high, adjusting mixing valve to: {newvalveposition}")
            await self.m1m3ts.cmd_setMixingValve.set_start(
                mixingValveTarget=newvalveposition,
                timeout=SAL_TIMEOUT,
            )
            self.oldvalveposition = newvalveposition
            self.log.debug(f"waiting {VALVE_SLEEP_TIME} seconds...")
            await asyncio.sleep(VALVE_SLEEP_TIME)
        elif currenttemp - targettemp <= -0.05:
            newvalveposition = max(0.0, self.oldvalveposition - 5.0)
            self.log.info(f"temp low, adjusting mixing valve to: {newvalveposition}")
            await self.m1m3ts.cmd_setMixingValve.set_start(
                mixingValveTarget=newvalveposition, timeout=5
            )
            self.oldvalveposition = newvalveposition
            self.log.debug(f"waiting {VALVE_SLEEP_TIME} seconds...")
            await asyncio.sleep(VALVE_SLEEP_TIME)
        else:
            self.log.debug(
                f"""
                doing nothing, valve position: {currentvalveposition}
                waiting {VALVE_SLEEP_TIME} seconds for update...
                """
            )
            await asyncio.sleep(VALVE_SLEEP_TIME)

    async def run_control(self) -> None:
        """Runs the control loop for the fans and the heaters."""

        if self.simulation_mode != 0:
            return

        try:
            mixing = await self.m1m3ts.tel_mixingValve.next(
                flush=True, timeout=SAL_TIMEOUT
            )
        except Exception:
            await self.fault(
                code=THERMAL_LOOP_ERROR,
                report="Failed to get mixing valve telemetry from M1M3TS.",
                traceback=traceback.format_exc(),
            )

        currentvalveposition = mixing.valvePosition
        self.oldvalveposition = currentvalveposition

        while True:
            try:
                await self.run_loop()

            except asyncio.CancelledError:
                self.log.info("M1M3 thermal control loop cancelled.")
                raise
            except Exception:
                self.log.exception("Error running the thermal loop.")
                await self.fault(
                    code=THERMAL_LOOP_ERROR,
                    report="Error running thermal loop.",
                    traceback=traceback.format_exc(),
                )
                break

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_ocs"
