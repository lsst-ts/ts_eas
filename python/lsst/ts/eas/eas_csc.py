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

import astropy.units as u
import lsst_efd_client
import pandas as pd
from astropy.time import Time
from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from . import __version__
from .config_schema import CONFIG_SCHEMA

SAL_TIMEOUT = 5.0  # SAL telemetry/command timeout
M1M3TS_STOP_TIMEOUT = 120.0  # Time to wait for fans to stop and valve to close
STOP_LOOP_TIME = 1.0  # How often to check fan and valves when stopping
SUMMARY_STATE_TIME = 5.0  # Wait time for a summary state change
FAN_SLEEP_TIME = 30.0  # Time to wait after changing the fans
VALVE_SLEEP_TIME = 60.0  # Time to wait after changing the valve
HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state
WINDSPEED_WINDOW = 30 * 60  # Maximum age of windspeed data to consider

THERMAL_LOOP_ERROR = 100
THERMAL_SHUTDOWN_ERROR = 101


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
        self.enabled_event = (
            asyncio.Event()
        )  # An event that is set when the CSC is enabled.
        self.control_loop_lock = asyncio.Lock()

        self.heater_demand: list[int] = [0] * 96
        self.fan_demand: list[int] = [30] * 96
        self.temperature_target_offset = -1.0

        self.old_valve_position: float = float("nan")

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
            # Start the thermal task if not already started
            if self.m1m3_thermal_task.done() and self.simulation_mode == 0:
                self.m1m3_thermal_task = asyncio.create_task(self.run_control())

            # Connect if not already connected
            if not self.connected:
                await self.connect()
        else:

            # Cancel the thermal task
            if not self.m1m3_thermal_task.done():
                self.m1m3_thermal_task.cancel()
                try:
                    await self.m1m3_thermal_task
                except asyncio.CancelledError:
                    pass

            # Disconnect
            await self.disconnect()

        # Set the state of the enabled event
        if self.summary_state == salobj.State.ENABLED:
            self.enabled_event.set()
        else:
            self.enabled_event.clear()

    async def begin_disable(self, id_data: salobj.BaseDdsDataType) -> None:
        """Begin do_disable; called before state changes.

        This method acknowledges the do_disable command and stops the
        M1M3TS control loop.

        Parameters
        ----------
        id_data: `CommandIdData`
            Command ID and data
        """

        await self.cmd_disable.ack_in_progress(id_data, timeout=SAL_TIMEOUT)
        try:
            async with self.control_loop_lock:
                await asyncio.wait_for(
                    self.stop_m1m3_thermal_system(),
                    timeout=M1M3TS_STOP_TIMEOUT,
                )
        except TimeoutError:
            await self.fault(
                code=THERMAL_SHUTDOWN_ERROR,
                report="Failed to stop fans/valve.",
                traceback=traceback.format_exc(),
            )
            raise
        await super().begin_disable(id_data)

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config

    @property
    def connected(self) -> bool:
        # TODO Add code to determine if the CSC is connected or not.
        return True

    async def run_loop(self) -> None:
        """The core loop that regulates the M1M3 temperature."""

        assert not isnan(self.old_valve_position)

        glycol = await self.m1m3ts.tel_glycolLoopTemperature.next(
            flush=True, timeout=SAL_TIMEOUT
        )
        mixing = await self.m1m3ts.tel_mixingValve.next(flush=True, timeout=SAL_TIMEOUT)
        fcu = await self.m1m3ts.tel_thermalData.next(flush=True, timeout=SAL_TIMEOUT)
        current_temp = (
            glycol.insideCellTemperature1
            + glycol.insideCellTemperature2
            + glycol.insideCellTemperature3
        ) / 3
        current_valve_position = mixing.valvePosition

        fcu = await self.m1m3ts.tel_thermalData.next(flush=True, timeout=SAL_TIMEOUT)
        fan_speed = fcu.fanRPM
        fcu_temp = fcu.absoluteTemperature

        air_temp = await self.ess.tel_temperature.next(flush=True, timeout=SAL_TIMEOUT)
        target_temp = air_temp.temperatureItem[0] + self.temperature_target_offset

        self.log.info(
            f"""
            target cell temp (above air temp): {target_temp}
            current cell temp: {current_temp}
            current valve position: {current_valve_position}
            current fan speed: {fan_speed[50]}
            current FCU temp: {fcu_temp[50]}
            """
        )

        # if the FCUs are off, try to turn them on
        if fan_speed[50] > 60000:
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
                heaterPWM=self.heater_demand,
                fanRPM=self.fan_demand,
                timeout=SAL_TIMEOUT,
            )
            await asyncio.sleep(FAN_SLEEP_TIME)
        elif fan_speed[50] < 50:
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
                heaterPWM=self.heater_demand,
                fanRPM=self.fan_demand,
                timeout=SAL_TIMEOUT,
            )
            await asyncio.sleep(FAN_SLEEP_TIME)

        if current_temp - target_temp >= 0.05:
            new_valve_position = min(10.0, self.old_valve_position + 5.0)
            self.log.info(f"temp high, adjusting mixing valve to: {new_valve_position}")
            await self.m1m3ts.cmd_setMixingValve.set_start(
                mixingValveTarget=new_valve_position,
                timeout=SAL_TIMEOUT,
            )
            self.old_valve_position = new_valve_position
            self.log.debug(f"waiting {VALVE_SLEEP_TIME} seconds...")
            await asyncio.sleep(VALVE_SLEEP_TIME)
        elif current_temp - target_temp <= -0.05:
            new_valve_position = max(0.0, self.old_valve_position - 5.0)
            self.log.info(f"temp low, adjusting mixing valve to: {new_valve_position}")
            await self.m1m3ts.cmd_setMixingValve.set_start(
                mixingValveTarget=new_valve_position, timeout=SAL_TIMEOUT
            )
            self.old_valve_position = new_valve_position
            self.log.debug(f"waiting {VALVE_SLEEP_TIME} seconds...")
            await asyncio.sleep(VALVE_SLEEP_TIME)
        else:
            self.log.debug(
                f"""
                doing nothing, valve position: {current_valve_position}
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

            current_valve_position = mixing.valvePosition
            self.old_valve_position = current_valve_position
        except Exception:
            await self.fault(
                code=THERMAL_LOOP_ERROR,
                report="Failed to get mixing valve telemetry from M1M3TS.",
                traceback=traceback.format_exc(),
            )

        while True:
            try:
                await self.enabled_event.wait()
                async with self.control_loop_lock:
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

    async def stop_m1m3_thermal_system(self) -> None:
        """Stops the M1M3 thermal control loop for maintenance or idling.

        Changes the summary state of MTM1M3TS to DISABLED. This
        has the effect of stopping the fans and closing the
        mixing valve. Once this is done, it is safe to enter
        and work inside the mirror cell.
        """

        try:
            await salobj.set_summary_state(
                self.m1m3ts,
                salobj.State.DISABLED,
                timeout=SAL_TIMEOUT,
            )
        except Exception:
            await self.fault(
                code=THERMAL_LOOP_ERROR,
                report="Failed to get mixing valve telemetry from M1M3TS.",
                traceback=traceback.format_exc(),
            )
            raise

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
