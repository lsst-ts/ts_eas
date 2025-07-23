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

import asyncio
import contextlib
import logging
import os
import typing
import unittest

from lsst.ts import eas, salobj

STD_TIMEOUT = 60

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)


class M1M3TSMock(salobj.BaseCsc):
    version = "?"

    def __init__(self) -> None:
        self.valid_simulation_modes = (0,)
        super().__init__(
            name="MTM1M3TS",
            index=None,
            initial_state=salobj.State.ENABLED,
            allow_missing_callbacks=True,
            extra_commands={"applySetpoint", "applySetpoints"},
        )
        self.glycol_setpoint: float | None = None
        self.heater_setpoint: float | None = None

    async def do_applySetpoints(self, data: salobj.BaseMsgType) -> None:
        self.glycol_setpoint = data.glycolSetpoint
        self.heater_setpoint = data.heatersSetpoint

    async def do_applySetpoint(self, data: salobj.BaseMsgType) -> None:
        self.glycol_setpoint = data.glycolSetpoint
        self.heater_setpoint = data.heatersSetpoint


class MTMountMock(salobj.BaseCsc):
    version = "?"

    def __init__(self) -> None:
        self.valid_simulation_modes = (0,)
        super().__init__(
            name="MTMount",
            index=None,
            initial_state=salobj.State.ENABLED,
            allow_missing_callbacks=True,
        )
        self.top_end_setpoint: float | None = None

    async def do_setThermal(self, data: salobj.BaseMsgType) -> None:
        self.top_end_setpoint = data.topEndChillerSetpoint


class TestTma(unittest.IsolatedAsyncioTestCase):
    def run(self, result: typing.Any = None) -> None:
        salobj.testutils.set_test_topic_subname(randomize=False)
        os.environ["LSST_SITE"] = "test"
        self.last_twilight_temperature: float | None = None
        super().run(result)  # type: ignore

    @contextlib.asynccontextmanager
    async def mock_extra_cscs(
        self, ess112_temperature: float | None
    ) -> typing.AsyncGenerator[None, None]:
        self.domain = salobj.Domain()
        self.log = logging.getLogger()
        self.ess112 = salobj.Controller("ESS", 112)
        self.dome = salobj.Controller("MTDome")
        await self.ess112.start_task
        await self.dome.start_task

        self.dome_model = eas.dome_model.DomeModel(
            domain=self.domain,
        )
        dome_monitor_task = asyncio.create_task(self.dome_model.monitor())
        self.weather_model = eas.weather_model.WeatherModel(
            domain=self.domain,
            log=self.log,
            diurnal_timer=self.diurnal_timer,
            dome_model=self.dome_model,
        )
        self.weather_model.last_twilight_temperature = self.last_twilight_temperature

        await asyncio.wait_for(
            self.dome_model.monitor_start_event.wait(),
            timeout=STD_TIMEOUT,
        )

        if ess112_temperature is not None:
            emit_ess112_temperature_task = asyncio.create_task(
                self.emit_ess112_temperature(ess112_temperature)
            )

        await self.dome.tel_apertureShutter.set_write(
            positionActual=(100.0, 100.0),
        )

        try:
            yield
        finally:
            dome_monitor_task.cancel()
            try:
                await dome_monitor_task
            except asyncio.CancelledError:
                pass

            if ess112_temperature is not None:
                emit_ess112_temperature_task.cancel()
                try:
                    await emit_ess112_temperature_task
                except asyncio.CancelledError:
                    pass
            await self.ess112.close()
            await self.domain.close()

    async def emit_ess112_temperature(self, ess112_temperature: float) -> None:
        while True:
            await asyncio.sleep(3)
            await self.ess112.tel_temperature.set_write(
                sensorName="",
                timestamp=0,
                numChannels=1,
                temperatureItem=[ess112_temperature] * 16,
                location="",
            )

    @contextlib.asynccontextmanager
    async def mock_m1m3ts(self) -> typing.AsyncGenerator[None, None]:
        self.m1m3ts = M1M3TSMock()
        await self.m1m3ts.start_task

        try:
            yield
        finally:
            try:
                await self.m1m3ts.close()
            except asyncio.CancelledError:
                pass

    async def run_with_parameters(
        self,
        ess112_temperature: float | None,
        signal_noon: bool = False,
        **model_args: typing.Any,
    ) -> tuple[float | None, float | None, float | None]:
        self.diurnal_timer = eas.diurnal_timer.DiurnalTimer()
        self.diurnal_timer.is_running = True

        async with (
            self.mock_extra_cscs(ess112_temperature),
            M1M3TSMock() as mock_m1m3ts,
            MTMountMock() as mock_mtmount,
        ):
            self.tma_model = eas.tma_model.TmaModel(
                domain=self.domain,
                log=mock_m1m3ts.log,
                diurnal_timer=self.diurnal_timer,
                dome_model=self.dome_model,
                weather_model=self.weather_model,
                indoor_ess_index=112,
                ess_timeout=20,
                glycol_setpoint_delta=model_args["glycol_setpoint_delta"],
                heater_setpoint_delta=model_args["heater_setpoint_delta"],
                top_end_setpoint_delta=model_args["top_end_setpoint_delta"],
                m1m3_setpoint_cadence=10,
                setpoint_deadband_heating=0,
                setpoint_deadband_cooling=0,
                maximum_heating_rate=100,
                features_to_disable=model_args["features_to_disable"],
            )
            monitor_task = asyncio.create_task(self.tma_model.monitor())

            await asyncio.sleep(30)

            if signal_noon:
                async with self.diurnal_timer.noon_condition:
                    self.diurnal_timer.noon_condition.notify_all()
                await asyncio.sleep(1)
            else:
                await asyncio.sleep(40)

            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return (
                mock_m1m3ts.glycol_setpoint,
                mock_m1m3ts.heater_setpoint,
                mock_mtmount.top_end_setpoint,
            )

    async def test_m1m3ts_applysetpoints(self) -> None:
        """M1M3TS.applySetpoint should be called at noon."""
        self.last_twilight_temperature = 20
        ess112_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -0.5

        glycol_setpoint, heater_setpoint, top_end_setpoint = (
            await self.run_with_parameters(
                ess112_temperature,
                glycol_setpoint_delta=glycol_setpoint_delta,
                heater_setpoint_delta=heater_setpoint_delta,
                top_end_setpoint_delta=top_end_setpoint_delta,
                signal_noon=True,
                features_to_disable=[],
            )
        )

        assert (
            self.last_twilight_temperature is not None and glycol_setpoint is not None
        )
        self.assertAlmostEqual(
            glycol_setpoint,
            self.last_twilight_temperature + glycol_setpoint_delta,
            places=4,
        )

        assert (
            self.last_twilight_temperature is not None and heater_setpoint is not None
        )
        self.assertAlmostEqual(
            heater_setpoint,
            self.last_twilight_temperature + heater_setpoint_delta,
            places=4,
        )

    async def test_disabled_m1m3ts(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        ess112_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1.5

        glycol_setpoint, heater_setpoint, top_end_setpoint = (
            await self.run_with_parameters(
                ess112_temperature,
                glycol_setpoint_delta=glycol_setpoint_delta,
                heater_setpoint_delta=heater_setpoint_delta,
                top_end_setpoint_delta=top_end_setpoint_delta,
                features_to_disable=["m1m3ts"],
            )
        )

        self.assertTrue(glycol_setpoint is None)
        self.assertTrue(heater_setpoint is None)
        self.assertTrue(top_end_setpoint is not None)

    async def test_disabled_top_end(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        ess112_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1.5

        glycol_setpoint, heater_setpoint, top_end_setpoint = (
            await self.run_with_parameters(
                ess112_temperature,
                glycol_setpoint_delta=glycol_setpoint_delta,
                heater_setpoint_delta=heater_setpoint_delta,
                top_end_setpoint_delta=top_end_setpoint_delta,
                features_to_disable=["top_end"],
            )
        )

        self.assertTrue(glycol_setpoint is not None)
        self.assertTrue(heater_setpoint is not None)
        self.assertTrue(top_end_setpoint is None)

    async def test_disabled_tma(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        ess112_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1.5

        glycol_setpoint, heater_setpoint, top_end_setpoint = (
            await self.run_with_parameters(
                ess112_temperature,
                glycol_setpoint_delta=glycol_setpoint_delta,
                heater_setpoint_delta=heater_setpoint_delta,
                top_end_setpoint_delta=top_end_setpoint_delta,
                features_to_disable=["m1m3ts", "top_end"],
            )
        )

        self.assertTrue(glycol_setpoint is None)
        self.assertTrue(heater_setpoint is None)
        self.assertTrue(top_end_setpoint is None)

    async def test_no_ess112(self) -> None:
        """tma_model.monitor() should raise if ESS112 does not send data."""
        ess112_temperature = None
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1

        with self.assertRaises(RuntimeError):
            glycol_setpoint, heater_setpoint, top_end_setpoint = (
                await self.run_with_parameters(
                    ess112_temperature,
                    glycol_setpoint_delta=glycol_setpoint_delta,
                    heater_setpoint_delta=heater_setpoint_delta,
                    top_end_setpoint_delta=top_end_setpoint_delta,
                    features_to_disable=[""],
                )
            )

            self.assertTrue(glycol_setpoint is None)
            self.assertTrue(heater_setpoint is None)
            self.assertTrue(top_end_setpoint is None)

            # Should not matter how long this sleep is because it should
            # be interrupted by the RuntimeError.
            await asyncio.sleep(60)
