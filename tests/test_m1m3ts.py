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
import typing
import unittest
from types import SimpleNamespace

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
        self.fan_rpm: list[int] | None = None

    async def do_applySetpoints(self, data: salobj.BaseMsgType) -> None:
        self.glycol_setpoint = data.glycolSetpoint
        self.heater_setpoint = data.heatersSetpoint

    async def do_applySetpoint(self, data: salobj.BaseMsgType) -> None:
        self.glycol_setpoint = data.glycolSetpoint
        self.heater_setpoint = data.heatersSetpoint

    async def do_heaterFanDemand(self, data: salobj.BaseMsgType) -> None:
        self.fan_rpm = data.fanRPM


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


class WeatherModelMock:
    def __init__(self, last_twilight_temperature: float) -> None:
        self.last_twilight_temperature = last_twilight_temperature

    async def get_last_twilight_temperature(self) -> float:
        return self.last_twilight_temperature


class DomeModelMock:
    def __init__(self, is_closed: bool) -> None:
        self.is_closed = is_closed


class TestTma(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.last_twilight_temperature = 20
        super().setUp()

    @contextlib.asynccontextmanager
    async def mock_extra_cscs(
        self, ess112_temperature: float | None
    ) -> typing.AsyncGenerator[None, None]:
        self.domain = salobj.Domain()
        self.log = logging.getLogger()
        self.ess112 = salobj.Controller("ESS", 112)

        await self.ess112.start_task

        self.weather_model = WeatherModelMock(self.last_twilight_temperature)
        self.dome_model = DomeModelMock(is_closed=False)

        if ess112_temperature is not None:
            emit_ess112_temperature_task = asyncio.create_task(
                self.emit_ess112_temperature(ess112_temperature)
            )

        try:
            yield
        finally:
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

    async def run_with_parameters(
        self,
        ess112_temperature: float | None,
        signal_sunrise: bool = False,
        **model_args: typing.Any,
    ) -> tuple[float | None, float | None, float | None, list[float] | None]:
        self.diurnal_timer = eas.diurnal_timer.DiurnalTimer()
        self.diurnal_timer.is_running = True

        async with (
            self.mock_extra_cscs(ess112_temperature),
            M1M3TSMock() as mock_m1m3ts,
            MTMountMock() as mock_mtmount,
        ):
            await mock_mtmount.evt_summaryState.set_write(
                summaryState=salobj.State.ENABLED
            )

            self.tma_model = eas.tma_model.TmaModel(
                domain=self.domain,
                log=mock_m1m3ts.log,
                diurnal_timer=self.diurnal_timer,
                dome_model=self.dome_model,
                glass_temperature_model=SimpleNamespace(median_temperature=0.0),
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
                slow_cooling_rate=1,
                fast_cooling_rate=10,
                features_to_disable=model_args["features_to_disable"],
            )
            monitor_task = asyncio.create_task(self.tma_model.monitor())

            await asyncio.sleep(15)

            if signal_sunrise:
                async with self.diurnal_timer.sunrise_condition:
                    self.diurnal_timer.sunrise_condition.notify_all()
                await asyncio.sleep(1)
            else:
                await asyncio.sleep(20)

            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            await mock_mtmount.close()
            await mock_m1m3ts.close()

            return (
                mock_m1m3ts.glycol_setpoint,
                mock_m1m3ts.heater_setpoint,
                mock_mtmount.top_end_setpoint,
                mock_m1m3ts.fan_rpm,
            )

    async def test_m1m3ts_applysetpoints(self) -> None:
        """M1M3TS.applySetpoint should be called at sunrise."""
        ess112_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -0.5

        glycol_setpoint, heater_setpoint, top_end_setpoint, _ = (
            await self.run_with_parameters(
                ess112_temperature,
                glycol_setpoint_delta=glycol_setpoint_delta,
                heater_setpoint_delta=heater_setpoint_delta,
                top_end_setpoint_delta=top_end_setpoint_delta,
                signal_sunrise=True,
                features_to_disable=[],
            )
        )

        assert (
            self.last_twilight_temperature is not None and glycol_setpoint is not None
        )
        self.assertAlmostEqual(
            glycol_setpoint,
            self.last_twilight_temperature + self.tma_model.glycol_setpoint_delta,
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

        glycol_setpoint, heater_setpoint, top_end_setpoint, fan_rpm = (
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
        self.assertIsNone(fan_rpm)

    async def test_disabled_top_end(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        ess112_temperature = 0.5
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1

        glycol_setpoint, heater_setpoint, top_end_setpoint, fan_rpm = (
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
        assert fan_rpm is not None
        expected_fan_rpm = int(
            0.1 * 0.5 * (eas.tma_model.MIN_FAN_RPM + eas.tma_model.MAX_FAN_RPM)
        )
        self.assertAlmostEqual(fan_rpm[0], expected_fan_rpm, 3)

    async def test_disabled_tma(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        ess112_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1.5

        glycol_setpoint, heater_setpoint, top_end_setpoint, _ = (
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
            glycol_setpoint, heater_setpoint, top_end_setpoint, _ = (
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

    async def test_fan_speed(self) -> None:
        """Compare behavior with specifications in OSW-820."""
        self.diurnal_timer = eas.diurnal_timer.DiurnalTimer()
        self.diurnal_timer.is_running = True

        async with (
            self.mock_extra_cscs(10),
            M1M3TSMock() as mock_m1m3ts,
            salobj.Remote(name="MTM1M3TS", domain=self.domain) as mtm1m3ts_remote,
        ):
            tma_model = eas.tma_model.TmaModel(
                domain=self.domain,
                log=mock_m1m3ts.log,
                diurnal_timer=self.diurnal_timer,
                dome_model=self.dome_model,
                glass_temperature_model=SimpleNamespace(median_temperature=0.0),
                weather_model=self.weather_model,
                indoor_ess_index=112,
                ess_timeout=20,
                glycol_setpoint_delta=-2,
                heater_setpoint_delta=-1,
                top_end_setpoint_delta=-1,
                m1m3_setpoint_cadence=10,
                setpoint_deadband_heating=0,
                setpoint_deadband_cooling=0,
                maximum_heating_rate=100,
                slow_cooling_rate=1,
                fast_cooling_rate=10,
                features_to_disable=[],
            )

            fan_minimum = int(0.1 * eas.tma_model.MIN_FAN_RPM)
            fan_maximum = int(0.1 * eas.tma_model.MAX_FAN_RPM)

            # No difference between glass and setpoint (with offset):
            #    * fan at minimum RPM (700)
            #    * glycol delta at minimum (-1)
            await tma_model.set_fan_speed(m1m3ts_remote=mtm1m3ts_remote, setpoint=1.0)
            self.assertEqual(mock_m1m3ts.fan_rpm, [fan_minimum] * 96)
            self.assertEqual(
                tma_model.glycol_setpoint_delta, eas.tma_model.OFFSET_AT_MIN_RPM
            )

            # Difference of +1°C between glass and setpoint (with offset):
            # (warming up the glass)
            #    * fan at maximum RPM (2500)
            #    * glycol delta at minimum (-1)
            await tma_model.set_fan_speed(m1m3ts_remote=mtm1m3ts_remote, setpoint=2.0)
            self.assertEqual(mock_m1m3ts.fan_rpm, [fan_maximum] * 96)
            self.assertEqual(
                tma_model.glycol_setpoint_delta, eas.tma_model.OFFSET_AT_MIN_RPM
            )

            # Difference of -1°C between glass and setpoint (with offset):
            # (cooling the glass)
            #    * fan at maximum RPM (2500)
            #    * glycol delta at maximum (-5)
            await tma_model.set_fan_speed(m1m3ts_remote=mtm1m3ts_remote, setpoint=0.0)
            self.assertEqual(mock_m1m3ts.fan_rpm, [fan_maximum] * 96)
            self.assertEqual(
                tma_model.glycol_setpoint_delta, eas.tma_model.OFFSET_AT_MAX_RPM
            )

    def basic_make_csc(
        self,
        initial_state: salobj.State | int | None,
        config_dir: str,
        simulation_mode: int,
    ) -> salobj.BaseCsc:
        """Make and return a CSC.

        Parameters
        ----------
        initial_state : `lsst.ts.salobj.State` or `int`
            The initial state of the CSC.
        config_dir : `str` or `pathlib.Path` or `None`
            Directory of configuration files, or None for the standard
            configuration directory (obtained from
            `ConfigureCsc._get_default_config_dir`).
        simulation_mode : `int`
            Simulation mode.
        kwargs : `dict`
            Extra keyword arguments, if needed.
        """
        raise NotImplementedError()
