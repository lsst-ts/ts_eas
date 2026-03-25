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
import logging
import math
import typing
import unittest
from collections import deque
from types import SimpleNamespace

from lsst.ts import eas, salobj

STD_TIMEOUT = 60
STD_SLEEP = 0.5

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG)


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
        self.current_temperature: float | None = 10
        self.current_indoor_temperature: float | None = 10

    async def get_last_twilight_temperature(self) -> float:
        return self.last_twilight_temperature


class DomeModelMock:
    def __init__(self, is_closed: bool) -> None:
        self.is_closed = is_closed
        self.on_open: deque[tuple[asyncio.Event, float]] = deque()


class TestTma(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.last_twilight_temperature = 20
        super().setUp()

    async def run_with_parameters(
        self,
        indoor_temperature: float | None = 10,
        signal_sunrise: bool = False,
        night: bool = False,
        dome_closed: bool = False,
        run_monitor: bool = True,
        **model_args: typing.Any,
    ) -> tuple[float | None, float | None, float | None, list[float] | None]:
        self.diurnal_timer = eas.diurnal_timer.DiurnalTimer()
        self.diurnal_timer.is_running = True
        self.diurnal_timer.is_night = lambda _: night

        self.weather_model = WeatherModelMock(self.last_twilight_temperature)
        self.weather_model.current_temperature = indoor_temperature
        self.weather_model.current_indoor_temperature = indoor_temperature
        self.dome_model = DomeModelMock(is_closed=dome_closed)
        cadence = 10 if run_monitor else STD_SLEEP

        async with (
            M1M3TSMock() as mock_m1m3ts,
            MTMountMock() as mock_mtmount,
            salobj.Remote(name="MTM1M3TS", domain=mock_m1m3ts.domain) as m1m3ts_remote,
            salobj.Remote(name="MTMount", domain=mock_m1m3ts.domain) as mtmount_remote,
        ):
            await mock_mtmount.evt_summaryState.set_write(summaryState=salobj.State.ENABLED)

            m1m3ts_delay_mode = model_args.get("m1m3ts_delay_mode", {"mode": "time_delay", "delay": 0.0})
            self.tma_model = eas.tma_model.TmaModel(
                log=mock_m1m3ts.log,
                diurnal_timer=self.diurnal_timer,
                dome_model=self.dome_model,
                glass_temperature_model=SimpleNamespace(median_temperature=0.0),
                weather_model=self.weather_model,
                m1m3ts_remote=m1m3ts_remote,
                mtmount_remote=mtmount_remote,
                glycol_setpoint_delta=model_args["glycol_setpoint_delta"],
                heater_setpoint_delta=model_args["heater_setpoint_delta"],
                top_end_setpoint_delta=model_args["top_end_setpoint_delta"],
                m1m3_extra_delta_closedatnite=model_args.get("m1m3_extra_delta_closedatnite", 0.0),
                top_end_setpoint_delta_closedatnite=model_args.get(
                    "top_end_setpoint_delta_closedatnite", model_args["top_end_setpoint_delta"]
                ),
                m1m3_setpoint_cadence=cadence,
                setpoint_deadband_heating=0,
                setpoint_deadband_cooling=0,
                maximum_heating_rate=100,
                slow_cooling_rate=1,
                fast_cooling_rate=10,
                fan_speed={
                    "fan_speed_min": 700.0,
                    "fan_speed_max": 2000.0,
                    "fan_glycol_heater_offset_min": -1.0,
                    "fan_glycol_heater_offset_max": -4.0,
                    "fan_throttle_turn_on_temp_diff": 0.0,
                    "fan_throttle_max_temp_diff": 1.0,
                },
                m1m3ts_delay_mode=m1m3ts_delay_mode,
                features_to_disable=model_args["features_to_disable"],
            )
            if run_monitor:
                model_task = asyncio.create_task(self.tma_model.monitor())
            else:
                model_task = asyncio.create_task(self.tma_model.apply_setpoint_at_night())

            await asyncio.sleep(2 * cadence)

            if signal_sunrise:
                self.diurnal_timer.is_night = lambda _: False
                self.dome_model.is_closed = True
                async with self.diurnal_timer.sunrise_condition:
                    self.diurnal_timer.sunrise_condition.notify_all()

            await asyncio.sleep(2 * cadence)

            model_task.cancel()
            try:
                await asyncio.wait_for(model_task, timeout=STD_TIMEOUT)
            except asyncio.CancelledError:
                pass
            except TimeoutError:
                task_name = "TmaModel.monitor" if run_monitor else "TmaModel.apply_setpoint_at_night"
                self.fail(f"{task_name} did not stop before timeout")

            return (
                mock_m1m3ts.glycol_setpoint,
                mock_m1m3ts.heater_setpoint,
                mock_mtmount.top_end_setpoint,
                mock_m1m3ts.fan_rpm,
            )

    async def test_apply_setpoint_at_night(self) -> None:
        """Nighttime closed-dome control should mirror the HVAC behavior."""

        class SubtestCase(typing.TypedDict):
            name: str
            night: bool
            dome_closed: bool
            indoor_temperature: float
            expect_m1m3: bool
            expect_top_end: bool

        scenarios: list[SubtestCase] = [
            {
                "name": "night closed apply setpoints",
                "night": True,
                "dome_closed": True,
                "indoor_temperature": 6.0,
                "expect_m1m3": True,
                "expect_top_end": True,
            },
            {
                "name": "night open no setpoints",
                "night": True,
                "dome_closed": False,
                "indoor_temperature": 9.0,
                "expect_m1m3": False,
                "expect_top_end": False,
            },
            {
                "name": "day closed no setpoints",
                "night": False,
                "dome_closed": True,
                "indoor_temperature": 12.0,
                "expect_m1m3": False,
                "expect_top_end": False,
            },
            {
                "name": "night closed NaN temp no setpoints",
                "night": True,
                "dome_closed": True,
                "indoor_temperature": math.nan,
                "expect_m1m3": False,
                "expect_top_end": False,
            },
        ]

        glycol_setpoint_delta = -2.0
        heater_setpoint_delta = -1.0
        top_end_setpoint_delta = -0.5
        m1m3_extra_delta_closedatnite = 0.5
        top_end_setpoint_delta_closedatnite = -1.5

        for case in scenarios:
            with self.subTest(case=case["name"]):
                glycol_setpoint, heater_setpoint, top_end_setpoint, _ = await self.run_with_parameters(
                    indoor_temperature=case["indoor_temperature"],
                    night=case["night"],
                    dome_closed=case["dome_closed"],
                    run_monitor=False,
                    glycol_setpoint_delta=glycol_setpoint_delta,
                    heater_setpoint_delta=heater_setpoint_delta,
                    top_end_setpoint_delta=top_end_setpoint_delta,
                    m1m3_extra_delta_closedatnite=m1m3_extra_delta_closedatnite,
                    top_end_setpoint_delta_closedatnite=top_end_setpoint_delta_closedatnite,
                    features_to_disable=[],
                )

                if case["expect_m1m3"]:
                    self.assertEqual(
                        glycol_setpoint,
                        case["indoor_temperature"] + glycol_setpoint_delta + m1m3_extra_delta_closedatnite,
                    )
                    self.assertEqual(
                        heater_setpoint,
                        case["indoor_temperature"] + heater_setpoint_delta + m1m3_extra_delta_closedatnite,
                    )
                else:
                    self.assertIsNone(glycol_setpoint)
                    self.assertIsNone(heater_setpoint)

                if case["expect_top_end"]:
                    self.assertEqual(
                        top_end_setpoint,
                        case["indoor_temperature"] + top_end_setpoint_delta_closedatnite,
                    )
                else:
                    self.assertIsNone(top_end_setpoint)

    async def test_m1m3ts_applysetpoints(self) -> None:
        """M1M3TS.applySetpoint should be called at sunrise."""
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -0.5

        glycol_setpoint, heater_setpoint, top_end_setpoint, _ = await self.run_with_parameters(
            indoor_temperature=10,
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            top_end_setpoint_delta=top_end_setpoint_delta,
            signal_sunrise=True,
            features_to_disable=[],
        )

        assert self.last_twilight_temperature is not None and glycol_setpoint is not None
        self.assertAlmostEqual(
            glycol_setpoint,
            self.last_twilight_temperature + self.tma_model.glycol_setpoint_delta,
            places=4,
        )

        assert self.last_twilight_temperature is not None and heater_setpoint is not None
        self.assertAlmostEqual(
            heater_setpoint,
            self.last_twilight_temperature + heater_setpoint_delta,
            places=4,
        )

    async def test_m1m3ts_applysetpoints_ignores_fan_adjustment(self) -> None:
        """Sunrise applySetpoints should ignore the fan-driven adjustment."""
        glycol_setpoint_delta = -3.0
        heater_setpoint_delta = -1.0
        top_end_setpoint_delta = -0.5

        glycol_setpoint, _, _, _ = await self.run_with_parameters(
            indoor_temperature=10,
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            top_end_setpoint_delta=top_end_setpoint_delta,
            signal_sunrise=True,
            features_to_disable=[],
        )

        assert self.last_twilight_temperature is not None and glycol_setpoint is not None
        self.assertAlmostEqual(
            glycol_setpoint,
            self.last_twilight_temperature + glycol_setpoint_delta,
            places=4,
        )
        self.assertNotAlmostEqual(
            glycol_setpoint,
            self.last_twilight_temperature
            + heater_setpoint_delta
            + self.tma_model.fan_glycol_heater_offset_min,
            places=4,
        )

    async def test_disabled_m1m3ts(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1.5

        glycol_setpoint, heater_setpoint, top_end_setpoint, fan_rpm = await self.run_with_parameters(
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            top_end_setpoint_delta=top_end_setpoint_delta,
            features_to_disable=["m1m3ts"],
        )

        self.assertTrue(glycol_setpoint is None)
        self.assertTrue(heater_setpoint is None)
        self.assertTrue(top_end_setpoint is not None)
        assert fan_rpm is not None
        expected_fan_rpm = int(0.1 * self.tma_model.fan_speed_max)
        self.assertAlmostEqual(fan_rpm[0], expected_fan_rpm, 3)

    async def test_disabled_top_end(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        indoor_temperature = 0.5
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1

        glycol_setpoint, heater_setpoint, top_end_setpoint, fan_rpm = await self.run_with_parameters(
            indoor_temperature=indoor_temperature,
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            top_end_setpoint_delta=top_end_setpoint_delta,
            features_to_disable=["top_end"],
        )

        self.assertTrue(glycol_setpoint is not None)
        self.assertTrue(heater_setpoint is not None)
        self.assertTrue(top_end_setpoint is None)
        assert fan_rpm is not None
        expected_fan_rpm = int(0.1 * 0.5 * (self.tma_model.fan_speed_min + self.tma_model.fan_speed_max))
        self.assertAlmostEqual(fan_rpm[0], expected_fan_rpm, 3)

    async def test_disabled_tma(self) -> None:
        """The applySetpoint should not be called when m1m3ts is disabled."""
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1
        top_end_setpoint_delta = -1.5

        glycol_setpoint, heater_setpoint, top_end_setpoint, _ = await self.run_with_parameters(
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            top_end_setpoint_delta=top_end_setpoint_delta,
            features_to_disable=["m1m3ts", "top_end"],
        )

        self.assertTrue(glycol_setpoint is None)
        self.assertTrue(heater_setpoint is None)
        self.assertTrue(top_end_setpoint is None)

    async def test_fan_speed(self) -> None:
        """Compare behavior with specifications in OSW-820."""
        diurnal_timer = eas.diurnal_timer.DiurnalTimer()
        diurnal_timer.is_running = True
        heater_setpoint_delta = -1

        weather_model = WeatherModelMock(self.last_twilight_temperature)
        weather_model.current_indoor_temperature = 10
        dome_model = DomeModelMock(is_closed=False)

        async with (
            M1M3TSMock() as mock_m1m3ts,
            salobj.Remote(name="MTM1M3TS", domain=mock_m1m3ts.domain) as m1m3ts_remote,
            salobj.Remote(name="MTMount", domain=mock_m1m3ts.domain) as mtmount_remote,
        ):
            tma_model = eas.tma_model.TmaModel(
                log=mock_m1m3ts.log,
                diurnal_timer=diurnal_timer,
                dome_model=dome_model,
                glass_temperature_model=SimpleNamespace(median_temperature=0.0),
                weather_model=weather_model,
                m1m3ts_remote=m1m3ts_remote,
                mtmount_remote=mtmount_remote,
                glycol_setpoint_delta=-2,
                heater_setpoint_delta=heater_setpoint_delta,
                top_end_setpoint_delta=-1,
                m1m3_extra_delta_closedatnite=0.0,
                top_end_setpoint_delta_closedatnite=-1,
                m1m3_setpoint_cadence=10,
                setpoint_deadband_heating=0,
                setpoint_deadband_cooling=0,
                maximum_heating_rate=100,
                slow_cooling_rate=1,
                fast_cooling_rate=10,
                fan_speed={
                    "fan_speed_min": 700.0,
                    "fan_speed_max": 2000.0,
                    "fan_glycol_heater_offset_min": -1.0,
                    "fan_glycol_heater_offset_max": -4.0,
                    "fan_throttle_turn_on_temp_diff": 0.0,
                    "fan_throttle_max_temp_diff": 1.0,
                },
                m1m3ts_delay_mode={"mode": "time_delay", "delay": 0.0},
                features_to_disable=[],
            )

            fan_minimum = int(0.1 * tma_model.fan_speed_min)
            fan_maximum = int(0.1 * tma_model.fan_speed_max)

            # No difference between glass and setpoint (with offset):
            #    * fan at minimum RPM (700)
            #    * glycol delta at minimum (-1)
            await tma_model.set_fan_speed(setpoint=1.0)
            await asyncio.sleep(STD_SLEEP)
            self.assertEqual(mock_m1m3ts.fan_rpm, [fan_minimum] * 96)
            self.assertEqual(
                tma_model.glycol_setpoint_delta_adjustment,
                tma_model.fan_glycol_heater_offset_min
                + heater_setpoint_delta
                - tma_model.glycol_setpoint_delta,
            )

            # Difference of +1°C between glass and setpoint (with offset):
            # (warming up the glass)
            #    * fan at maximum RPM (2500)
            #    * glycol delta at minimum (-1)
            await tma_model.set_fan_speed(setpoint=2.0)
            await asyncio.sleep(STD_SLEEP)
            self.assertEqual(mock_m1m3ts.fan_rpm, [fan_maximum] * 96)
            self.assertEqual(
                tma_model.glycol_setpoint_delta_adjustment,
                tma_model.fan_glycol_heater_offset_min
                + heater_setpoint_delta
                - tma_model.glycol_setpoint_delta,
            )

            # Difference of -1°C between glass and setpoint (with offset):
            # (cooling the glass)
            #    * fan at maximum RPM (2500)
            #    * glycol delta at maximum (-5)
            await tma_model.set_fan_speed(setpoint=0.0)
            self.assertEqual(mock_m1m3ts.fan_rpm, [fan_maximum] * 96)
            self.assertEqual(
                tma_model.glycol_setpoint_delta_adjustment,
                tma_model.fan_glycol_heater_offset_max
                + heater_setpoint_delta
                - tma_model.glycol_setpoint_delta,
            )

    async def test_delay_policy_time_delay(self) -> None:
        """Delay policy should gate tracking until the time delay elapses."""
        cadence = 0.05
        delay_seconds = 10 * cadence

        async with (
            M1M3TSMock() as mock_m1m3ts,
            MTMountMock() as mock_mtmount,
            salobj.Remote(name="MTM1M3TS", domain=mock_m1m3ts.domain) as m1m3ts_remote,
            salobj.Remote(name="MTMount", domain=mock_m1m3ts.domain) as mtmount_remote,
        ):
            await mock_mtmount.evt_summaryState.set_write(summaryState=salobj.State.ENABLED)

            diurnal_timer = eas.diurnal_timer.DiurnalTimer()
            diurnal_timer.is_running = True

            weather_model = WeatherModelMock(self.last_twilight_temperature)
            weather_model.current_indoor_temperature = 10.0
            dome_model = DomeModelMock(is_closed=True)

            tma_model = eas.tma_model.TmaModel(
                log=mock_m1m3ts.log,
                diurnal_timer=diurnal_timer,
                dome_model=dome_model,
                glass_temperature_model=SimpleNamespace(median_temperature=0.0),
                weather_model=weather_model,
                m1m3ts_remote=m1m3ts_remote,
                mtmount_remote=mtmount_remote,
                glycol_setpoint_delta=-2,
                heater_setpoint_delta=0.0,
                top_end_setpoint_delta=-1,
                m1m3_extra_delta_closedatnite=0.0,
                top_end_setpoint_delta_closedatnite=-1,
                m1m3_setpoint_cadence=cadence,
                setpoint_deadband_heating=0,
                setpoint_deadband_cooling=0,
                maximum_heating_rate=100,
                slow_cooling_rate=1,
                fast_cooling_rate=10,
                fan_speed={
                    "fan_speed_min": 700.0,
                    "fan_speed_max": 2000.0,
                    "fan_glycol_heater_offset_min": -1.0,
                    "fan_glycol_heater_offset_max": -4.0,
                    "fan_throttle_turn_on_temp_diff": 0.0,
                    "fan_throttle_max_temp_diff": 1.0,
                },
                m1m3ts_delay_mode={"mode": "time_delay", "delay": delay_seconds},
                features_to_disable=[],
            )

            task = asyncio.create_task(tma_model.follow_ess_indoor())
            await asyncio.sleep(0)  # Give the task a chance to start.

            # Delay controller has registered an open event.
            self.assertEqual(len(dome_model.on_open), 1)
            event, _ = dome_model.on_open.popleft()

            # Fire the event, but dome is still closed.
            # (This occurs if the dome is re-closed before the delay time.)
            event.set()
            await asyncio.sleep(STD_SLEEP)
            self.assertIsNone(mock_m1m3ts.heater_setpoint)
            self.assertIsNone(mock_m1m3ts.glycol_setpoint)

            # TmaModel should have queued up a new event.
            self.assertEqual(len(dome_model.on_open), 1)
            event, _ = dome_model.on_open.popleft()

            # Fire the event, but this time the dome is open.
            dome_model.is_closed = False
            event.set()
            await asyncio.sleep(cadence)
            self.assertIsNone(mock_m1m3ts.heater_setpoint)
            self.assertIsNone(mock_m1m3ts.glycol_setpoint)

            # After the delay elapses, the controller should allow tracking.
            await asyncio.sleep(2 * (cadence + delay_seconds))

            self.assertIsNotNone(mock_m1m3ts.heater_setpoint)
            self.assertIsNotNone(mock_m1m3ts.glycol_setpoint)

            await tma_model.close()
            task.cancel()
            done, pending = await asyncio.wait({task}, timeout=STD_TIMEOUT)
            if pending:
                self.fail("follow_ess_indoor did not stop before timeout")

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
