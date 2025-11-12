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
import unittest
from typing import TypedDict

import astropy
from lsst.ts import salobj
from lsst.ts.eas import hvac_model
from lsst.ts.xml.enums.HVAC import DeviceId

STD_TIMEOUT = 10
STD_SLEEP = 2


class WeatherModelMock:
    def __init__(
        self,
        *,
        last_twilight_temperature: float | None = None,
        average_windspeed: float | None = None,
        current_temperature: float | None = None,
        current_indoor_temperature: float | None = None,
        nightly_minimum_temperature: float = math.nan,
        nightly_maximum_indoor_dew_point: float | None = None,
    ) -> None:
        self.last_twilight_temperature = last_twilight_temperature
        self.average_windspeed = average_windspeed
        self.current_temperature = current_temperature
        self.current_indoor_temperature = current_indoor_temperature
        self.nightly_minimum_temperature = nightly_minimum_temperature
        self.nightly_maximum_indoor_dew_point = nightly_maximum_indoor_dew_point

    async def get_last_twilight_temperature(self) -> float | None:
        return self.last_twilight_temperature


class DomeModelMock:
    def __init__(self, is_closed: bool | None = None) -> None:
        self.is_closed = is_closed


class DiurnalTimerMock:
    def __init__(self, *, running: bool = True, night: bool = False) -> None:
        self.is_running = running
        self._night = night
        self.noon_condition = asyncio.Condition()
        self.sunrise_condition = asyncio.Condition()

    def is_night(self, time: astropy.time.Time) -> bool:
        return self._night

    async def stop(self) -> None:
        self.is_running = False
        async with self.noon_condition:
            self.noon_condition.notify_all()
        async with self.sunrise_condition:
            self.sunrise_condition.notify_all()


async def signal_noon(timer: DiurnalTimerMock) -> None:
    async def signal_coroutine() -> None:
        await asyncio.sleep(STD_SLEEP)
        async with timer.noon_condition:
            timer.noon_condition.notify_all()
        await asyncio.sleep(STD_SLEEP)
        await timer.stop()

    await asyncio.wait_for(
        asyncio.create_task(signal_coroutine()),
        timeout=STD_TIMEOUT,
    )


async def signal_sunrise(timer: DiurnalTimerMock) -> None:
    async def signal_coroutine() -> None:
        await asyncio.sleep(STD_SLEEP)
        async with timer.sunrise_condition:
            timer.sunrise_condition.notify_all()
        await asyncio.sleep(STD_SLEEP)
        await timer.stop()

    await asyncio.wait_for(
        asyncio.create_task(signal_coroutine()),
        timeout=STD_TIMEOUT,
    )


class HvacMock(salobj.BaseCsc):
    version = "?"

    def __init__(self) -> None:
        self.valid_simulation_modes = (0,)
        super().__init__(
            name="HVAC",
            index=None,
            initial_state=salobj.State.ENABLED,
            allow_missing_callbacks=True,
        )
        self.enable_called: set[int] = set()
        self.disable_called: set[int] = set()
        self.chiller_setpoints: dict[int, float] = dict()  # Calls to configChiller
        self.ahu_setpoints: dict[int, float] = dict()  # Calls to configLowerAhu

    async def do_enableDevice(self, data: salobj.BaseMsgType) -> None:
        self.enable_called.add(data.device_id)

    async def do_disableDevice(self, data: salobj.BaseMsgType) -> None:
        self.disable_called.add(data.device_id)

    async def do_configLowerAhu(self, data: salobj.BaseMsgType) -> None:
        self.ahu_setpoints[data.device_id] = data.workingSetpoint

    async def do_configChiller(self, data: salobj.BaseMsgType) -> None:
        self.chiller_setpoints[data.device_id] = data.activeSetpoint


class TestHvac(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.log = logging.getLogger("hvac")
        self.hvac = HvacMock()
        await self.hvac.start_task

        self.remote = salobj.Remote(name="HVAC", domain=self.hvac.domain)
        await self.remote.start_task

        self.diurnal = DiurnalTimerMock(running=True, night=False)
        self.dome = DomeModelMock(is_closed=True)
        self.weather = WeatherModelMock(
            last_twilight_temperature=8.0,
            average_windspeed=3.0,
            current_temperature=7.5,
            current_indoor_temperature=15.0,
            nightly_minimum_temperature=6.0,
            nightly_maximum_indoor_dew_point=-10.0,
        )

    def make_model(self, **overrides: float | list[str] | None) -> hvac_model.HvacModel:
        params = dict(
            setpoint_lower_limit=6.0,
            wind_threshold=10.0,
            vec04_hold_time=0.0,
            glycol_band_low=-10.0,
            glycol_band_high=-5.0,
            glycol_average_offset=-7.5,
            glycol_dew_point_margin=1.0,
            glycol_setpoints_delta=1.0,
            glycol_absolute_minimum=-10.0,
            features_to_disable=[],
        )
        params.update(overrides)

        return hvac_model.HvacModel(
            log=self.log,
            diurnal_timer=self.diurnal,
            dome_model=self.dome,
            weather_model=self.weather,
            hvac_remote=self.remote,
            **params,
        )

    async def asyncTearDown(self) -> None:
        try:
            await self.remote.close()
            await self.hvac.close()
        finally:
            await super().asyncTearDown()

    # --- GLYCOL CHILLER TESTS --- #

    def test_average_and_delta(self) -> None:
        """Basic test that average and delta match the stated requirement."""
        model = self.make_model()
        s1, s2 = model.compute_glycol_setpoints(self.weather.current_indoor_temperature)

        expected_avg = (
            self.weather.current_indoor_temperature + model.glycol_average_offset
        )  # 15 - 7.5 = 7.5

        # Average of the two setpoints should differ from the ambient
        # by `glycol_average_offset`.
        self.assertAlmostEqual((s1 + s2) / 2.0, expected_avg, places=6)

        # The two setpoints should differ by `glycol_setpoints_delta`
        self.assertAlmostEqual(s1 - s2, model.glycol_setpoints_delta, places=6)

        # Chiller 1 setpoint should be the warmer.
        self.assertGreater(s1, s2)

        # Average is inside the prescribed band
        band_low = self.weather.current_indoor_temperature + model.glycol_band_low
        band_high = self.weather.current_indoor_temperature + model.glycol_band_high
        self.assertTrue(band_low <= expected_avg <= band_high)

    def test_dew_point_guard_raises_average(self) -> None:
        """If nighttime dewpoint is high, raise setpoint average."""
        self.weather.nightly_maximum_indoor_dew_point = 9.0
        model = self.make_model()  # margin default = 1.0

        # Nominal average would be 7.5, but dew point 9 + margin 1 = 10
        #   --> average raised to 10.
        s1, s2 = model.compute_glycol_setpoints(self.weather.current_indoor_temperature)
        self.assertAlmostEqual((s1 + s2) / 2.0, 10.0, places=6)

    def test_average_inside_band_returns_true(self) -> None:
        """Average offset within [band_low, band_high] should be True."""
        model = self.make_model()
        # Choose setpoints around the nominal average 7.5 with delta=1.0
        model.glycol_setpoint1 = 8.0
        model.glycol_setpoint2 = 7.0

        self.assertTrue(
            model.check_glycol_setpoint(self.weather.current_indoor_temperature)
        )

    def test_average_below_band_returns_false(self) -> None:
        """Test glycol average below band."""
        model = self.make_model()
        # Push average to 4.0 (below band_low=5.0)
        model.glycol_setpoint1 = 4.5
        model.glycol_setpoint2 = 3.5

        self.assertFalse(
            model.check_glycol_setpoint(self.weather.current_indoor_temperature)
        )

    def test_average_above_band_returns_false(self) -> None:
        """Test glycol average above band."""
        model = self.make_model()
        # Push average to 11.0 (above band_high=10.0)
        model.glycol_setpoint1 = 11.5
        model.glycol_setpoint2 = 10.5

        self.assertFalse(
            model.check_glycol_setpoint(self.weather.current_indoor_temperature)
        )

    async def test_adjust_glycol_at_noon(self) -> None:
        """Signal noon and verify that glycol setpoints are issued."""
        model = self.make_model()

        signal_task = asyncio.create_task(signal_noon(self.diurnal))
        await model.adjust_glycol_chillers_at_noon()

        await signal_task

        # Verify that the applied setpoints match expectation:
        #  * Average of the two setpoints should be
        #    last night's minimum temperature plus the configured offset
        #  * The two sepoints should be separated by the
        #    configured delta.
        setpoints = (model.glycol_setpoint1, model.glycol_setpoint2)
        setpoint_average = sum(setpoints) / len(setpoints)
        self.assertAlmostEqual(
            model.glycol_setpoint1 - model.glycol_setpoint2,
            model.glycol_setpoints_delta,
        )
        self.assertAlmostEqual(
            setpoint_average,
            self.weather.nightly_minimum_temperature + model.glycol_average_offset,
        )

        # HVAC setpoints should match model setpoints.
        self.assertAlmostEqual(
            self.hvac.chiller_setpoints[DeviceId.chiller01P01],
            model.glycol_setpoint1,
        )
        self.assertAlmostEqual(
            self.hvac.chiller_setpoints[DeviceId.chiller02P01],
            model.glycol_setpoint2,
        )

    async def test_disable_glycol_chiller(self) -> None:
        model = self.make_model(features_to_disable=["glycol_chillers"])

        monitor_task = asyncio.create_task(model.monitor())
        await asyncio.sleep(STD_SLEEP)
        await signal_noon(self.diurnal)

        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass  # expected

        self.assertIsNone(model.glycol_setpoint1)
        self.assertIsNone(model.glycol_setpoint2)

    async def test_missing_nightly_minimum(self) -> None:
        """Signal noon without a nightly minimum temperature."""
        self.weather.nightly_minimum_temperature = math.nan
        model = self.make_model()

        signal_task = asyncio.create_task(signal_noon(self.diurnal))
        await model.adjust_glycol_chillers_at_noon()
        await signal_task

        self.assertEqual(len(self.hvac.chiller_setpoints), 0)
        self.assertIsNone(model.glycol_setpoint1)
        self.assertIsNone(model.glycol_setpoint2)

    async def test_recalculate_glycol_setpoints(self) -> None:
        """Glycol setpoints recalculate if ambient pushes them out of band."""
        hvac_model.HVAC_SLEEP_TIME = 0.0
        model = self.make_model()

        model.glycol_setpoint1 = -10
        model.glycol_setpoint2 = -11

        signal_task = asyncio.create_task(signal_noon(self.diurnal))
        await model.monitor_glycol_chillers()
        await signal_task

        # Setpoints should be recalculated based on
        # `current_indoor_temperature` in the weather model.
        average = sum(self.hvac.chiller_setpoints.values()) / len(
            self.hvac.chiller_setpoints
        )
        difference = (
            self.hvac.chiller_setpoints[DeviceId.chiller01P01]
            - self.hvac.chiller_setpoints[DeviceId.chiller02P01]
        )
        self.assertAlmostEqual(difference, model.glycol_setpoints_delta)
        self.assertAlmostEqual(
            average,
            self.weather.current_indoor_temperature + model.glycol_average_offset,
        )

    async def test_dont_recalculate_glycol_setpoints(self) -> None:
        """Glycol setpoints don't recalculate if they remain in band."""
        hvac_model.HVAC_SLEEP_TIME = 0.0
        model = self.make_model()

        initial_setpoint1 = 9
        initial_setpoint2 = 8
        model.glycol_setpoint1 = initial_setpoint1
        model.glycol_setpoint2 = initial_setpoint2

        signal_task = asyncio.create_task(signal_noon(self.diurnal))
        await model.monitor_glycol_chillers()
        await signal_task

        # Setpoints should NOT be recalculated
        self.assertAlmostEqual(
            self.hvac.chiller_setpoints[DeviceId.chiller01P01],
            initial_setpoint1,
        )
        self.assertAlmostEqual(
            self.hvac.chiller_setpoints[DeviceId.chiller02P01],
            initial_setpoint2,
        )

    # --- LOWER AHU TESTS --- #

    # control_ahus_and_vec04
    async def test_dome_open_and_calm_wind(self) -> None:
        # Dome open and calm winds
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
        self.dome.is_closed = False
        self.weather.average_windspeed = 3.0  # below threshold 10.0

        model = self.make_model()
        task = asyncio.create_task(model.control_ahus_and_vec04())

        await asyncio.sleep(STD_SLEEP)  # Allow enough time for 5 SAL commands.

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        # VEC-04 was enabled
        self.assertIn(DeviceId.loadingBayFan04P04, self.hvac.enable_called)
        self.assertNotIn(DeviceId.loadingBayFan04P04, self.hvac.disable_called)

        for ahu in (
            DeviceId.lowerAHU01P05,
            DeviceId.lowerAHU02P05,
            DeviceId.lowerAHU03P05,
            DeviceId.lowerAHU04P05,
        ):
            # All 4 AHUs should have been disabled when dome opened
            self.assertIn(ahu, self.hvac.disable_called)

            # And none should have been enabled
            self.assertNotIn(ahu, self.hvac.enable_called)

    async def test_vec04_turns_off(self) -> None:
        # Start open and calm so VEC-04 enables first
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
        self.dome.is_closed = False
        self.weather.average_windspeed = 3.0

        model = self.make_model()
        task = asyncio.create_task(model.control_ahus_and_vec04())

        # Let it enable VEC-04
        await asyncio.sleep(STD_SLEEP)
        self.assertIn(DeviceId.loadingBayFan04P04, self.hvac.enable_called)
        self.assertNotIn(DeviceId.loadingBayFan04P04, self.hvac.disable_called)

        # Wind rises above threshold --> should disable VEC-04
        self.weather.average_windspeed = 12.0
        await asyncio.sleep(STD_SLEEP)
        self.assertIn(DeviceId.loadingBayFan04P04, self.hvac.disable_called)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

    async def test_ahus_on_shutter_close(self) -> None:
        """If shutter closes, AHUs should enable and VEC-04 should disable."""
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
        self.dome.is_closed = False
        self.weather.average_windspeed = 3.0

        model = self.make_model()
        task = asyncio.create_task(model.control_ahus_and_vec04())

        # First iteration: VEC-04 ON
        await asyncio.sleep(STD_SLEEP)

        for ahu in (
            DeviceId.lowerAHU01P05,
            DeviceId.lowerAHU02P05,
            DeviceId.lowerAHU03P05,
            DeviceId.lowerAHU04P05,
        ):
            # AHUs not enabled because dome is open.
            self.assertNotIn(ahu, self.hvac.enable_called)

        # Close the shutter --> AHUs enabled, VEC-04 OFF
        self.dome.is_closed = True
        await asyncio.sleep(STD_SLEEP)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        # VEC-04 forced OFF on close
        self.assertIn(DeviceId.loadingBayFan04P04, self.hvac.disable_called)

        for ahu in (
            DeviceId.lowerAHU01P05,
            DeviceId.lowerAHU02P05,
            DeviceId.lowerAHU03P05,
            DeviceId.lowerAHU04P05,
        ):
            # AHUs enabled on close
            self.assertIn(ahu, self.hvac.enable_called)

    async def test_vec04_disabled(self) -> None:
        """VEC04 commands are not sent if 'vec04' in `features_to_disable`."""
        self.dome.is_closed = False
        self.weather.average_windspeed = 3.0

        model = self.make_model(features_to_disable=["vec04"])
        task = asyncio.create_task(model.control_ahus_and_vec04())

        await asyncio.sleep(STD_SLEEP)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        self.assertNotIn(DeviceId.loadingBayFan04P04, self.hvac.enable_called)
        self.assertNotIn(DeviceId.loadingBayFan04P04, self.hvac.disable_called)

    async def test_ahu_disabled(self) -> None:
        """AHU commands are not sent if 'ahu' in `features_to_disable`."""
        self.dome.is_closed = False

        model = self.make_model(features_to_disable=["ahu"])
        task = asyncio.create_task(model.control_ahus_and_vec04())

        await asyncio.sleep(STD_SLEEP)
        self.dome.is_closed = True  # transition to open
        await asyncio.sleep(STD_SLEEP)
        self.dome.is_closed = True  # transition back to closed

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        for ahu in (
            DeviceId.lowerAHU01P05,
            DeviceId.lowerAHU02P05,
            DeviceId.lowerAHU03P05,
            DeviceId.lowerAHU04P05,
        ):
            # No commands sent to lower AHUs
            self.assertNotIn(ahu, self.hvac.enable_called)
            self.assertNotIn(ahu, self.hvac.disable_called)

    async def test_wait_for_sunrise(self) -> None:
        class SubtestCase(TypedDict):
            name: str
            last_twilight: float
            features_to_disable: list[str]
            expect_setpoints: dict[int, float]

        cases: list[SubtestCase] = [
            {
                "name": "apply last-twilight setpoint",
                "last_twilight": 10.0,
                "features_to_disable": [],
                "expect_setpoints": {
                    DeviceId.lowerAHU01P05: 10.0,
                    DeviceId.lowerAHU02P05: 10.0,
                    DeviceId.lowerAHU03P05: 10.0,
                    DeviceId.lowerAHU04P05: 10.0,
                },
            },
            {
                "name": "enforce lower limit",
                "last_twilight": 4.0,
                "features_to_disable": [],
                "expect_setpoints": {
                    DeviceId.lowerAHU01P05: 6.0,
                    DeviceId.lowerAHU02P05: 6.0,
                    DeviceId.lowerAHU03P05: 6.0,
                    DeviceId.lowerAHU04P05: 6.0,
                },
            },
            {
                "name": "room_setpoint disabled",
                "last_twilight": 4.0,
                "features_to_disable": ["room_setpoint"],
                "expect_setpoints": {},  # type: ignore[typeddict-item]
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
                self.diurnal.is_running = True
                self.weather.last_twilight_temperature = case["last_twilight"]
                model = self.make_model(
                    features_to_disable=case.get("features_to_disable"),
                )

                task = asyncio.create_task(model.wait_for_sunrise())
                await signal_sunrise(self.diurnal)
                await asyncio.sleep(STD_SLEEP)  # let it apply
                await self.diurnal.stop()

                await asyncio.wait_for(task, timeout=STD_SLEEP)

                self.assertDictEqual(self.hvac.ahu_setpoints, case["expect_setpoints"])

                # Reset between subtests
                self.hvac.ahu_setpoints.clear()

    async def test_apply_setpoint_at_night(self) -> None:
        """Test `apply_setpoint_at_night`."""
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP

        class SubtestCase(TypedDict):
            name: str
            night: bool
            closed: bool
            temp: float
            expect_setpoints: dict[str, float]

        scenarios: list[SubtestCase] = [
            {
                "name": "night closed apply setpoints",
                "night": True,
                "closed": True,
                "temp": 6.0,
                "expect_setpoints": {
                    ahu: 6.0
                    for ahu in (
                        DeviceId.lowerAHU01P05,
                        DeviceId.lowerAHU02P05,
                        DeviceId.lowerAHU03P05,
                        DeviceId.lowerAHU04P05,
                    )
                },
            },
            {
                "name": "night open --> no setpoints",
                "night": True,
                "closed": False,
                "temp": 9.0,
                "expect_setpoints": {},
            },
            {
                "name": "day closed --> no setpoints",
                "night": False,
                "closed": True,
                "temp": 12.0,
                "expect_setpoints": {},
            },
            {
                "name": "night closed NaN temp --> no setpoints",
                "night": True,
                "closed": True,
                "temp": math.nan,
                "expect_setpoints": {},
            },
        ]

        for case in scenarios:
            with self.subTest(case=case["name"]):
                self.diurnal._night = case["night"]
                self.dome.is_closed = case["closed"]
                self.weather.current_temperature = case["temp"]

                model = self.make_model()
                task = asyncio.create_task(model.apply_setpoint_at_night())

                await asyncio.sleep(STD_SLEEP)
                await self.diurnal.stop()
                await asyncio.wait_for(task, timeout=STD_SLEEP)

                self.assertEqual(self.hvac.ahu_setpoints, case["expect_setpoints"])
                # reset for next scenario
                self.hvac.ahu_setpoints.clear()

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
        """
        raise NotImplementedError()
