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
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NotRequired, TypedDict

import astropy
import jsonschema
import yaml
from astropy.time import Time, TimeDelta

from lsst.ts import salobj
from lsst.ts.eas import hvac_model
from lsst.ts.eas.weatherforecast_model import DELTA_TIME, WeatherForecastModel
from lsst.ts.xml.enums.HVAC import DeviceId

STD_TIMEOUT = 10
STD_SLEEP = 2
CONFIG_PATH = Path(__file__).parent / "config"


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
        self.twilight_condition = asyncio.Condition()
        self.twilight_time: Time | None = None

    def is_night(self, time: astropy.time.Time) -> bool:
        return self._night

    def get_twilight_time(self, *, after: Time) -> Time:
        return after + TimeDelta(45 * 60, format="sec")

    async def stop(self) -> None:
        self.is_running = False
        async with self.noon_condition:
            self.noon_condition.notify_all()
        async with self.sunrise_condition:
            self.sunrise_condition.notify_all()
        async with self.twilight_condition:
            self.twilight_condition.notify_all()


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


async def signal_twilight(timer: DiurnalTimerMock) -> None:
    async def signal_coroutine() -> None:
        await asyncio.sleep(STD_SLEEP)
        async with timer.twilight_condition:
            timer.twilight_condition.notify_all()
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
        self.fan_frequencies: dict[int, float] = dict()  # Calls to configFan

    async def do_enableDevice(self, data: salobj.BaseMsgType) -> None:
        self.enable_called.add(data.device_id)

    async def do_disableDevice(self, data: salobj.BaseMsgType) -> None:
        self.disable_called.add(data.device_id)

    async def do_configLowerAhu(self, data: salobj.BaseMsgType) -> None:
        self.ahu_setpoints[data.device_id] = data.workingSetpoint

    async def do_configChiller(self, data: salobj.BaseMsgType) -> None:
        self.chiller_setpoints[data.device_id] = data.activeSetpoint

    async def do_configFan(self, data: salobj.BaseMsgType) -> None:
        self.fan_frequencies[data.device_id] = data.frequency


class TestHvac(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    def get_config(self, filename: str) -> types.SimpleNamespace:
        """Get a config dict from tests/data.

        This should always be a good config,
        because validation is done by the ESS CSC,
        not the data client.

        Parameters
        ----------
        filename : `str` or `pathlib.Path`
            Name of config file, including ".yaml" suffix.

        Returns
        -------
        config : types.SimpleNamespace
            The config dict.
        """
        with open(CONFIG_PATH / filename, "r") as f:
            config_dict = yaml.safe_load(f.read())
        return types.SimpleNamespace(**config_dict)

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
        self.weatherforecast = WeatherForecastModel(log=self.log)

    def make_model(
        self, **overrides: float | list[int] | list[float] | list[str] | None
    ) -> hvac_model.HvacModel:
        params = dict(
            ahu_setpoint_delta=0.0,
            ahu_setpoint_delta_closed_at_night=0.0,
            ahu_control=[1, 2, 3, 4],
            ahu_off_catchup_deltas=[0.0, -1.0, -2.0, -3.0],
            ahu_off_catchup_rate=1.0,
            ahu_off_catchup_poll_interval=900.0,
            ahu_off_catchup_threshold=1.0,
            setpoint_lower_limit=6.0,
            wind_threshold=10.0,
            vec04_hold_time=0.0,
            vec04_fan_frequency=55.0,
            glycol_band_low=-10.0,
            glycol_band_high=-5.0,
            glycol_average_offset=-7.5,
            glycol_dew_point_margin=1.0,
            glycol_setpoints_delta=1.0,
            glycol_absolute_minimum=-10.0,
            glycol_absolute_maximum=10.0,
            features_to_disable=[],
        )
        params.update(overrides)

        return hvac_model.HvacModel(
            log=self.log,
            diurnal_timer=self.diurnal,
            dome_model=self.dome,
            weather_model=self.weather,
            weatherforecast_model=self.weatherforecast,
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

        expected_avg = self.weather.current_indoor_temperature + model.glycol_average_offset  # 15 - 7.5 = 7.5

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
        self.weather.nightly_maximum_indoor_dew_point = 7.625
        model = self.make_model()  # margin default = 1.0

        # Nominal average would be 7.5, but dew point 7.625 + margin 1 = 8.625
        #   --> average raised to 8.625.
        s1, s2 = model.compute_glycol_setpoints(self.weather.current_indoor_temperature)
        assert s1 is not None
        assert s2 is not None
        setpoints_avg = (s1 + s2) / 2.0
        self.assertAlmostEqual(setpoints_avg, 8.625, places=6)

    def test_absolute_minimum_enforced(self) -> None:
        """Setpoints should not drop below the configured absolute minimum."""
        model = self.make_model(glycol_absolute_minimum=-10.0)
        self.weather.nightly_maximum_indoor_dew_point = -20.0

        s1, s2 = model.compute_glycol_setpoints(ambient_temperature=-5.0)

        self.assertAlmostEqual(s1, -9.0, places=6)
        self.assertAlmostEqual(s2, -10.0, places=6)

    def test_absolute_maximum_enforced(self) -> None:
        """Setpoints should not exceed the configured absolute maximum."""
        model = self.make_model(glycol_absolute_maximum=10.0)

        s1, s2 = model.compute_glycol_setpoints(ambient_temperature=30.0)

        self.assertAlmostEqual(s1, 10.0, places=6)
        self.assertAlmostEqual(s2, 9.0, places=6)

    def test_average_inside_band_returns_true(self) -> None:
        """Average offset within [band_low, band_high] should be True."""
        model = self.make_model()
        # Choose setpoints around the nominal average 7.5 with delta=1.0
        model.glycol_setpoint1 = 8.0
        model.glycol_setpoint2 = 7.0

        self.assertTrue(model.check_glycol_setpoint(self.weather.current_indoor_temperature))

    def test_average_below_band_returns_false(self) -> None:
        """Test glycol average below band."""
        model = self.make_model()
        # Push average to 4.0 (below band_low=5.0)
        model.glycol_setpoint1 = 4.5
        model.glycol_setpoint2 = 3.5

        self.assertFalse(model.check_glycol_setpoint(self.weather.current_indoor_temperature))

    def test_average_above_band_returns_false(self) -> None:
        """Test glycol average above band."""
        model = self.make_model()
        # Push average to 11.0 (above band_high=10.0)
        model.glycol_setpoint1 = 11.5
        model.glycol_setpoint2 = 10.5

        self.assertFalse(model.check_glycol_setpoint(self.weather.current_indoor_temperature))

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
            self.hvac.chiller_setpoints[DeviceId.coldGlycolChiller01],
            model.glycol_setpoint1,
        )
        self.assertAlmostEqual(
            self.hvac.chiller_setpoints[DeviceId.coldGlycolChiller02],
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
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
        model = self.make_model()

        model.glycol_setpoint1 = -10
        model.glycol_setpoint2 = -11

        signal_task = asyncio.create_task(signal_noon(self.diurnal))
        await model.monitor_glycol_chillers()
        await signal_task

        # Setpoints should be recalculated based on
        # `current_indoor_temperature` in the weather model.
        average = sum(self.hvac.chiller_setpoints.values()) / len(self.hvac.chiller_setpoints)
        difference = (
            self.hvac.chiller_setpoints[DeviceId.coldGlycolChiller01]
            - self.hvac.chiller_setpoints[DeviceId.coldGlycolChiller02]
        )
        self.assertAlmostEqual(difference, model.glycol_setpoints_delta)
        self.assertAlmostEqual(
            average,
            self.weather.current_indoor_temperature + model.glycol_average_offset,
        )

    async def test_dont_recalculate_glycol_setpoints(self) -> None:
        """Glycol setpoints don't recalculate if they remain in band."""
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
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
            self.hvac.chiller_setpoints[DeviceId.coldGlycolChiller01],
            initial_setpoint1,
        )
        self.assertAlmostEqual(
            self.hvac.chiller_setpoints[DeviceId.coldGlycolChiller02],
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

        await asyncio.sleep(5 * STD_SLEEP)  # Allow enough time for 5 SAL commands.

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        # VEC-04 was enabled and its frequency was configured
        self.assertIn(DeviceId.airExtractionFan04Dome, self.hvac.enable_called)
        self.assertNotIn(DeviceId.airExtractionFan04Dome, self.hvac.disable_called)
        self.assertEqual(
            self.hvac.fan_frequencies.get(DeviceId.airExtractionFan04Dome),
            55.0,
        )

        for ahu in (
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
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
        self.assertIn(DeviceId.airExtractionFan04Dome, self.hvac.enable_called)
        self.assertNotIn(DeviceId.airExtractionFan04Dome, self.hvac.disable_called)
        self.assertEqual(
            self.hvac.fan_frequencies.get(DeviceId.airExtractionFan04Dome),
            55.0,
        )

        # Wind rises above threshold --> should disable VEC-04
        self.weather.average_windspeed = 12.0
        await asyncio.sleep(STD_SLEEP)
        self.assertIn(DeviceId.airExtractionFan04Dome, self.hvac.disable_called)

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
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
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
        self.assertIn(DeviceId.airExtractionFan04Dome, self.hvac.disable_called)

        for ahu in (
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
        ):
            # AHUs enabled on close
            self.assertIn(ahu, self.hvac.enable_called)

    async def test_dome_transition_clears_hold_retains_setpoint(self) -> None:
        """A dome open/close transition should clear the catch-up hold."""
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
        self.dome.is_closed = False
        self.weather.average_windspeed = 3.0

        model = self.make_model()
        task = asyncio.create_task(model.control_ahus_and_vec04())

        # Let the loop settle into the dome-open state.
        await asyncio.sleep(STD_SLEEP)

        # Simulate a stale, latched daytime catch-up state.
        model.cached_ahu_setpoint = 10.0
        model.catchup_delta = -3.0

        # Close the dome: the transition should reset the catch-up state.
        self.dome.is_closed = True
        await asyncio.sleep(STD_SLEEP)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        # The catch-up hold is cleared, but the base setpoint is retained so it
        # can be re-commanded to the AHUs on the re-close.
        self.assertEqual(model.cached_ahu_setpoint, 10.0)
        self.assertEqual(model.catchup_delta, 0.0)

        # And the AHUs are enabled on close, and the base setpoint is
        # re-commanded to them immediately (not deferred to the next forecast).
        for ahu in (
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
        ):
            self.assertIn(ahu, self.hvac.enable_called)
            self.assertEqual(self.hvac.ahu_setpoints.get(ahu), 10.0)

    async def test_ahu_control_limits_shutter_commands(self) -> None:
        """Only configured AHUs should be enabled and disabled."""
        hvac_model.HVAC_SLEEP_TIME = STD_SLEEP
        self.dome.is_closed = False
        self.weather.average_windspeed = 3.0

        model = self.make_model(ahu_control=[2, 4])
        task = asyncio.create_task(model.control_ahus_and_vec04())

        await asyncio.sleep(STD_SLEEP)
        self.dome.is_closed = True
        await asyncio.sleep(STD_SLEEP)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected

        for ahu in (DeviceId.airHandlingUnit02Dome, DeviceId.airHandlingUnit04Dome):
            self.assertIn(ahu, self.hvac.disable_called)
            self.assertIn(ahu, self.hvac.enable_called)

        for ahu in (DeviceId.airHandlingUnit01Dome, DeviceId.airHandlingUnit03Dome):
            self.assertNotIn(ahu, self.hvac.disable_called)
            self.assertNotIn(ahu, self.hvac.enable_called)

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

        self.assertNotIn(DeviceId.airExtractionFan04Dome, self.hvac.enable_called)
        self.assertNotIn(DeviceId.airExtractionFan04Dome, self.hvac.disable_called)

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
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
        ):
            # No commands sent to lower AHUs
            self.assertNotIn(ahu, self.hvac.enable_called)
            self.assertNotIn(ahu, self.hvac.disable_called)

    async def test_wait_for_sunrise(self) -> None:
        class SubtestCase(TypedDict):
            name: str
            last_twilight: float
            ahu_setpoint_delta: NotRequired[float]  # non-required
            features_to_disable: list[str]
            expect_setpoints: dict[int, float]

        cases: list[SubtestCase] = [
            {
                "name": "apply last-twilight setpoint",
                "last_twilight": 10.0,
                "features_to_disable": [],
                "expect_setpoints": {
                    DeviceId.airHandlingUnit01Dome: 10.0,
                    DeviceId.airHandlingUnit02Dome: 10.0,
                    DeviceId.airHandlingUnit03Dome: 10.0,
                    DeviceId.airHandlingUnit04Dome: 10.0,
                },
            },
            {
                "name": "enforce lower limit",
                "last_twilight": 4.0,
                "features_to_disable": [],
                "expect_setpoints": {
                    DeviceId.airHandlingUnit01Dome: 6.0,
                    DeviceId.airHandlingUnit02Dome: 6.0,
                    DeviceId.airHandlingUnit03Dome: 6.0,
                    DeviceId.airHandlingUnit04Dome: 6.0,
                },
            },
            {
                "name": "ahu setpoint delta",
                "last_twilight": 9.0,
                "features_to_disable": [],
                "ahu_setpoint_delta": -1.0,
                "expect_setpoints": {
                    DeviceId.airHandlingUnit01Dome: 8.0,
                    DeviceId.airHandlingUnit02Dome: 8.0,
                    DeviceId.airHandlingUnit03Dome: 8.0,
                    DeviceId.airHandlingUnit04Dome: 8.0,
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
                    ahu_setpoint_delta=case.get("ahu_setpoint_delta", 0.0),
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
            ahu_setpoint_delta: NotRequired[float]
            ahu_setpoint_delta_closed_at_night: NotRequired[float]
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
                        DeviceId.airHandlingUnit01Dome,
                        DeviceId.airHandlingUnit02Dome,
                        DeviceId.airHandlingUnit03Dome,
                        DeviceId.airHandlingUnit04Dome,
                    )
                },
            },
            {
                "name": "apply setpoints with delta",
                "night": True,
                "closed": True,
                "temp": 9.0,
                "ahu_setpoint_delta": 0.0,
                "ahu_setpoint_delta_closed_at_night": -1.5,
                "expect_setpoints": {
                    ahu: 7.5
                    for ahu in (
                        DeviceId.airHandlingUnit01Dome,
                        DeviceId.airHandlingUnit02Dome,
                        DeviceId.airHandlingUnit03Dome,
                        DeviceId.airHandlingUnit04Dome,
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
                self.diurnal.is_running = True
                self.diurnal._night = case["night"]
                self.dome.is_closed = case["closed"]
                self.weather.current_temperature = case["temp"]

                model = self.make_model(
                    ahu_setpoint_delta=case.get("ahu_setpoint_delta", 0.0),
                    ahu_setpoint_delta_closed_at_night=case.get("ahu_setpoint_delta_closed_at_night", 0.0),
                )
                task = asyncio.create_task(model.apply_setpoint_at_night())

                await asyncio.sleep(STD_SLEEP)
                await self.diurnal.stop()
                await asyncio.wait_for(task, timeout=STD_SLEEP)

                self.assertEqual(self.hvac.ahu_setpoints, case["expect_setpoints"])
                # reset for next scenario
                self.hvac.ahu_setpoints.clear()

    def test_config_schema_ahu_control(self) -> None:
        validator = salobj.DefaultingValidator(hvac_model.HvacModel.get_config_schema())
        scenarios = [
            ("hvac_ahu_control_default.yaml", [1, 2, 3, 4]),
            ("hvac_ahu_control_custom.yaml", [1, 3]),
        ]

        for filename, expected_ahu_control in scenarios:
            with self.subTest(config_path=filename):
                validated = validator.validate(vars(self.get_config(filename)))
                self.assertEqual(validated["ahu_control"], expected_ahu_control)

    def test_closed_at_night_setpoint_cadence(self) -> None:
        """Schema and constructor handling of closed_at_night cadence."""
        validator = salobj.DefaultingValidator(hvac_model.HvacModel.get_config_schema())
        base_config = vars(self.get_config("hvac_ahu_control_default.yaml"))

        # Missing -> default null in the validated dict.
        validated = validator.validate(dict(base_config))
        self.assertIsNone(validated["closed_at_night_setpoint_cadence"])

        # A positive number is accepted.
        validated = validator.validate({**base_config, "closed_at_night_setpoint_cadence": 30.0})
        self.assertEqual(validated["closed_at_night_setpoint_cadence"], 30.0)

        # exclusiveMinimum: 0 rejects zero and negative values.
        for bad_value in (0, -1.0):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(jsonschema.exceptions.ValidationError):
                    validator.validate({**base_config, "closed_at_night_setpoint_cadence": bad_value})

        # Constructor: omitted/None falls back to HVAC_SLEEP_TIME;
        # explicit value is used as-is.
        default_model = self.make_model()
        self.assertEqual(default_model.closed_at_night_setpoint_cadence, hvac_model.HVAC_SLEEP_TIME)

        override_model = self.make_model(closed_at_night_setpoint_cadence=42.0)
        self.assertEqual(override_model.closed_at_night_setpoint_cadence, 42.0)

    async def test_forecast_ahu_applies_setpoint(self) -> None:
        """Forecast-based AHU setpoints should be applied."""
        self.diurnal.is_running = True
        model = self.make_model(ahu_setpoint_delta=-1.0, setpoint_lower_limit=6.0)

        model.handle_twilight_forecast(7.0)
        await asyncio.sleep(STD_SLEEP)

        expected = 6.0
        for ahu in (
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
        ):
            setpoint = self.hvac.ahu_setpoints.get(ahu)
            assert setpoint is not None
            self.assertAlmostEqual(setpoint, expected, places=6)

        self.hvac.ahu_setpoints.clear()

    async def test_forecast_ahu_respects_ahu_control_subset(self) -> None:
        """Forecast AHU setpoints should only target configured AHUs."""
        self.diurnal.is_running = True
        model = self.make_model(
            ahu_control=[2, 4],
            ahu_setpoint_delta=0.0,
            setpoint_lower_limit=-100.0,
        )

        model.handle_twilight_forecast(7.0)
        await asyncio.sleep(STD_SLEEP)

        self.assertDictEqual(
            self.hvac.ahu_setpoints,
            {
                DeviceId.airHandlingUnit02Dome: 7.0,
                DeviceId.airHandlingUnit04Dome: 7.0,
            },
        )

        self.hvac.ahu_setpoints.clear()

    async def test_forecast_ahu_disabled_flags(self) -> None:
        """Forecast-based AHU setpoints should honor disable flags."""
        for feature in ("forecast_ahu", "forecast"):
            with self.subTest(feature=feature):
                self.diurnal.is_running = True
                model = self.make_model(features_to_disable=[feature])

                self.hvac.ahu_setpoints.clear()
                model.handle_twilight_forecast(10.0)
                await asyncio.sleep(STD_SLEEP)

                self.assertDictEqual(self.hvac.ahu_setpoints, {})

        self.hvac.ahu_setpoints.clear()

    async def test_monitor_twilight_forecast_applies_setpoint(self) -> None:
        """Forecast callback should apply AHU setpoints during the window."""
        self.diurnal.is_running = True
        model = self.make_model(ahu_setpoint_delta=0.0, setpoint_lower_limit=-100.0)

        task = asyncio.create_task(model.monitor_twilight_forecast())

        # Wait for task to start waiting for noon.
        await asyncio.sleep(STD_SLEEP)

        # Signal that it's noon.
        async with self.diurnal.noon_condition:
            timestamp = Time.now()
            self.diurnal.noon_condition.notify_all()
        await asyncio.sleep(STD_SLEEP)

        target_time = self.diurnal.get_twilight_time(after=timestamp)  # Time of twilight
        timestamp = timestamp.unix  # Convert to unix epoch

        def prediction(time: float) -> float:
            return 10.0 + (time - (timestamp + DELTA_TIME)) / DELTA_TIME

        temperatures = [prediction(timestamp + (idx + 1) * DELTA_TIME) for idx in range(12)]
        telemetry = SimpleNamespace(
            temperature=temperatures,
            private_sndStamp=timestamp,
        )
        await self.weatherforecast.hourly_trend_callback(telemetry)
        await asyncio.sleep(STD_SLEEP)

        expected = prediction(target_time.unix)
        for ahu in (
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
        ):
            setpoint = self.hvac.ahu_setpoints.get(ahu)
            assert setpoint is not None

            # There's a small difference between the setpoint and
            # expected because of a small time difference between
            # Time.now call in the test and the call when the
            # callback is scheduled in WeatherForecastModel.
            #
            # We could patch Time.now but it's easier just
            # to make our assertAlmostEqual a bit more permissive.
            self.assertAlmostEqual(setpoint, expected, places=4)

        await signal_twilight(self.diurnal)
        await asyncio.wait_for(task, timeout=STD_TIMEOUT)

        self.hvac.ahu_setpoints.clear()

    async def test_disabled_ahus_change_delta(self) -> None:
        """Each number of AHUs off selects the matching catch-up delta."""
        deltas = [0.0, -1.0, -2.0, -3.0]

        # Index i is the expected delta when i AHUs are off (0 AHUs off -> 0).
        expected_by_n_off = [0.0] + deltas
        for n_off, expected in enumerate(expected_by_n_off):
            model = self.make_model(ahu_off_catchup_deltas=deltas)
            # Turn the first n_off AHUs off; the rest report on.
            for ahu in range(1, 5):
                working_state = ahu > n_off
                await model.ahu_working_state_callback(ahu, types.SimpleNamespace(workingState=working_state))
            self.assertEqual(model.catchup_delta, expected)

    async def test_catchup_delta_latches_until_all_on(self) -> None:
        """The catch-up delta holds until every AHU is back on, and the
        setpoint is re-applied whenever the delta changes."""
        model = self.make_model(ahu_off_catchup_deltas=[0.0, -1.0, -2.0, -3.0])

        # A base setpoint must be known for the callback to re-apply setpoints
        # (normally set by sunrise/forecast). Record the setpoints applied
        # instead of commanding the AHUs, so the catch-up behavior is visible.
        base_setpoint = 10.0
        model.cached_ahu_setpoint = base_setpoint
        applied_setpoints: list[float] = []

        async def record(setpoint: float, respect_lower_limit: bool = True) -> None:
            applied_setpoints.append(setpoint)

        model.apply_ahu_setpoints = record  # type: ignore[method-assign]

        async def set_ahu(ahu: int, on: bool) -> None:
            await model.ahu_working_state_callback(ahu, types.SimpleNamespace(workingState=on))

        # All four AHUs start on: no offset, nothing applied.
        for ahu in range(1, 5):
            await set_ahu(ahu, True)
        self.assertEqual(model.catchup_delta, 0.0)
        self.assertEqual(applied_setpoints, [])

        # Three AHUs go off: hold the three-off delta and lower the setpoint as
        # the delta deepens (one-off is a zero delta, so no setpoint yet).
        await set_ahu(1, False)
        await set_ahu(2, False)
        await set_ahu(3, False)
        self.assertEqual(model.catchup_delta, -2.0)
        self.assertEqual(applied_setpoints, [base_setpoint - 1.0, base_setpoint - 2.0])

        # Two come back (one still off): the delta is held, not relaxed, so no
        # new setpoint is applied.
        await set_ahu(1, True)
        await set_ahu(2, True)
        self.assertEqual(model.catchup_delta, -2.0)
        self.assertEqual(applied_setpoints, [base_setpoint - 1.0, base_setpoint - 2.0])

        # All four off deepens the held delta even after partial recovery, and
        # lowers the setpoint further.
        await set_ahu(1, False)
        await set_ahu(2, False)
        await set_ahu(4, False)
        self.assertEqual(model.catchup_delta, -3.0)
        self.assertEqual(
            applied_setpoints,
            [base_setpoint - 1.0, base_setpoint - 2.0, base_setpoint - 3.0],
        )

        # Everything comes back on: the hold releases and the base setpoint is
        # restored.
        for ahu in range(1, 5):
            await set_ahu(ahu, True)
        self.assertEqual(model.catchup_delta, 0.0)
        self.assertEqual(
            applied_setpoints,
            [base_setpoint - 1.0, base_setpoint - 2.0, base_setpoint - 3.0, base_setpoint],
        )

        # Recovery also spawns the ambient-overshoot catch-up loop; cancel it
        # so it does not linger past the test.
        model.cancel_ahu_off_catchup()

    async def test_ahu_off_catchup_lowers_setpoint_on_overshoot(self) -> None:
        for rate, expected in ((1.0, 7.0), (0.5, 8.5)):
            with self.subTest(rate=rate):
                model = self.make_model(ahu_off_catchup_rate=rate, ahu_off_catchup_threshold=1.0)
                model.cached_ahu_setpoint = 10.0
                model.ahu_working_states = [True, True, True, True]
                self.weather.current_indoor_temperature = 13.0  # excess of 3 °C

                applied: list[float] = []

                async def record(setpoint: float, respect_lower_limit: bool = True) -> None:
                    applied.append(setpoint)

                model.apply_ahu_setpoints = record  # type: ignore[method-assign]

                stop = await model.apply_ahu_off_catchup()

                # Still overshooting, so the loop should continue.
                self.assertFalse(stop)
                self.assertEqual(applied, [expected])

    async def test_ahu_off_catchup_completes_within_threshold(self) -> None:
        """Once ambient is within the threshold, the base setpoint is restored
        and the loop reports completion."""
        model = self.make_model(ahu_off_catchup_rate=1.0, ahu_off_catchup_threshold=1.0)
        model.cached_ahu_setpoint = 10.0
        model.ahu_working_states = [True, True, True, True]
        self.weather.current_indoor_temperature = 10.5  # excess of 0.5 °C (<= threshold)

        applied: list[float] = []

        async def record(setpoint: float, respect_lower_limit: bool = True) -> None:
            applied.append(setpoint)

        model.apply_ahu_setpoints = record  # type: ignore[method-assign]

        stop = await model.apply_ahu_off_catchup()

        self.assertTrue(stop)
        self.assertEqual(applied, [10.0])

    async def test_ahu_off_catchup_clamps_to_lower_limit(self) -> None:
        """The catch-up setpoint never drops below setpoint_lower_limit."""
        model = self.make_model(
            ahu_off_catchup_rate=1.0,
            ahu_off_catchup_threshold=1.0,
            setpoint_lower_limit=6.0,
        )
        model.cached_ahu_setpoint = 7.0
        model.ahu_working_states = [True, True, True, True]
        self.weather.current_indoor_temperature = 20.0  # large overshoot

        self.hvac.ahu_setpoints.clear()
        stop = await model.apply_ahu_off_catchup()
        await asyncio.sleep(STD_SLEEP)  # let the command task reach the mock

        self.assertFalse(stop)
        for ahu in (
            DeviceId.airHandlingUnit01Dome,
            DeviceId.airHandlingUnit02Dome,
            DeviceId.airHandlingUnit03Dome,
            DeviceId.airHandlingUnit04Dome,
        ):
            self.assertEqual(self.hvac.ahu_setpoints[ahu], 6.0)
        self.hvac.ahu_setpoints.clear()

    async def test_ahu_off_catchup_stops_when_context_gone(self) -> None:
        """The catch-up reports completion (and sends nothing) when its
        regulating context no longer holds."""
        applied: list[float] = []

        async def record(setpoint: float, respect_lower_limit: bool = True) -> None:
            applied.append(setpoint)

        # (case name, is night, dome closed, fourth AHU on, features disabled)
        cases: list[tuple[str, bool, bool, bool, list[str]]] = [
            ("night", True, True, True, []),
            ("dome_open", False, False, True, []),
            ("ahu_off", False, True, False, []),
            ("disabled", False, True, True, ["ahu_off_catchup"]),
        ]
        for name, night, dome_closed, fourth_on, features in cases:
            with self.subTest(case=name):
                model = self.make_model(features_to_disable=features)
                model.cached_ahu_setpoint = 10.0
                model.ahu_working_states = [True, True, True, fourth_on]
                self.diurnal._night = night
                self.dome.is_closed = dome_closed
                self.weather.current_indoor_temperature = 20.0
                applied.clear()
                model.apply_ahu_setpoints = record  # type: ignore[method-assign]

                stop = await model.apply_ahu_off_catchup()

                self.assertTrue(stop)
                self.assertEqual(applied, [])

    async def test_ahu_off_catchup_waits_for_data(self) -> None:
        """Missing base setpoint or ambient temperature keeps the loop polling
        without sending a command."""

        async def record(setpoint: float, respect_lower_limit: bool = True) -> None:
            raise AssertionError("No setpoint should be sent while data is missing.")

        # No base setpoint yet.
        model = self.make_model()
        model.ahu_working_states = [True, True, True, True]
        model.cached_ahu_setpoint = None
        model.apply_ahu_setpoints = record  # type: ignore[method-assign]
        self.assertFalse(await model.apply_ahu_off_catchup())

        # Base setpoint known but ambient temperature unavailable.
        model.cached_ahu_setpoint = 10.0
        self.weather.current_indoor_temperature = math.nan
        self.assertFalse(await model.apply_ahu_off_catchup())

    async def test_recovery_manages_catchup_task(self) -> None:
        """Recovery (all AHUs back on after any number were off) spawns the
        catch-up loop, and an AHU going off again cancels it."""
        model = self.make_model(ahu_off_catchup_deltas=[0.0, -1.0, -2.0, -3.0])
        model.cached_ahu_setpoint = 10.0

        async def record(setpoint: float, respect_lower_limit: bool = True) -> None:
            pass

        model.apply_ahu_setpoints = record  # type: ignore[method-assign]

        async def set_ahu(ahu: int, on: bool) -> None:
            await model.ahu_working_state_callback(ahu, types.SimpleNamespace(workingState=on))

        # All AHUs on from the start: no recovery, so no catch-up task.
        for ahu in range(1, 5):
            await set_ahu(ahu, True)
        self.assertIsNone(model.ahu_off_catchup_task)

        # A single AHU goes off: still no overshoot task (stage one owns the
        # setpoint). One off uses a zero delta, so this also confirms the
        # trigger does not depend on a non-zero catch-up having been held.
        await set_ahu(1, False)
        self.assertIsNone(model.ahu_off_catchup_task)

        # Back on: AHU recovery spawns the overshoot task.
        await set_ahu(1, True)
        task = model.ahu_off_catchup_task
        self.assertIsNotNone(task)

        # An AHU goes off again: the task is cancelled and cleared.
        await set_ahu(3, False)
        self.assertIsNone(model.ahu_off_catchup_task)
        await asyncio.sleep(0)  # let the cancellation settle
        assert task is not None
        self.assertTrue(task.cancelled())

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
