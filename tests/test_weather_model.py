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
from unittest import mock

import pandas as pd
from astropy.time import Time
from lsst.ts import eas, salobj, utils

STD_TIMEOUT = 10


class MockDiurnalTimer:
    is_running = True
    twilight_condition = asyncio.Condition()

    def get_twilight_time(self, of_date: Time) -> Time:
        return Time("2025-01-01T00:00:00")  # Not important

    def is_night(self, time: Time) -> bool:
        return True


class TestGetLastTwilightTemperature(
    salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        """Constructs a WeatherModel object for testing."""
        log = logging.getLogger()
        self.domain = salobj.Domain()
        self.ess = salobj.Controller("ESS", 301)
        self.indoor_ess = salobj.Controller("ESS", 112)
        self.diurnal_timer = MockDiurnalTimer()
        await self.ess.start_task
        await self.indoor_ess.start_task

        self.weather_model = eas.weather_model.WeatherModel(
            domain=self.domain,
            log=log,
            diurnal_timer=self.diurnal_timer,
            efd_name="mocked",
            ess_index=301,
            indoor_ess_index=112,
            wind_average_window=1800,
            wind_minimum_window=600,
        )
        await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        try:
            await self.ess.close()
            await self.indoor_ess.close()
            await self.domain.close()

        finally:
            await super().asyncTearDown()

    async def test_returns_cached_value_when_present(self) -> None:
        """The model should not use EFD if a value is cached."""
        self.weather_model.last_twilight_temperature = 7.0
        with mock.patch("lsst_efd_client.EfdClient") as EfdClient:
            result = await self.weather_model.get_last_twilight_temperature()
        self.assertEqual(result, 7.0)
        EfdClient.assert_not_called()

    async def test_fetches_and_returns_median_temperature(self) -> None:
        """The model should use the median of all samples returned from EFD."""
        # First day: empty -> forces loop to continue
        # Second day: valid data -> median should be 8.0
        df_empty = pd.DataFrame(columns=["temperatureItem0"])
        df_valid = pd.DataFrame({"temperatureItem0": [7.0, 8.0, 9.0]})

        mock_client = mock.MagicMock()
        mock_client.select_time_series = mock.AsyncMock(
            side_effect=[
                df_empty,  # day 1
                df_valid,  # day 2
            ]
        )

        with mock.patch(
            "lsst_efd_client.EfdClient", return_value=mock_client
        ) as EfdClient:
            result = await self.weather_model.get_last_twilight_temperature()

        self.assertEqual(result, 8.0)
        # Confirm cache populated
        self.assertEqual(self.weather_model.last_twilight_temperature, 8.0)
        # Confirm EfdClient was constructed only once.
        self.assertGreaterEqual(EfdClient.call_count, 1)
        # Check select_time_series called with expected args on the second call
        args_list = mock_client.select_time_series.call_args_list
        self.assertEqual(args_list[1].args[0], "lsst.sal.ESS.temperature")
        self.assertEqual(args_list[1].args[1], ["temperatureItem0"])
        self.assertEqual(args_list[1].kwargs["index"], self.weather_model.ess_index)

    async def test_returns_none_when_all_days_missing_or_nan(self) -> None:
        """The model should return None if EFD data are missing or invalid.

        The "NaN" scenario is not expected but it doesn't hurt anything to
        be prepared for the possibility.
        """
        # Mix of empty frames and frames whose median is NaN
        df_empty = pd.DataFrame(columns=["temperatureItem0"])
        df_nan = pd.DataFrame({"temperatureItem0": [math.nan, math.nan]})
        side_effect = [df_empty, df_nan] * 5  # 10 array elements for 10 days of history

        mock_client = mock.MagicMock()
        mock_client.select_time_series = mock.AsyncMock(side_effect=side_effect)

        with mock.patch("lsst_efd_client.EfdClient", return_value=mock_client):
            result = await self.weather_model.get_last_twilight_temperature()

        self.assertIsNone(result)
        self.assertIsNone(self.weather_model.last_twilight_temperature)

    async def test_temperature_updated(self) -> None:
        """At twilight, the last twilight_temperature should be updated."""

        monitor_task = asyncio.create_task(self.weather_model.monitor())
        await asyncio.wait_for(
            self.weather_model.monitor_start_event.wait(), timeout=STD_TIMEOUT
        )

        for i in range(2):
            await self.ess.tel_temperature.set_write(
                sensorName="",
                timestamp=utils.current_tai(),
                numChannels=1,
                temperatureItem=[12.0] + [0.0] * 15,
                location="",
            )
            await self.indoor_ess.tel_dewPoint.set_write(
                sensorName="",
                timestamp=utils.current_tai(),
                dewPointItem=-10.0 + i,
                location="",
            )

        await asyncio.sleep(1)  # Give the telemetry time to get through
        async with self.diurnal_timer.twilight_condition:
            self.diurnal_timer.twilight_condition.notify_all()

        # Pass control to the event loop so that the condition can notify:
        await asyncio.sleep(0)

        with mock.patch("lsst_efd_client.EfdClient") as EfdClient:
            result = await self.weather_model.get_last_twilight_temperature()
        self.assertEqual(result, 12.0)

        # Dewpoint from first sample is -10., second sample is -9.
        self.assertEqual(self.weather_model.nightly_maximum_indoor_dew_point, -9.0)
        EfdClient.assert_not_called()

        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    def basic_make_csc(
        self,
        initial_state: None,
        config_dir: None,
        simulation_mode: int,
    ) -> salobj.BaseCsc:
        raise NotImplementedError("Not actually using basic_make_csc for anything.")
