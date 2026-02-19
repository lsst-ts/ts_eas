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

import logging
import unittest
from types import SimpleNamespace

from lsst.ts.eas.weatherforecast_model import DELTA_TIME, WeatherForecastModel

# Starting timestamp for the tests (sec)
timestamp0 = 1000.0


def _prediction(time: float) -> float:
    """A dummy forecast that starts at 9.0°C at timestamp0 and increases at
    the rate of 1.0°C every 5 minutes.

    Parameters
    ----------
    time : `float`
        The time (TAI seconds) of the desired forecast data point.

    Returns
    -------
    `float`
        Forecast temperature in °C.
    """
    m = 1.0 / 300.0
    b = 9.0 - m * timestamp0

    return m * time + b


class TestWeatherForecastModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.log = logging.getLogger("weatherforecast")
        self.model = WeatherForecastModel(log=self.log)

    async def test_single_callback_lifecycle(self) -> None:
        """The forecast model should call a callback and respect removal."""
        callback_values: list[float] = []

        def callback(value: float) -> None:
            callback_values.append(value)

        target_time = timestamp0 + 47.5 * 60

        callback_id = self.model.add_callback(target_time, callback)

        temperatures = [_prediction(timestamp0 + (idx + 1) * DELTA_TIME) for idx in range(12)]
        telemetry = SimpleNamespace(
            temperature=temperatures,
            private_sndStamp=timestamp0,
        )

        await self.model.hourly_trend_callback(telemetry)
        self.assertEqual(len(callback_values), 1)
        expected = _prediction(target_time)
        self.assertAlmostEqual(callback_values[0], expected)

        self.model.remove_callback(callback_id)
        self.assertDictEqual(self.model.callbacks, {})

        timestamp2 = timestamp0 + DELTA_TIME
        temperatures2 = [_prediction(timestamp2 + (idx + 1) * DELTA_TIME) for idx in range(12)]
        telemetry2 = SimpleNamespace(
            temperature=temperatures2,
            private_sndStamp=timestamp2,
        )
        await self.model.hourly_trend_callback(telemetry2)
        self.assertEqual(len(callback_values), 1)

    async def test_two_callbacks_lifecycle(self) -> None:
        """The forecast model should handle multiple callbacks."""
        callback_values: list[float] = []

        def callback(value: float) -> None:
            callback_values.append(value)

        target_time_1 = timestamp0 + 46 * 60
        target_time_2 = timestamp0 + 56 * 60

        callback_id_1 = self.model.add_callback(target_time_1, callback)
        callback_id_2 = self.model.add_callback(target_time_2, callback)

        temperatures = [_prediction(timestamp0 + (idx + 1) * DELTA_TIME) for idx in range(12)]
        telemetry = SimpleNamespace(
            temperature=temperatures,
            private_sndStamp=timestamp0,
        )

        await self.model.hourly_trend_callback(telemetry)
        self.assertEqual(len(callback_values), 2)
        expected_1 = _prediction(target_time_1)
        expected_2 = _prediction(target_time_2)
        self.assertAlmostEqual(callback_values[0], expected_1)
        self.assertAlmostEqual(callback_values[1], expected_2)

        self.model.remove_callback(callback_id_1)
        self.model.remove_callback(callback_id_2)
        self.assertDictEqual(self.model.callbacks, {})

        timestamp2 = timestamp0 + DELTA_TIME
        temperatures2 = [_prediction(timestamp2 + (idx + 1) * DELTA_TIME) for idx in range(12)]
        telemetry2 = SimpleNamespace(
            temperature=temperatures2,
            private_sndStamp=timestamp2,
        )
        await self.model.hourly_trend_callback(telemetry2)
        self.assertEqual(len(callback_values), 2)

    async def test_callback_not_called_when_target_in_past(self) -> None:
        """The model should not call a callback when target time has passed."""
        callback_values: list[float] = []

        def callback(value: float) -> None:
            callback_values.append(value)

        target_time = timestamp0 - 60.0
        self.model.add_callback(target_time, callback)

        temperatures = [_prediction(timestamp0 + (idx + 1) * DELTA_TIME) for idx in range(12)]
        telemetry = SimpleNamespace(
            temperature=temperatures,
            private_sndStamp=timestamp0,
        )

        await self.model.hourly_trend_callback(telemetry)
        self.assertEqual(len(callback_values), 0)

    async def test_remove_missing_callback_noop(self) -> None:
        """Removing a non-existent callback should be a no-op."""
        self.model.remove_callback(9999)
        self.assertDictEqual(self.model.callbacks, {})
