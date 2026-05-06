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

__all__ = ["WeatherForecastModel"]

import logging
import math
from typing import Callable

import numpy as np

from lsst.ts import salobj

TEMPERATURE_EPSILON = 0.01

# Expected grid size for the hourlyTrend.temperature data item.
DELTA_TIME = 5 * 60


class WeatherForecastModel:
    """A class to monitor the Prophet-model based forecast.

    This class provides monitoring of the WeatherForecast CSC
    hourlyTrend.temperature. It accepts new telemetry for that
    topic and manages a callback to take action when new forecast
    data are available.

    Parameters
    ----------
    log : `~logging.Logger`
        A logger for log messages.
    """

    def __init__(
        self,
        *,
        log: logging.Logger,
    ) -> None:
        self.log = log

        self.cached_temperature: None | list[float] = None
        self.cached_timestamp: None | float = None

        self.callbacks: dict[int, tuple[float, Callable[[float], None]]] = dict()
        self.next_callback_id: int = 0

    async def hourly_trend_callback(self, hourly_trend: salobj.BaseMsgType) -> None:
        """Callback for WeatherForecast.tel_hourlyTrend

        This function processes incoming hourlyTrend telmetry. If the
        temperature item has changed, it calls the needed callbacks.

        Parameters
        ----------
        hourly_trend: `~lsst.ts.salobj.BaseMsgType`
           A newly received hourlyTrend telemetry item.
        """
        # Compare the temperature item to the cached value. If it hasn't
        # changed, take no action.
        self.log.debug("Received hourlyTrend.")
        if self.cached_temperature is not None and len(self.cached_temperature) == len(
            hourly_trend.temperature
        ):
            for t1, t2 in zip(hourly_trend.temperature, self.cached_temperature):
                if math.fabs(t1 - t2) > TEMPERATURE_EPSILON:
                    break
            else:
                return

        # Cache the received telemetry.
        self.cached_temperature = hourly_trend.temperature
        self.cached_timestamp = hourly_trend.private_sndStamp

        self.call_callbacks()

    def predict_temperature_at_time(self, time: float) -> float | None:
        if self.cached_temperature is None or self.cached_timestamp is None:
            raise RuntimeError("predict_temperature_at_time called with invalid cache")

        # Do an interpolation to get the temperature prediction.
        index_float = (time - self.cached_timestamp) / DELTA_TIME - 1
        if index_float < 0 or index_float > len(self.cached_temperature) - 1:
            return None

        return np.interp(
            index_float,
            np.arange(len(self.cached_temperature)),
            self.cached_temperature,
        )

    def call_callbacks(self) -> None:
        for time, callback in self.callbacks.values():
            prediction = self.predict_temperature_at_time(time)
            if prediction is None:
                continue
            callback(prediction)

    def add_callback(self, time: float, callback: Callable[[float], None]) -> int:
        callback_id = self.next_callback_id
        self.next_callback_id += 1
        self.callbacks[callback_id] = (time, callback)
        return callback_id

    def remove_callback(self, callback_id: int) -> None:
        self.callbacks.pop(callback_id, None)
