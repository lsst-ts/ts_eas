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

__all__ = ["GlassTemperatureModel"]

import logging
import re
import statistics
from dataclasses import dataclass

from lsst.ts import salobj, utils
from lsst.ts.xml.tables.m1m3 import find_thermocouple

N_THERMOCOUPLES = 146
MAX_TIMESTAMP_AGE = 300


@dataclass(frozen=True)
class ThermocoupleSample:
    """An EFD sample obtained from the M1M3TS thermocouples."""

    temperature: float | None
    timestamp: float | None


class GlassTemperatureModel:
    """A class to monitor the glass temperature for control of FCU fan speed.

    This class provides monitoring of the glass temperature probes.
    Fan speed scales linearly from 500 RPM at zero temperature difference
    to 2000 RPM at ±1.0 °C difference. Values in between are interpolated,
    and differences greater than or equal to 1.0 °C run at maximum speed.

    The glass temperature probes publish to four different ESS CSCs, and
    the measurements are multiplexed using the `sensorName` item in the
    temperature telemetry.

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

        # A list of most recent temperature probe samples.
        self.thermocouple_cache: list[ThermocoupleSample | None] = [
            None
        ] * N_THERMOCOUPLES

        self.compiled_regex = re.compile(r"m1m3-ts-\d+ (\d+)/\d+")

    async def temperature_callback(self, temperature: salobj.BaseMsgType) -> None:
        """Callback for ESS.tel_temperature.

        This function sorts out the data in the temperatureItem
        array and stores it in the appropriate place in
        `self.thermocouple_cache`. It is very much tailored
        to the existing structure of the M1M3 thermal scanner
        telemetry.

        Parameters
        ----------
        temperature : `~lsst.ts.salobj.BaseMsgType`
           A newly received temperature telemetry item.
        """
        sal_index = temperature.salIndex
        sensor_name = temperature.sensorName  # Has format 'm1m3-ts-<index> <n>/6'
        regex_match = self.compiled_regex.match(temperature.sensorName)
        if regex_match is None:
            message = (
                f"M1M3TS ESS temperature sample has unexpected sensorName {sensor_name}"
            )
            self.log.error(message)
            raise RuntimeError(message)
        sequence_num = int(regex_match[1])

        n_channels = temperature.numChannels
        temperatures = temperature.temperatureItem
        temperatures = temperatures[:n_channels]

        timestamp = temperature.timestamp

        for i, temperature in enumerate(temperature.temperatureItem):
            channel = 16 * sequence_num + i
            thermocouple = find_thermocouple(sal_index, channel)
            if thermocouple is None:
                continue

            if thermocouple.index < 0 or thermocouple.index >= N_THERMOCOUPLES:
                raise ValueError("Unexpected thermocouple index")

            self.thermocouple_cache[thermocouple.index] = ThermocoupleSample(
                temperature, timestamp
            )

    @property
    def median_temperature(self) -> float | None:
        """Compute median glass temperature.

        This getter finds all temperature samples
        that are less than 5 minutes old, and
        computes the median. If no samples are
        available, None is returned.

        Returns
        -------
        float | None
            The median of all the most recent temperature
            samples, or None if no temperature samples are
            available.
        """
        cutoff_time = utils.current_tai() - MAX_TIMESTAMP_AGE
        valid_temperatures = [
            sample.temperature
            for sample in self.thermocouple_cache
            if (
                sample is not None
                and sample.temperature is not None
                and sample.timestamp is not None
                and sample.timestamp > cutoff_time
            )
        ]
        if not valid_temperatures:
            return None
        return statistics.median(valid_temperatures)
