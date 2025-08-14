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

import asyncio
import logging
import statistics
from contextlib import AsyncExitStack

from lsst.ts import salobj, utils
from lsst.ts.xml.tables.m1m3 import find_thermocouple

SAL_INDICES = (114, 115, 116, 117)
N_THERMOCOUPLES = 146
MAX_TIMESTAMP_AGE = 300


class GlassTemperatureModel:
    """A class to monitor the glass temperature for control of FCU fan speed.

    This class provides monitoring of the glass temperature probes.
    Fan speed scales linearly from 500 RPM at zero temperature difference
    to 2500 RPM at ±1.0 °C difference. Values in between are interpolated,
    and differences greater than or equal to 1.0 °C run at maximum speed.

    The glass temperature probes publish to four different ESS CSCs, and
    the measurements are multiplexed using the `sensorName` item in the
    temperature telemetry.

    Parameters
    ----------
    domain : `~lsst.ts.salobj.Domain`
        A SAL domain object for obtaining remotes.
    log : `~logging.Logger`
        A logger for log messages.
    """

    def __init__(self, *, domain: salobj.Domain, log: logging.Logger) -> None:
        self.domain = domain
        self.log = log

        self.monitor_start_event = asyncio.Event()
        self.monitor_stop = asyncio.Event()

        # A list of most recent temperature probe samples.
        self.thermocouple_cache: list[tuple[float | None, float | None]] = [
            (None, None)
        ] * N_THERMOCOUPLES

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
        sequence_num = int(sensor_name.split()[1].split("/")[0])

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

            self.thermocouple_cache[thermocouple.index] = (temperature, timestamp)

    @property
    def median_temperature(self) -> float | None:
        """Computes median glass temperature.

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
        self.log.debug("median_temperature")
        cutoff_time = utils.current_tai() - MAX_TIMESTAMP_AGE
        valid_temperatures = [
            temperature
            for temperature, timestamp in self.thermocouple_cache
            if (
                temperature is not None
                and timestamp is not None
                and timestamp > cutoff_time
            )
        ]
        if not valid_temperatures:
            return None
        return statistics.median(valid_temperatures)

    async def monitor(self) -> None:
        """Starts the monitor for this model.

        Connects to the four thermal scanner ESS controllers,
        sets up callbacks, and idles.
        """
        self.log.debug("GlassTemperatureModel.monitor")
        indices = (114, 115, 116, 117)

        async with AsyncExitStack() as stack:
            remotes = []
            for index in indices:
                remote = await stack.enter_async_context(
                    salobj.Remote(
                        domain=self.domain,
                        name="ESS",
                        index=index,
                        include=("temperature",),
                    )
                )
                remote.tel_temperature.callback = self.temperature_callback
                remotes.append(remote)

            self.monitor_start_event.set()

            await self.monitor_stop.wait()

        self.log.debug("GlassTemperatureModel monitor stops.")
