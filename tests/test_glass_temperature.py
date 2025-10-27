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
import unittest

from lsst.ts import eas, salobj, utils

STD_TIMEOUT = 10


class TestGlassTemperatureModel(
    salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self.log = logging.getLogger()
        self.domain = salobj.Domain()
        self.temperatures: list[list[float]] | None = None
        self.timestamp: float | None = None

        self.glass_temperature_model = (
            eas.glass_temperature_model.GlassTemperatureModel(
                domain=self.domain, log=self.log
            )
        )
        self.monitor_task = asyncio.create_task(self.glass_temperature_model.monitor())
        await asyncio.wait_for(
            self.glass_temperature_model.monitor_start_event.wait(),
            timeout=STD_TIMEOUT,
        )

        self.thermal_scanners_ready = asyncio.Event()
        self.thermal_scanner_task = asyncio.create_task(self.run_thermal_scanners())
        await asyncio.wait_for(self.thermal_scanners_ready.wait(), timeout=STD_TIMEOUT)

        await super().asyncSetUp()

    async def kill_task(self, task: asyncio.Task) -> None:
        try:
            await asyncio.wait_for(task, timeout=STD_TIMEOUT)
        except Exception:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.log.exception("task raised")
                pass

    async def asyncTearDown(self) -> None:
        self.glass_temperature_model.monitor_stop.set()
        self.running_thermal_scanners = False

        await self.kill_task(self.thermal_scanner_task)
        await self.kill_task(self.monitor_task)

        await super().asyncTearDown()

    async def run_thermal_scanners(self) -> None:
        self.running_thermal_scanners = True
        thermal_scanners = [
            salobj.Controller("ESS", sal_index) for sal_index in (114, 115, 116, 117)
        ]
        for controller in thermal_scanners:
            await controller.start_task

        while self.running_thermal_scanners:
            self.thermal_scanners_ready.set()
            await asyncio.sleep(0.5)
            if self.temperatures is None:
                continue

            timestamp = (
                utils.current_tai() if self.timestamp is None else self.timestamp
            )
            for controller, temperature in zip(thermal_scanners, self.temperatures):
                for i in range(6):
                    await controller.tel_temperature.set_write(
                        sensorName=f"m1m3-ts-0{controller.salinfo.index-114} {i + 1}/6",
                        timestamp=timestamp,
                        numChannels=15 if i == 5 else 16,
                        temperatureItem=temperature,
                        location="",
                    )

        for controller in thermal_scanners:
            await controller.close()

    async def test_glass_temperature(self) -> None:
        """In the normal case, the first channel should be ignored."""
        self.temperatures = [[3.14] * 16 for _ in range(4)]
        for i in range(4):
            self.temperatures[i][0] = 1_000_000_000

        await asyncio.sleep(4)  # Give time for the telemetry to get through
        median_temperature = self.glass_temperature_model.median_temperature

        self.assertFalse(median_temperature is None)
        self.assertAlmostEqual(median_temperature, 3.14, 3)

    async def test_glass_median(self) -> None:
        """Model should compute median."""
        self.temperatures = [
            list(range(100, 116)),
            list(range(200, 216)),
            list(range(300, 316)),
            list(range(400, 416)),
        ]

        await asyncio.sleep(2)  # Give time for the telemetry to get through
        median_temperature = self.glass_temperature_model.median_temperature

        self.assertFalse(median_temperature is None)
        self.assertAlmostEqual(median_temperature, 214.0, 3)

    async def test_glass_no_data(self) -> None:
        """If no data provided, median temperature should report as None"""
        await asyncio.sleep(2)  # Give time for the telemetry to get through
        median_temperature = self.glass_temperature_model.median_temperature
        self.assertTrue(median_temperature is None)

    async def test_glass_old_data(self) -> None:
        """If stale data provided, median temperature should report as None"""
        self.temperatures = [[3.14] * 16 for _ in range(4)]
        self.timestamp = utils.current_tai() - 3600

        await asyncio.sleep(2)
        median_temperature = self.glass_temperature_model.median_temperature
        self.assertTrue(median_temperature is None)

    def basic_make_csc(
        self,
        initial_state: salobj.State,
        config_dir: str,
        simulation_mode: int,
    ) -> None:
        raise NotImplementedError("Not used in this test.")
