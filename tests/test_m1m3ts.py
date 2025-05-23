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
import os
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
        )
        self.glycol_setpoint: float | None = None
        self.heater_setpoint: float | None = None

    async def do_applySetpoints(self, data: salobj.BaseMsgType) -> None:
        self.glycol_setpoint = data.glycolSetpoint
        self.heater_setpoint = data.heatersSetpoint


class TestM1M3(unittest.IsolatedAsyncioTestCase):
    def run(self, result: typing.Any = None) -> None:
        salobj.testutils.set_test_topic_subname(randomize=False)
        os.environ["LSST_SITE"] = "test"
        super().run(result)  # type: ignore

    @contextlib.asynccontextmanager
    async def mock_m1m3ts(self) -> typing.AsyncGenerator[None, None]:
        self.m1m3ts = M1M3TSMock()
        await self.m1m3ts.start_task

        try:
            yield
        finally:
            try:
                await self.m1m3ts.close()
            except asyncio.CancelledError:
                pass

    async def run_with_parameters(
        self, twilight_temperature: float, **model_args: typing.Any
    ) -> tuple[float | None, float | None]:
        diurnal_timer = eas.diurnal_timer.DiurnalTimer()
        diurnal_timer.is_running = True

        weather_model = SimpleNamespace(last_twilight_temperature=twilight_temperature)
        async with M1M3TSMock() as mock_m1m3ts, salobj.Domain() as domain:
            self.m1m3ts_model = eas.m1m3ts_model.M1M3TSModel(
                domain=domain,
                log=mock_m1m3ts.log,
                diurnal_timer=diurnal_timer,
                weather_model=weather_model,
                glycol_setpoint_delta=model_args["glycol_setpoint_delta"],
                heater_setpoint_delta=model_args["heater_setpoint_delta"],
                features_to_disable=model_args["features_to_disable"],
            )
            monitor_task = asyncio.create_task(self.m1m3ts_model.monitor())
            await asyncio.sleep(1)

            # Fake a noon condition...
            async with diurnal_timer.noon_condition:
                diurnal_timer.noon_condition.notify_all()
            await asyncio.sleep(1)

            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return mock_m1m3ts.glycol_setpoint, mock_m1m3ts.heater_setpoint

    async def test_noon(self) -> None:
        """M1M3TS.applySetpoints should be called at noon."""
        last_twilight_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1

        glycol_setpoint, heater_setpoint = await self.run_with_parameters(
            last_twilight_temperature,
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            features_to_disable=[],
        )

        assert glycol_setpoint is not None
        self.assertAlmostEqual(
            glycol_setpoint,
            last_twilight_temperature + glycol_setpoint_delta,
            places=4,
        )

        assert heater_setpoint is not None
        self.assertAlmostEqual(
            heater_setpoint,
            last_twilight_temperature + heater_setpoint_delta,
            places=4,
        )

    async def test_disabled(self) -> None:
        """The applySetpoints should not be called when m1m3ts is disabled."""
        last_twilight_temperature = 10
        glycol_setpoint_delta = -2
        heater_setpoint_delta = -1

        glycol_setpoint, heater_setpoint = await self.run_with_parameters(
            last_twilight_temperature,
            glycol_setpoint_delta=glycol_setpoint_delta,
            heater_setpoint_delta=heater_setpoint_delta,
            features_to_disable=["m1m3ts"],
        )

        self.assertTrue(glycol_setpoint is None)
        self.assertTrue(heater_setpoint is None)
