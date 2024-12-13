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

import contextlib
import logging
import typing
import unittest

from lsst.ts import eas, salobj

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)


class CscTestCase(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    @contextlib.asynccontextmanager
    async def mock_mtm1m3ts(self) -> typing.AsyncGenerator[None, None]:
        try:
            async with salobj.Controller("MTM1M3TS") as self.mtm1m3ts:
                log = logging.getLogger("root")
                log.error("Started MTM1M3TS.")
                await self.mtm1m3ts.evt_summaryState.set_write(
                    summaryState=salobj.State.DISABLED
                )
                await self.mtm1m3ts.tel_mixingValve.set_write(
                    rawValvePosition=0, valvePosition=0
                )
                yield
        except Exception as exception:
            raise exception

    def basic_make_csc(
        self,
        initial_state: salobj.State,
        config_dir: str,
        simulation_mode: int,
        **kwargs: typing.Any,
    ) -> None:
        return eas.EasCsc(
            initial_state=initial_state,
            config_dir=config_dir,
            simulation_mode=simulation_mode,
        )

    async def test_standard_state_transitions(self) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            simulation_mode=1,
        ), self.mock_mtm1m3ts():
            await self.check_standard_state_transitions(
                enabled_commands=(),
            )

    async def test_version(self) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            simulation_mode=1,
        ):
            await self.assert_next_sample(
                self.remote.evt_softwareVersions,
                cscVersion=eas.__version__,
                subsystemVersions="",
            )

    async def test_bin_script(self) -> None:
        await self.check_bin_script(name="EAS", index=None, exe_name="run_eas")
