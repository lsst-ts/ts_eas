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
import pathlib
import typing
import unittest

import numpy as np
import pandas as pd
from astropy.table import Table
from lsst.ts import eas, salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

STD_TIMEOUT = 60
STD_SLEEP = 3

TEST_CONFIG_DIR = pathlib.Path(__file__).parents[1].joinpath("tests", "config")
TEST_EFD_DIR = pathlib.Path(__file__).parent

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)


class MockInfluxDBClient:
    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}  # measurement -> DataFrame

    async def __aenter__(self) -> "MockInfluxDBClient":
        return self

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException],
        exc_val: BaseException,
        exc_tb: object | None,
    ) -> None:
        pass

    async def create_database(self) -> None:
        pass  # no-op for mock

    async def write(self, df: pd.DataFrame, measurement: str) -> None:
        self._store[measurement] = df

    async def query(self, query: str, **kwargs: typing.Any) -> pd.DataFrame:
        # For simplicity, pretend the query contains the measurement name
        for measurement, df in self._store.items():
            if measurement in query:
                return df
        return pd.DataFrame()


class CscTestCase(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    @contextlib.asynccontextmanager
    async def mock_extra_cscs(self) -> typing.AsyncGenerator[None, None]:
        self.ahu1_state: bool | None = None
        self.ahu2_state: bool | None = None
        self.ahu3_state: bool | None = None
        self.ahu4_state: bool | None = None
        self.vec04_state: bool | None = None

        self.mtdome = salobj.Controller("MTDome")
        self.hvac = salobj.Controller("HVAC")
        self.ess = salobj.Controller("ESS", 301)

        await asyncio.wait_for(
            asyncio.gather(
                self.mtdome.start_task,
                self.hvac.start_task,
                self.ess.start_task,
            ),
            timeout=STD_TIMEOUT,
        )

        self.hvac.cmd_enableDevice.callback = self.enable_callback
        self.hvac.cmd_disableDevice.callback = self.disable_callback

        try:
            yield
        finally:
            await self.mtdome.close()
            await self.hvac.close()
            await self.ess.close()

    async def enable_callback(self, message: salobj.topics.BaseTopic.DataType) -> None:
        """Callback for HVAC.cmd_enableDevice."""
        self.log.info("enable_callback {message.device_id=}")
        match message.device_id:
            case DeviceId.lowerAHU01P05:
                self.ahu1_state = True
            case DeviceId.lowerAHU02P05:
                self.ahu2_state = True
            case DeviceId.lowerAHU03P05:
                self.ahu3_state = True
            case DeviceId.lowerAHU04P05:
                self.ahu4_state = True
            case DeviceId.lowerDamperFan03P04:
                self.vec04_state = True

    async def disable_callback(self, message: salobj.topics.BaseTopic.DataType) -> None:
        """Callback for HVAC.cmd_disableDevice."""
        self.log.info("disable_callback {message.device_id=}")
        match message.device_id:
            case DeviceId.lowerAHU01P05:
                self.ahu1_state = False
            case DeviceId.lowerAHU02P05:
                self.ahu2_state = False
            case DeviceId.lowerAHU03P05:
                self.ahu3_state = False
            case DeviceId.lowerAHU04P05:
                self.ahu4_state = False
            case DeviceId.lowerDamperFan03P04:
                self.vec04_state = False

    def basic_make_csc(
        self,
        initial_state: salobj.State,
        config_dir: str,
        simulation_mode: int,
        **kwargs: typing.Any,
    ) -> None:
        self.log = logging.getLogger("CscTestCase")
        csc = eas.EasCsc(
            initial_state=initial_state,
            config_dir=config_dir,
            simulation_mode=simulation_mode,
        )
        csc.efd_client = self.efd_client_db
        return csc

    @contextlib.asynccontextmanager
    async def efd_client(
        self, table_file: str = "air_flow.ecsv"
    ) -> typing.AsyncGenerator[None, None]:
        df = Table.read(TEST_EFD_DIR / table_file, format="ascii.ecsv").to_pandas()
        df["salIndex"] = 301

        # Make the data file refer to current conditions
        df["private_sndStamp"] += utils.current_tai() - df["private_sndStamp"].max()

        self.efd_data = df

        async with MockInfluxDBClient() as client:
            await client.write(df, measurement="lsst.sal.ESS.airFlow")
            self.efd_client_db = client

            yield

    async def test_standard_state_transitions(self) -> None:
        async with (
            self.efd_client(),
            self.make_csc(
                initial_state=salobj.State.STANDBY,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            await self.check_standard_state_transitions(
                enabled_commands=(),
            )

    async def test_version(self) -> None:
        async with (
            self.efd_client(),
            self.make_csc(
                initial_state=salobj.State.STANDBY,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            await self.assert_next_sample(
                self.remote.evt_softwareVersions,
                cscVersion=eas.__version__,
                subsystemVersions="",
            )

    async def test_bin_script(self) -> None:
        await self.check_bin_script(name="EAS", index=None, exe_name="run_eas")

    async def test_dome_closed(self) -> None:
        """For dome closed, AHUs should turn on and VEC-04 should turn off."""
        async with (
            self.mock_extra_cscs(),
            self.efd_client(),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.sleep(STD_SLEEP)

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(0.0, 0.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.efd_data["speed"].mean()
        )
        self.assertEqual(self.ahu1_state, True)
        self.assertEqual(self.ahu2_state, True)
        self.assertEqual(self.ahu3_state, True)
        self.assertEqual(self.ahu4_state, True)
        self.assertEqual(self.vec04_state, False)

    async def test_dome_open(self) -> None:
        """For dome open, AHUs should turn off and VEC-04 should turn on."""
        async with (
            self.mock_extra_cscs(),
            self.efd_client(),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.sleep(STD_SLEEP)

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(1.0, 1.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.efd_data["speed"].mean()
        )
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, True)

    async def test_efd_wind(self) -> None:
        """In high wind, turn VEC-04 off."""
        async with (
            self.mock_extra_cscs(),
            self.efd_client("high_wind.ecsv"),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.sleep(STD_SLEEP)

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(1.0, 1.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.efd_data["speed"].mean()
        )
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, False)

    async def test_csc_wind(self) -> None:
        """Respond correctly to wind data from the ESS CSC."""
        samples = (100, 110, 110)
        async with (
            self.mock_extra_cscs(),
            self.efd_client("empty.ecsv"),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)

            for sample in samples:
                await self.ess.tel_airFlow.set_write(
                    speed=sample,
                )

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(1.0, 1.0),
            )
            await asyncio.sleep(STD_SLEEP)

        self.assertAlmostEqual(self.csc.average_windspeed, np.mean(samples))
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, False)

    async def test_no_wind(self) -> None:
        """If no wind data available, turn VEC-04 off."""
        async with (
            self.mock_extra_cscs(),
            self.efd_client("empty.ecsv"),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(1.0, 1.0),
            )
            await asyncio.sleep(STD_SLEEP)

        self.assertTrue(np.isnan(self.csc.average_windspeed))
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, False)

    async def test_stale_wind_data(self) -> None:
        """Old wind data should be filtered out.

        The data file stale_wind_data.ecsv contains
        recent wind data plus old data with 100 m/s
        added to it.
        """
        async with (
            self.mock_extra_cscs(),
            self.efd_client("stale_wind_data.ecsv"),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.sleep(STD_SLEEP)

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(1.0, 1.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)

        mask = self.efd_data["speed"] < 100
        self.efd_data = self.efd_data[mask]

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.efd_data["speed"].mean()
        )
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, True)
