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
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from lsst.ts import eas, salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

STD_TIMEOUT = 60
STD_SLEEP = 5
LONG_SLEEP = 15

TEST_CONFIG_DIR = pathlib.Path(__file__).parents[1].joinpath("tests", "config")
TEST_WIND_DATA_DIR = pathlib.Path(__file__).parent

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG
)


class MTMountMock(salobj.BaseCsc):
    version = "?"

    def __init__(self) -> None:
        self.valid_simulation_modes = (0,)
        super().__init__(
            name="MTMount",
            index=None,
            initial_state=salobj.State.ENABLED,
            allow_missing_callbacks=True,
        )
        self.top_end_setpoint: float | None = None

    async def do_setThermal(self, data: salobj.BaseMsgType) -> None:
        self.top_end_setpoint = data.topEndChillerSetpoint


class CscTestCase(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.fake_time = SimpleNamespace(offset=0.0)
        self._real_current_tai = utils.current_tai

        def fake_current_tai() -> float:
            return self._real_current_tai() + self.fake_time.offset

        self.current_tai_patcher = mock.patch(
            "lsst.ts.utils.current_tai", side_effect=fake_current_tai
        )
        self.mock_current_tai = self.current_tai_patcher.start()
        self.addCleanup(self.current_tai_patcher.stop)

        patcher = mock.patch("lsst_efd_client.EfdClient", autospec=True)
        self.MockEfdClient = patcher.start()
        self.addCleanup(patcher.stop)

        mock_instance = self.MockEfdClient.return_value

        async def fake_select_time_series(
            topic: str,
            fields: list[str],
            start: Time,
            end: Time,
            index: int,
        ) -> pd.DataFrame:
            """Always return a minimal single-row DataFrame."""
            return pd.DataFrame({"temperatureItem0": [0.0], "dewPointItem": [-10.0]})

        mock_instance.select_time_series = mock.AsyncMock(
            side_effect=fake_select_time_series
        )
        self.mock_efd_client = mock_instance

    def offset_clock(self, offset: float) -> None:
        """Applies an offset to current_tai clock mock.

        The offset must increase monotonically with each call.

        Parameters
        ----------
        offset : float
            The offset to apply to `current_tai` (seconds).
        """
        self.assertGreaterEqual(
            offset,
            self.fake_time.offset,
            msg="Clock offset must increase monotonically.",
        )
        self.fake_time.offset = offset

    @contextlib.asynccontextmanager
    async def mock_extra_cscs(self) -> typing.AsyncGenerator[None, None]:
        self.ahu1_state: bool | None = None
        self.ahu2_state: bool | None = None
        self.ahu3_state: bool | None = None
        self.ahu4_state: bool | None = None
        self.vec04_state: bool | None = None

        self.hvac_events = {
            DeviceId.lowerAHU01P05: asyncio.Event(),
            DeviceId.lowerAHU02P05: asyncio.Event(),
            DeviceId.lowerAHU03P05: asyncio.Event(),
            DeviceId.lowerAHU04P05: asyncio.Event(),
            DeviceId.loadingBayFan04P04: asyncio.Event(),
        }

        self.mtdome = salobj.Controller("MTDome")
        self.hvac = salobj.Controller("HVAC")
        self.ess = salobj.Controller("ESS", 301)
        self.ess112: salobj.Controller | None = salobj.Controller("ESS", 112)
        self.mtmount = MTMountMock()

        eas.hvac_model.HVAC_SLEEP_TIME = 1

        await asyncio.wait_for(
            asyncio.gather(
                self.mtdome.start_task,
                self.hvac.start_task,
                self.ess.start_task,
                self.ess112.start_task,
                self.mtmount.start_task,
            ),
            timeout=STD_TIMEOUT,
        )

        await self.mtmount.evt_summaryState.set_write(summaryState=salobj.State.ENABLED)

        emit_ess112_temperature_task = asyncio.create_task(
            self.emit_ess112_temperature()
        )

        self.hvac.cmd_enableDevice.callback = self.enable_callback
        self.hvac.cmd_disableDevice.callback = self.disable_callback

        try:
            yield
        finally:
            self.log.warning("Extra CSCs shutting down")
            ess112 = self.ess112
            self.ess112 = None
            emit_ess112_temperature_task.cancel()
            try:
                await emit_ess112_temperature_task
            except asyncio.CancelledError:
                pass

            await self.mtdome.close()
            await self.hvac.close()
            await self.ess.close()
            await self.mtmount.close()
            await ess112.close()

    async def emit_ess112_temperature(self) -> None:
        while True:
            await asyncio.sleep(1)
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            timestamp = 0
            if self.ess112 is not None:
                await self.ess112.tel_temperature.set_write(
                    sensorName="",
                    timestamp=timestamp,
                    numChannels=1,
                    temperatureItem=[0] * 16,
                    location="",
                )
                timestamp += 1

    async def enable_callback(self, message: salobj.topics.BaseTopic.DataType) -> None:
        """Callback for HVAC.cmd_enableDevice."""
        self.log.info(f"enable_callback {message.device_id=}")
        match message.device_id:
            case DeviceId.lowerAHU01P05:
                self.ahu1_state = True
            case DeviceId.lowerAHU02P05:
                self.ahu2_state = True
            case DeviceId.lowerAHU03P05:
                self.ahu3_state = True
            case DeviceId.lowerAHU04P05:
                self.ahu4_state = True
            case DeviceId.loadingBayFan04P04:
                self.vec04_state = True

        self.hvac_events[message.device_id].set()

    async def disable_callback(self, message: salobj.topics.BaseTopic.DataType) -> None:
        """Callback for HVAC.cmd_disableDevice."""
        self.log.info(f"disable_callback {message.device_id=}")
        match message.device_id:
            case DeviceId.lowerAHU01P05:
                self.ahu1_state = False
            case DeviceId.lowerAHU02P05:
                self.ahu2_state = False
            case DeviceId.lowerAHU03P05:
                self.ahu3_state = False
            case DeviceId.lowerAHU04P05:
                self.ahu4_state = False
            case DeviceId.loadingBayFan04P04:
                self.vec04_state = False

        self.hvac_events[message.device_id].set()

    async def wait_for_all_hvac_events(self) -> None:
        await asyncio.wait_for(
            asyncio.gather(*[event.wait() for _, event in self.hvac_events.items()]),
            timeout=STD_TIMEOUT,
        )
        for _, event in self.hvac_events.items():
            event.clear()

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
        return csc

    async def test_standard_state_transitions(self) -> None:
        async with (
            self.mock_extra_cscs(),
            self.make_csc(
                initial_state=salobj.State.STANDBY,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            await self.check_standard_state_transitions(
                enabled_commands=(),
            )

    async def load_wind_history(self, wind_data_file: str) -> None:
        self.wind_data = Table.read(
            TEST_WIND_DATA_DIR / wind_data_file, format="ascii.ecsv"
        )
        self.wind_data["private_sndStamp"] -= self.wind_data["private_sndStamp"].min()
        for row in self.wind_data:
            self.offset_clock(row["private_sndStamp"])
            await self.ess.tel_airFlow.set_write(
                speed=row["speed"],
            )
            await asyncio.sleep(0.1)

    async def test_version(self) -> None:
        async with (
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
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            await self.load_wind_history("air_flow.ecsv")
            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(0.0, 0.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)
            await self.csc.close_tasks()

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.wind_data["speed"].mean(), delta=0.1
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
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            await self.load_wind_history("air_flow.ecsv")
            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(100.0, 100.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)
            await self.csc.close_tasks()

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.wind_data["speed"].mean(), delta=0.1
        )
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, True)

    async def test_high_wind(self) -> None:
        """In high wind, turn VEC-04 off."""
        async with (
            self.mock_extra_cscs(),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            await self.load_wind_history("high_wind.ecsv")
            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(100.0, 100.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)
            await self.csc.close_tasks()

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.wind_data["speed"].mean(), delta=0.1
        )
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, False)

    async def test_fresh_wind(self) -> None:
        """Respond correctly to wind data from the ESS CSC."""
        async with (
            self.mock_extra_cscs(),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(100.0, 100.0),
            )
            await self.ess.tel_airFlow.set_write(
                speed=0,
            )

            await self.wait_for_all_hvac_events()

            # The dome is closed. AHUs are shut off.
            # Wind is reported as low, but VEC-04
            # is off because there is not enough
            # history of wind data.
            self.assertTrue(np.isnan(self.csc.average_windspeed))
            self.assertEqual(self.ahu1_state, False)
            self.assertEqual(self.ahu2_state, False)
            self.assertEqual(self.ahu3_state, False)
            self.assertEqual(self.ahu4_state, False)
            self.assertEqual(self.vec04_state, False)

            self.vec04_state = None

            # Increment by 20 minutes so that we have
            # enough historical data.
            self.offset_clock(1200)
            await self.ess.tel_airFlow.set_write(
                speed=0,
            )

            # Also have to refresh the apertureShutter telemetry
            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(100.0, 100.0),
            )

            # And in the next run of the control loop, the
            # VEC-04 fan is enabled.
            await asyncio.wait_for(
                self.hvac_events[DeviceId.loadingBayFan04P04].wait(),
                timeout=STD_TIMEOUT,
            )

            # Now, we expect a valid windspeed to be reported.
            self.assertAlmostEqual(self.csc.average_windspeed, 0.0)

            self.assertEqual(self.vec04_state, True)
            await self.csc.close_tasks()

    async def test_no_wind(self) -> None:
        """If no wind data available, turn VEC-04 off."""
        async with (
            self.mock_extra_cscs(),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(100.0, 100.0),
            )

            await self.wait_for_all_hvac_events()

            await self.csc.close_tasks()

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
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            # This ecsv file contains 30 minutes of high winds (>100)
            # followed by 30 minutes of low winds.
            await self.load_wind_history("stale_wind_data.ecsv")
            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(100.0, 100.0),
            )
            # Give the telemetry time to propagate into the EAS CSC
            await asyncio.sleep(STD_SLEEP)
            await self.csc.close_tasks()

        mask = self.wind_data["speed"] < 100
        self.wind_data = self.wind_data[mask]

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.wind_data["speed"].mean(), delta=0.1
        )
        self.assertEqual(self.ahu1_state, False)
        self.assertEqual(self.ahu2_state, False)
        self.assertEqual(self.ahu3_state, False)
        self.assertEqual(self.ahu4_state, False)
        self.assertEqual(self.vec04_state, True)

    async def test_csc_dropout(self) -> None:
        """The dome monitor loop should recover gracefully on losing a remote.

        This test is a copy of test_dome_closed, with a disconnect and
        re-connect of the CSCs thrown in.
        """
        async with (
            self.mock_extra_cscs(),
            self.make_csc(
                initial_state=salobj.State.ENABLED,
                config_dir=TEST_CONFIG_DIR,
                simulation_mode=1,
            ),
        ):
            # Give the EAS CSC time to establish an MTDome remote
            await asyncio.wait_for(
                self.csc.monitor_start_event.wait(), timeout=STD_TIMEOUT
            )

            # Close and re-open the remotes.
            await self.mtdome.close()
            await self.hvac.close()
            await self.ess.close()

            # Close ESS:112, being careful to stop telemetry first.
            assert self.ess112 is not None
            ess112 = self.ess112
            self.ess112 = None
            await ess112.close()

            await asyncio.sleep(LONG_SLEEP)

            self.mtdome = salobj.Controller("MTDome")
            self.hvac = salobj.Controller("HVAC")
            self.ess = salobj.Controller("ESS", 301)
            ess112 = salobj.Controller("ESS", 112)

            self.hvac.cmd_enableDevice.callback = self.enable_callback
            self.hvac.cmd_disableDevice.callback = self.disable_callback

            await asyncio.wait_for(
                asyncio.gather(
                    self.mtdome.start_task,
                    self.hvac.start_task,
                    self.ess.start_task,
                    ess112.start_task,
                ),
                timeout=STD_TIMEOUT,
            )
            self.ess112 = ess112

            await self.load_wind_history("air_flow.ecsv")
            await self.mtdome.tel_apertureShutter.set_write(
                positionActual=(0.0, 0.0),
            )

            await self.wait_for_all_hvac_events()

            await self.csc.close_tasks()

        self.assertAlmostEqual(
            self.csc.average_windspeed, self.wind_data["speed"].mean(), delta=0.1
        )
        self.assertEqual(self.ahu1_state, True)
        self.assertEqual(self.ahu2_state, True)
        self.assertEqual(self.ahu3_state, True)
        self.assertEqual(self.ahu4_state, True)
        self.assertEqual(self.vec04_state, False)
