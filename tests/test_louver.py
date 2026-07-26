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
from types import SimpleNamespace
from typing import Any
from unittest import mock

from lsst.ts import salobj
from lsst.ts.eas.cmdwrapper import close_command_tasks
from lsst.ts.eas.louver_model import LouverControlMode, LouverModel

try:
    from lsst.ts.xml.tables.mtdome import find_louver
except ImportError:
    # TODO: OSW-2359 Remove this backward compatibility once the louver table
    # is available in the ts_xml conda package.
    from lsst.ts.eas.louver_table import find_louver

STD_SLEEP = 0.2


class FakeEvtSummaryState:
    def __init__(self) -> None:
        self._msg: SimpleNamespace | None = None

    def set_state(self, state: salobj.State | None) -> None:
        self._msg = None if state is None else SimpleNamespace(summaryState=state)

    def get(self) -> SimpleNamespace | None:
        return self._msg


class FakeDomeLouverCommand:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.topic_info = SimpleNamespace(attr_name="cmd_setLouvers")

    async def set_start(self, **kwargs: Any) -> None:
        filtered = {k: v for k, v in kwargs.items() if k != "timeout"}
        self.calls.append(filtered)


class FakeDomeRemote:
    def __init__(self) -> None:
        self.evt_summaryState = FakeEvtSummaryState()
        self.salinfo = SimpleNamespace(name="MTDome", index=None)
        self.cmd_setLouvers = FakeDomeLouverCommand()


class FakeDomeModel:
    """Stand-in for DomeModel, which owns MTDome louver telemetry."""

    def __init__(self) -> None:
        self.louvers_telemetry: SimpleNamespace | None = None


async def spin_until(pred: Any, *, timeout: float = 5.0, step: float = 0.01) -> None:
    """Poll `pred()` until true or raise TimeoutError.

    Parameters
    ----------
    pred : callable
        A predicate to test. When it returns True, the polling stops.
    timeout : `float`
        Maximum time (seconds) to wait before raising `TimeoutError`.
    step : `float`
        Seconds to sleep between successive polls of the predicate.

    Raises
    ------
    `TimeoutError`
        If the predicate is not satisfied before the timeout elapses.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return
        await asyncio.sleep(step)
    raise TimeoutError("Condition not met before timeout")


def make_sun_mock(altitude: float, azimuth: float = 0.0) -> mock.MagicMock:
    """Return a mock sun whose transform_to reports the given altitude/azimuth.

    Parameters
    ----------
    altitude : `float`
        Sun altitude in degrees to report.
    azimuth : `float`
        Sun azimuth in degrees to report.

    Returns
    -------
    `~unittest.mock.MagicMock`
        A stand-in for the value returned by ``get_sun``.
    """
    mock_altaz = mock.MagicMock()
    mock_altaz.alt.deg = altitude
    mock_altaz.az.deg = azimuth
    mock_sun = mock.MagicMock()
    mock_sun.transform_to.return_value = mock_altaz
    return mock_sun


class TestLouverModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.fake_remote = FakeDomeRemote()
        self.dome_model = FakeDomeModel()
        self.model = LouverModel(
            log=logging.getLogger(),
            dome_model=self.dome_model,
            dome_remote=self.fake_remote,
            louver_sun_angle=60.0,
            louver_exposed_command=50.0,
            sun_altitude_threshold=0.0,
            features_to_disable=[],
        )

    async def asyncTearDown(self) -> None:
        await close_command_tasks(self.model)

    async def run_monitor_until(self, pred: Any) -> None:
        """Run the control loop until `pred` holds, then cancel it.

        Parameters
        ----------
        pred : callable
            Predicate polled while the loop runs.
        """
        monitor_task = asyncio.create_task(self.model.monitor())
        try:
            await spin_until(pred)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    # ----- mode selection -----

    async def test_starts_idle(self) -> None:
        """A freshly constructed model controls nothing until it has run."""
        self.assertIs(self.model.mode, LouverControlMode.IDLE)

    async def test_daytime_mode_when_sun_above_threshold(self) -> None:
        """Sun above `sun_altitude_threshold` selects daytime sun avoidance."""
        self.assertIs(
            self.model.select_mode(45.0),
            LouverControlMode.DAYTIME_LOUVER_CONTROL,
        )

    async def test_idle_when_sun_below_threshold(self) -> None:
        """After sunset leaves louvers to the observer."""
        self.assertIs(self.model.select_mode(-10.0), LouverControlMode.IDLE)

    async def test_idle_when_day_louvers_disabled(self) -> None:
        """`day_louvers` in features_to_disable suppresses daytime control."""
        self.model.features_to_disable = ["day_louvers"]

        self.assertIs(self.model.select_mode(45.0), LouverControlMode.IDLE)

    # ----- azimuth callback -----

    async def test_azimuth_callback_updates_dome_azimuth(self) -> None:
        """azimuth_callback should update dome_azimuth from positionActual."""
        self.assertIsNone(self.model.dome_azimuth)

        await self.model.azimuth_callback(SimpleNamespace(positionActual=123.45))

        self.assertAlmostEqual(self.model.dome_azimuth, 123.45)

    async def test_azimuth_callback_requests_update(self) -> None:
        """azimuth_callback wakes the control loop but never commands.

        The first reading and any change greater than
        `AZIMUTH_CHANGE_THRESHOLD` request an immediate pass; a sub-threshold
        change does not. The callback itself must not send a command, which is
        what keeps the control loop the single source of setpoints.
        """
        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.0))
        self.assertTrue(self.model.update_requested.is_set())
        self.assertEqual(self.fake_remote.cmd_setLouvers.calls, [])

        self.model.update_requested.clear()
        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.05))
        self.assertFalse(self.model.update_requested.is_set())

        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.5))
        self.assertTrue(self.model.update_requested.is_set())

    # ----- command gating -----

    async def test_set_louvers_does_not_send_when_not_enabled(self) -> None:
        """set_louvers should not send if the remote has never been ENABLED."""
        self.fake_remote.evt_summaryState.set_state(None)

        position = [50.0] * 34
        await self.model.set_louvers(position=position)
        await asyncio.sleep(STD_SLEEP)

        self.assertEqual(self.fake_remote.cmd_setLouvers.calls, [])

    # ----- daytime sun avoidance -----

    async def test_monitor_calls_adjust_louvers_when_sun_is_up(self) -> None:
        """monitor() adjusts louvers based on sun azimuth.

        With dome at azimuth 0 (slit pointing north) and sun at azimuth 90
        (east), with louver_sun_angle=60 and an observer command of 100:
        - Louver A1 (index 0) is uncommanded (positionCommanded=-1) => -1
        - Louvers A2-E3 (indices 1-13) face the sun (panel normals are
          53.1-120.75 deg, within 60 deg of az 90) => capped at the exposed
          command min(100, 50) => 50.0
        - Louvers F1-N2 (indices 14-33) face away from the sun (panel
          normals 180-306.9 deg, outside the 60 deg window) => left at the
          observer command (shaded louvers are uncapped) => 100.0
        """
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[-1.0] + [100.0] * 33,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        with (
            mock.patch("lsst.ts.eas.louver_model.DORMANT_TIME", 0.01),
            mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(45.0, 90.0)),
        ):
            await self.run_monitor_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        self.assertIs(self.model.mode, LouverControlMode.DAYTIME_LOUVER_CONTROL)
        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(position[0], -1.0)
        self.assertTrue(all(p == 50.0 for p in position[1:14]))
        self.assertTrue(all(p == 100.0 for p in position[14:]))

    async def test_adjust_louvers_dome_opposite_sun_exposes_f1(self) -> None:
        """Geometric sanity check: when the slit points opposite the sun,
        the F panel (dome-local offset 180 deg) faces the sun directly,
        so F1 must be commanded to ``louver_exposed_command``.
        """
        sun_az = 73.0
        self.model.dome_azimuth = (sun_az + 180.0) % 360.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[100.0] * 34,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        f1_index = find_louver("F1").index
        self.assertEqual(
            position[f1_index],
            self.model.louver_exposed_command,
            "F1 should be exposed when the dome slit points opposite the sun",
        )

    async def test_adjust_louvers_below_exposed_limit_never_moves(self) -> None:
        """A louver opened to <= louver_exposed_command is left alone.

        With the observer commanding 10 (below the exposed cap of 50), every
        commanded louver is left at 10 whether it faces the sun or not, because
        ``min(10, 50) == 10`` and shaded louvers are uncapped.
        """
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[10.0] * 34,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(90.0)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertTrue(all(p == 10.0 for p in position))

    async def test_adjust_louvers_splits_at_exposed_limit(self) -> None:
        """Commanded position is limited for exposed louvers.

        With the observer commanding 80 (above the exposed cap of 50),
        sun-facing louvers are capped to 50 while shaded louvers stay at 80.
        """
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[80.0] * 34,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(90.0)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        # With dome az 0 and sun az 90, indices 0-13 face the sun and 14-33 are
        # shaded (see test_monitor_calls_adjust_louvers_when_sun_is_up).
        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertTrue(all(p == 50.0 for p in position[:14]))
        self.assertTrue(all(p == 80.0 for p in position[14:]))

    async def test_adjust_louvers_closed_stays_closed(self) -> None:
        """Louvers the observer has not opened remain uncommanded (-1)."""
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[0.0] * 34,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(90.0)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertTrue(all(p == -1.0 for p in position))

    async def test_adjust_louvers_reopens_after_leaving_sun(self) -> None:
        """A louver re-opens to the observer command after leaving the sun.

        This is the regression guard for the overwrite problem: EAS commands
        through the same path the observer uses, so its own cap is echoed back
        in ``positionCommanded``. The observer baseline must be remembered so a
        louver that was capped while exposed returns to the full observer
        command once it is shaded, rather than staying at the cap.
        """
        sun_az = 73.0
        f1_index = find_louver("F1").index

        # Cycle 1: dome points opposite the sun, so F1 faces the sun and is
        # capped from the observer command of 80 down to 50.
        self.model.dome_azimuth = (sun_az + 180.0) % 360.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[80.0] * 34,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
        cycle1 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(cycle1[f1_index], 50.0)

        # Echo EAS's own command back as the new telemetry, exactly as MTDome
        # would report it.
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=list(cycle1),
        )

        # Cycle 2: dome now points at the sun, so F1 is shaded and must return
        # to the remembered observer command of 80, not stay at the 50 cap.
        self.model.dome_azimuth = sun_az
        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: len(self.fake_remote.cmd_setLouvers.calls) >= 2)

        cycle2 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(cycle2[f1_index], 80.0)

    async def test_adjust_louvers_new_observer_command_overrides_cap(self) -> None:
        """A fresh observer command replaces the remembered baseline.

        After EAS caps an exposed louver, the observer commanding a new
        position (distinct from EAS's last command) must update the baseline.
        """
        sun_az = 73.0
        f1_index = find_louver("F1").index

        # Cycle 1: F1 faces the sun, capped from 80 to 50.
        self.model.dome_azimuth = (sun_az + 180.0) % 360.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=[80.0] * 34,
        )
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
        cycle1 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]

        # The observer now commands F1 to 30, which differs from EAS's last
        # command of 50 and so is taken as a fresh observer request.
        echoed = list(cycle1)
        echoed[f1_index] = 30.0
        self.dome_model.louvers_telemetry = SimpleNamespace(
            positionCommanded=echoed,
        )

        # Cycle 2: F1 still faces the sun; min(30, 50) == 30 confirms the
        # baseline was updated to 30.
        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: len(self.fake_remote.cmd_setLouvers.calls) >= 2)

        cycle2 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(cycle2[f1_index], 30.0)

    # ----- leaving daytime control -----

    async def test_leaving_daytime_resets_baseline(self) -> None:
        """The observer baseline is cleared when daytime control ends.

        The transition depends only on sun altitude, not on dome state.
        """
        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        self.model.louver_operator_command = [80.0] * 34
        self.model.louver_eas_command = [50.0] * 34

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()

        self.assertIs(self.model.mode, LouverControlMode.IDLE)
        self.assertIsNone(self.model.louver_operator_command)
        self.assertIsNone(self.model.louver_eas_command)

    async def test_leaving_daytime_opens_to_operator_command(self) -> None:
        """At sundown each opened louver is restored to the observer command.

        A sun-facing louver may have been capped below the observer's command
        during the day. Since the daytime logic does not run again until
        sunrise, the observer baseline is commanded one last time at sundown so
        louvers the observer opened are not left capped overnight. Louvers the
        observer did not open (<= 0) stay uncommanded (-1).
        """
        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        self.model.louver_operator_command = [-1.0] + [100.0] * 33
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(position[0], -1.0)
        self.assertTrue(all(p == 100.0 for p in position[1:]))

    async def test_monitor_skips_adjust_louvers_when_sun_is_down(self) -> None:
        """monitor() should not call adjust_louvers after sundown."""
        self.model.adjust_louvers = mock.AsyncMock()

        with (
            mock.patch("lsst.ts.eas.louver_model.DORMANT_TIME", 0.01),
            mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)),
        ):
            monitor_task = asyncio.create_task(self.model.monitor())
            await asyncio.sleep(STD_SLEEP)
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        self.model.adjust_louvers.assert_not_called()
