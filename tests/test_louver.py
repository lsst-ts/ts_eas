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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import asyncio
import logging
import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from lsst.ts import salobj, utils
from lsst.ts.eas.cmdwrapper import close_command_tasks
from lsst.ts.eas.louver_model import (
    DO_NOT_MOVE,
    DORMANT_TIME,
    FULLY_CLOSED,
    FULLY_OPEN,
    LouverControlMode,
    LouverModel,
)

try:
    from lsst.ts.xml.tables.mtdome import LouverTable, find_louver
except ImportError:
    # TODO: OSW-2359 Remove this backward compatibility once the louver table
    # is available in the ts_xml conda package.
    from lsst.ts.eas.louver_table import LouverTable, find_louver

STD_SLEEP = 0.2


class FakeEvtSummaryState:
    def __init__(self) -> None:
        self._msg: SimpleNamespace | None = None

    def set_state(self, state: salobj.State | None) -> None:
        self._msg = None if state is None else SimpleNamespace(summaryState=state)

    def get(self) -> SimpleNamespace | None:
        return self._msg


class FakeDomeLouverCommand:
    """Records setLouvers commands, checking the standing invariants on each.

    Every command EAS sends is verified here rather than in individual tests,
    so a violation is caught wherever it originates -- `adjust_louvers`,
    `exit_mode`, the nighttime control law, or any future caller.

    Attributes
    ----------
    observer_command : `list` [`float`] | None
        The observer's standing command, as declared by the test through
        `TestLouverModel.declare_observer_command`. While None the
        observer-intent invariant is not checked, because the test has not
        said what the observer asked for.
    dome_model : `FakeDomeModel` | None
        If set, each command is echoed into its louver telemetry's
        ``positionCommanded``, as MTDome does, leaving a louver commanded
        `DO_NOT_MOVE` as it was.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.topic_info = SimpleNamespace(attr_name="cmd_setLouvers")
        self.observer_command: list[float] | None = None
        self.dome_model: FakeDomeModel | None = None

    async def set_start(self, **kwargs: Any) -> None:
        filtered = {k: v for k, v in kwargs.items() if k != "timeout"}
        position = filtered.get("position")
        if position is not None:
            self.check_command_range(position)
            self.check_observer_intent(position)
            self.echo(position)
        self.calls.append(filtered)

    def echo(self, position: list[float]) -> None:
        """Report ``position`` as commanded in the dome model's telemetry."""
        if self.dome_model is None or self.dome_model.louvers_telemetry is None:
            return
        telemetry = self.dome_model.louvers_telemetry
        telemetry.positionCommanded = [
            old if new == DO_NOT_MOVE else new for old, new in zip(telemetry.positionCommanded, position)
        ]

    def check_command_range(self, position: list[float]) -> None:
        """Every commanded position is DO_NOT_MOVE or within 0-100 percent."""
        for index, value in enumerate(position):
            if value == DO_NOT_MOVE:
                continue
            if not FULLY_CLOSED <= value <= FULLY_OPEN:
                raise AssertionError(
                    f"Louver {index} commanded to {value}, which is outside "
                    f"{FULLY_CLOSED}-{FULLY_OPEN} and is not the "
                    f"{DO_NOT_MOVE} do-not-move sentinel."
                )

    def check_observer_intent(self, position: list[float]) -> None:
        """EAS never opens a louver the observer did not open.

        A louver the observer never commanded (-1) is left alone entirely, and
        one the observer commanded shut (0) may be re-commanded shut but never
        opened. Note that this constrains only louvers the observer left shut:
        nighttime control is allowed to open a louver past the position the
        observer set, which is what the control law is for.
        """
        if self.observer_command is None:
            return

        for index, value in enumerate(position):
            observer_value = self.observer_command[index]
            if observer_value > FULLY_CLOSED:
                continue
            if value > FULLY_CLOSED:
                raise AssertionError(
                    f"Louver {index} commanded to {value} but the observer "
                    f"commanded it to {observer_value}: EAS must not open a "
                    "louver the observer did not open."
                )
            if observer_value < FULLY_CLOSED and value != DO_NOT_MOVE:
                raise AssertionError(
                    f"Louver {index} commanded to {value} but the observer "
                    "never commanded it: EAS must leave it alone."
                )


class FakeDomeRemote:
    def __init__(self) -> None:
        self.evt_summaryState = FakeEvtSummaryState()
        self.salinfo = SimpleNamespace(name="MTDome", index=None)
        self.cmd_setLouvers = FakeDomeLouverCommand()


class FakeDomeModel:
    """Stand-in for DomeModel, which owns MTDome louver telemetry."""

    def __init__(self) -> None:
        self.louvers_telemetry: SimpleNamespace | None = None
        self.louvers_open_time: float | None = None


class FakeWeatherModel:
    """Stand-in for WeatherModel, which owns the ESS wind measurements."""

    def __init__(self) -> None:
        self.wind_direction: float = math.nan
        self.indoor_windspeed: float = math.nan

    def average_wind_direction(self, window: float) -> float:
        return self.wind_direction

    def average_indoor_windspeed(self, window: float) -> float:
        return self.indoor_windspeed


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
        self.weather_model = FakeWeatherModel()
        self.fake_remote.cmd_setLouvers.dome_model = self.dome_model
        self.model = LouverModel(
            log=logging.getLogger(),
            dome_model=self.dome_model,
            weather_model=self.weather_model,
            dome_remote=self.fake_remote,
            louver_sun_angle=60.0,
            louver_exposed_command=50.0,
            sun_altitude_threshold=0.0,
            nighttime_vent_delay=1800.0,
            inside_windspeed_threshold=2.0,
            inside_windspeed_deadband=0.2,
            upwind_interval=10.0,
            downwind_interval=25.0,
            wind_interval=60.0,
            anemometer_interval=15.0,
            nighttime_adjustment_interval=15.0,
            features_to_disable=[],
        )

    async def asyncTearDown(self) -> None:
        await close_command_tasks(self.model)

    def declare_observer_command(self, command: list[float]) -> None:
        """Set the observer's standing louver command.

        Publishes it as MTDome ``positionCommanded`` telemetry and tells the
        command harness about it, so that every command the model sends is
        checked against what the observer actually asked for.

        Parameters
        ----------
        command : `list` [`float`]
            Observer-commanded position per louver.
        """
        self.dome_model.louvers_telemetry = SimpleNamespace(positionCommanded=list(command))
        self.fake_remote.cmd_setLouvers.observer_command = list(command)

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

    async def test_nighttime_mode_after_vent_delay(self) -> None:
        """Sun down, louvers open and vented long enough selects night control.

        Nighttime control only starts once the louvers have been open for
        ``nighttime_vent_delay`` seconds, so that the dome has had a chance
        to vent before EAS begins balancing the inside windspeed.
        """
        self.dome_model.louvers_open_time = utils.current_tai() - 1801.0

        self.assertIs(
            self.model.select_mode(-10.0),
            LouverControlMode.NIGHTTIME_LOUVER_CONTROL,
        )

    async def test_louvers_grouped_by_relative_wind_direction(self) -> None:
        """Louvers within 90 degrees of the wind direction are upwind.

        With the dome at azimuth 0, each louver's absolute azimuth is just its
        dome-local offset. For wind out of the north, the A/B panels (53.1 and
        67.5 degrees) and the M/N panels (292.5 and 306.9) fall inside the 90
        degree window, and everything from C round to L is downwind.
        """
        self.model.dome_azimuth = 0.0

        upwind = self.model.is_upwind(wind_direction=0.0)

        self.assertEqual(
            [index for index, is_up in enumerate(upwind) if is_up],
            [0, 1, 2, 3, 4, 29, 30, 31, 32, 33],
        )

    async def test_deadband_holds_the_louvers_where_they_are(self) -> None:
        """Inside the deadband the targets do not move at all.

        Crucially they are held where the control law has put them, not
        returned to the observer's command. Returning would undo the very
        adjustment that brought the windspeed into the deadband, which takes
        it straight back out again -- a state with no fixed point, so the
        louvers would oscillate forever in steady wind.
        """
        observer_command = [40.0, 60.0] + [10.0] * 32
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, False] + [False] * 32

        # One step away from the observer's command, then back into the band.
        stepped = self.model.louver_targets(inside_windspeed=1.0, upwind=upwind)
        self.model.expected_commanded = stepped
        self.assertEqual(stepped[1], 85.0, "the downwind louver stepped open")

        held = self.model.louver_targets(inside_windspeed=2.1, upwind=upwind)

        self.assertEqual(held, stepped, "the deadband must hold, not revert")
        self.assertNotEqual(
            held[1],
            observer_command[1],
            "holding means staying stepped, not returning to the observer's command",
        )

    async def test_below_threshold_trims_downwind_louvers_open(self) -> None:
        """Too still inside: each downwind louver opens one interval.

        The trim is measured from the observer's own command for that louver,
        so relative intent survives, and the upwind louvers are untouched
        while the downwind group still has travel.
        """
        observer_command = [40.0, 60.0, 80.0] + [10.0] * 31
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, False, False] + [False] * 31

        targets = self.model.louver_targets(inside_windspeed=1.0, upwind=upwind)

        self.assertEqual(targets[:3], [40.0, 85.0, 100.0])

    async def test_below_threshold_trims_upwind_only_when_downwind_cannot_open(self) -> None:
        """The upwind louvers open only when the downwind ones cannot.

        The two are alternatives, not a pair: open the downwind louvers if
        that is possible, and only if it is not, open the upwind ones. With
        every downwind louver already fully open there is nowhere left to go,
        so the upwind group takes the increment and the downwind group holds.
        """
        observer_command = [40.0] + [FULLY_OPEN] * 33
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True] + [False] * 33

        targets = self.model.louver_targets(inside_windspeed=1.0, upwind=upwind)

        self.assertEqual(targets[0], 50.0, "upwind opens by upwind_interval")
        self.assertTrue(all(target == FULLY_OPEN for target in targets[1:]), "downwind unchanged")

    async def test_above_threshold_trims_upwind_louvers_shut(self) -> None:
        """Too windy inside: each upwind louver closes one interval.

        Closing starts with the louvers facing into the wind, since those are
        what is driving the inside windspeed up. A louver with less than one
        interval of travel simply clamps shut, and the downwind group is left
        at the observer's command.
        """
        observer_command = [40.0, 5.0, 60.0] + [10.0] * 31
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, True, False] + [False] * 31

        targets = self.model.louver_targets(inside_windspeed=3.0, upwind=upwind)

        self.assertEqual(targets[:3], [30.0, 0.0, 60.0])

    async def test_above_threshold_trims_downwind_only_when_upwind_cannot_close(self) -> None:
        """The downwind louvers close only when the upwind ones cannot.

        A louver the observer left shut is already at 0% and takes no part, so
        with nothing open on the upwind side there is nothing to close there
        and the downwind group takes the decrement instead.
        """
        observer_command = [FULLY_CLOSED, FULLY_CLOSED, 60.0] + [FULLY_CLOSED] * 31
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, True, False] + [False] * 31

        targets = self.model.louver_targets(inside_windspeed=3.0, upwind=upwind)

        self.assertEqual(targets[:3], [FULLY_CLOSED, FULLY_CLOSED, 35.0])

    async def test_entering_nighttime_seeds_louver_targets(self) -> None:
        """Each louver's target starts at the position the observer commanded.

        Handing control over must not move anything by itself, and the
        observer's differing positions have to survive the handover.
        """
        observer_command = [0.0, 40.0, 80.0] + [0.0] * 31

        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)

        self.assertEqual(self.model.expected_commanded[:3], [0.0, 40.0, 80.0])

    async def test_unopened_louvers_never_join_nighttime_control(self) -> None:
        """Louvers the observer left shut stay shut, and never go negative.

        ``positionCommanded`` reports -1 for a louver under no command and 0
        for one commanded shut. Neither may be trimmed: doing so would open a
        louver the observer never opened, and trimming -1 upward would put a
        nonsensical value into the target. Only louvers the observer opened
        take part.
        """
        observer_command = [-1.0, 0.0, 40.0] + [0.0] * 31
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [False] * len(self.model.expected_commanded)

        targets = self.model.louver_targets(inside_windspeed=1.0, upwind=upwind)

        self.assertEqual(targets[:3], [-1.0, 0.0, 65.0])

    async def test_entering_nighttime_without_daytime_history_uses_telemetry(self) -> None:
        """A restart after sundown takes the baseline from telemetry.

        The preferred baseline is the one captured during daytime control, but
        an EAS that started up after sundown never ran it, so it has to fall
        back to what MTDome reports as commanded.
        """
        self.declare_observer_command([40.0] * 34)
        self.assertIsNone(self.model.louver_observer_command)

        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)

        self.assertEqual(self.model.expected_commanded, [40.0] * 34)

    async def test_update_louvers_commands_nighttime_targets(self) -> None:
        """A nighttime pass groups by wind, steps, and commands the result.

        With the dome at azimuth 0 and wind out of the north, the A/B and M/N
        panels are upwind and everything from C round to L is downwind. The
        inside windspeed is below the threshold, so the downwind louvers open
        by their 25 percent interval while the upwind ones hold station.
        """
        self.declare_observer_command([40.0] * 34)
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_open_time = utils.current_tai() - 1801.0
        self.weather_model.wind_direction = 0.0
        self.weather_model.indoor_windspeed = 1.0
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        self.assertIs(self.model.mode, LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(position[0], DO_NOT_MOVE, "A1 is upwind and is not being moved")
        self.assertEqual(position[5], 65.0, "C1 is downwind and should open")

    async def test_cadence_follows_the_active_mode(self) -> None:
        """Nighttime runs on its adjustment interval, daytime on DORMANT_TIME.

        Nighttime control steps by a fixed interval per pass, so its cadence
        sets how fast the louvers converge; the daytime pass only tracks the
        sun and can be far lazier.
        """
        self.model.mode = LouverControlMode.NIGHTTIME_LOUVER_CONTROL
        self.assertEqual(self.model.cadence, 15.0)

        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        self.assertEqual(self.model.cadence, DORMANT_TIME)

    async def test_nighttime_commands_only_when_something_moves(self) -> None:
        """A pass that changes nothing says nothing to MTDome.

        The deadband is the ordinary case: repeating passes would otherwise
        spam MTDome every adjustment interval with positions it is already
        holding. Stepping must resume the moment the windspeed leaves the
        deadband again, so silence in the band is not the loop giving up.
        """
        self.declare_observer_command([40.0] * 34)
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_open_time = utils.current_tai() - 1801.0
        self.weather_model.wind_direction = 0.0
        self.weather_model.indoor_windspeed = 1.0  # below the deadband: step
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
            await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
            after_first = len(self.fake_remote.cmd_setLouvers.calls)

            # Into the deadband: the louvers hold, so nothing is sent.
            self.weather_model.indoor_windspeed = 2.0
            await self.model.update_louvers()
            await asyncio.sleep(STD_SLEEP)
            self.assertEqual(
                len(self.fake_remote.cmd_setLouvers.calls),
                after_first,
                "a pass that moves nothing should not be sent",
            )

            # Back out of the deadband: stepping resumes from where it stopped.
            self.weather_model.indoor_windspeed = 1.0
            await self.model.update_louvers()
            await spin_until(lambda: len(self.fake_remote.cmd_setLouvers.calls) > after_first)
            downwind = find_louver("F2").index
            self.assertEqual(
                self.fake_remote.cmd_setLouvers.calls[-1]["position"][downwind],
                90.0,
                "the second step accumulates on the first rather than repeating it",
            )

    async def test_nighttime_without_a_baseline_commands_nothing(self) -> None:
        """With no observer baseline at all, the pass is skipped cleanly.

        `enter_mode` seeds from the daytime baseline and falls back to
        telemetry, but if neither is available the seed is empty. The control
        law must then skip the pass rather than index into an empty target
        list, which would otherwise raise inside the monitor loop and leave
        louver control silently doing nothing on every cycle.
        """
        self.model.mode = LouverControlMode.NIGHTTIME_LOUVER_CONTROL
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        self.model.dome_azimuth = 0.0
        self.weather_model.wind_direction = 0.0
        self.weather_model.indoor_windspeed = 1.0
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.balance_louvers()
        await asyncio.sleep(STD_SLEEP)

        self.assertEqual(self.fake_remote.cmd_setLouvers.calls, [])

    async def test_below_threshold_moves_only_the_downwind_group(self) -> None:
        """While the downwind louvers can still open, the upwind ones do not.

        The two clauses are alternatives: open the downwind louvers if that is
        possible, and only if it is not, open the upwind ones. Here F2 sits at
        80 and has room to go further, so it takes the increment on its own
        and A1 stays exactly where the observer put it.
        """
        a1 = find_louver("A1").index
        f2 = find_louver("F2").index
        commands = [FULLY_CLOSED] * len(LouverTable)
        commands[a1] = 50.0
        commands[f2] = 80.0
        self.model.dome_azimuth = 0.0
        self.model.louver_observer_command = commands
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)

        upwind = self.model.is_upwind(wind_direction=0.0)
        self.assertTrue(upwind[a1], "A1 faces the wind")
        self.assertFalse(upwind[f2], "F2 faces away from it")

        targets = self.model.louver_targets(inside_windspeed=1.0, upwind=upwind)

        self.assertEqual(targets[f2], FULLY_OPEN, "downwind opens by its interval, clamped")
        self.assertEqual(targets[a1], 50.0, "upwind must not move while downwind still can")

    def step(self, inside_windspeed: float, upwind: list[bool]) -> list[float]:
        """Run one control pass and commit its targets.

        Parameters
        ----------
        inside_windspeed : `float`
            Windspeed (m/s) inside the dome for this pass.
        upwind : `list` [`bool`]
            Per-louver flags from `is_upwind`.

        Returns
        -------
        `list` [`float`]
            The targets after this pass.
        """
        targets = self.model.louver_targets(inside_windspeed=inside_windspeed, upwind=upwind)
        self.model.expected_commanded = targets
        return targets

    async def test_successive_passes_accumulate(self) -> None:
        """The targets step further open on every pass, not just the first.

        A single interval of travel is not a control law: while the inside
        windspeed stays below the threshold the louvers must keep opening
        until they reach it or run out of travel.
        """
        observer_command = [40.0, 25.0] + [0.0] * 32
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, False] + [False] * 32

        walk = [self.step(1.0, upwind)[1] for _ in range(4)]

        self.assertEqual(walk, [50.0, 75.0, 100.0, 100.0], "25 percent per pass, clamped at 100")

    async def test_saturated_downwind_group_hands_over_to_upwind(self) -> None:
        """Once the downwind louvers are fully open the upwind ones take over.

        The handover reads the *current* targets, so it happens when the
        downwind group has been stepped to the end of its travel -- not only
        when the observer happened to command it there to begin with.
        """
        observer_command = [40.0, 75.0] + [0.0] * 32
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, False] + [False] * 32

        self.step(1.0, upwind)  # downwind 75 -> 100, upwind holds
        self.assertEqual(self.model.expected_commanded[:2], [40.0, 100.0])

        targets = self.step(1.0, upwind)

        self.assertEqual(targets[0], 50.0, "upwind now takes the increment")
        self.assertEqual(targets[1], 100.0, "downwind is already fully open")

    async def test_upwind_stepped_shut_hands_over_to_downwind(self) -> None:
        """Once the upwind louvers are shut the downwind ones start closing.

        This branch is only reachable because the targets accumulate: a louver
        the observer opened has to be stepped down to zero before the downwind
        group is asked to close, and a single pass could never get it there.
        """
        observer_command = [20.0, 60.0] + [0.0] * 32
        self.model.louver_observer_command = observer_command
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        upwind = [True, False] + [False] * 32

        self.assertEqual(self.step(3.0, upwind)[0], 10.0, "upwind closes by its interval")
        self.assertEqual(self.step(3.0, upwind)[0], 0.0, "and again, down to shut")

        targets = self.step(3.0, upwind)

        self.assertEqual(targets[0], 0.0, "upwind has nothing left to close")
        self.assertEqual(targets[1], 35.0, "so the downwind group closes instead")

    def start_nighttime_control(self, observer_command: list[float]) -> None:
        """Enter nighttime control from telemetry, with the dome at azimuth 0
        and wind out of the north, so C1, D1 and F1 are all downwind.

        Parameters
        ----------
        observer_command : `list` [`float`]
            Observer-commanded position per louver.
        """
        self.declare_observer_command(observer_command)
        self.model.mode = LouverControlMode.NIGHTTIME_LOUVER_CONTROL
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        self.model.dome_azimuth = 0.0
        self.weather_model.wind_direction = 0.0
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

    async def balance_once(self) -> list[float]:
        """Run one nighttime pass and return the position it sent.

        Returns
        -------
        `list` [`float`]
            The position sent on this pass.
        """
        calls = self.fake_remote.cmd_setLouvers.calls
        sent_before = len(calls)
        await self.model.balance_louvers()
        await spin_until(lambda: len(calls) > sent_before)
        return calls[-1]["position"]

    async def test_nighttime_adopts_observer_commands(self) -> None:
        """The observer keeps control of the louvers at night.

        A louver the observer moves is stepped from its new position, one
        they shut is not reopened, and one they open joins nighttime control.
        """
        c1, d1, f1 = (find_louver(name).index for name in ("C1", "D1", "F1"))
        command = [DO_NOT_MOVE] * 34
        command[c1] = command[f1] = 40.0
        self.start_nighttime_control(command)
        self.weather_model.indoor_windspeed = 1.0  # below threshold: downwind opens

        command[c1], command[d1], command[f1] = 70.0, 30.0, FULLY_CLOSED
        self.declare_observer_command(command)
        position = await self.balance_once()

        self.assertEqual(position[c1], 95.0, "moved C1 is stepped from the observer's 70")
        self.assertEqual(position[d1], 55.0, "newly opened D1 joins nighttime control")
        self.assertEqual(position[f1], DO_NOT_MOVE, "shut F1 is left alone")

    async def test_louver_eas_walks_shut_stays_under_control(self) -> None:
        """A louver EAS itself stepped shut is not mistaken for an observer's.

        The observer shutting a louver takes it out of nighttime control, but
        EAS's own command echoed back as 0 must not, or the louver could never
        reopen when the wind drops.
        """
        c1 = find_louver("C1").index
        command = [DO_NOT_MOVE] * 34
        command[c1] = 40.0
        self.start_nighttime_control(command)

        self.weather_model.indoor_windspeed = 3.0  # too windy: close
        self.assertEqual((await self.balance_once())[c1], 15.0)
        self.assertEqual((await self.balance_once())[c1], FULLY_CLOSED)

        self.weather_model.indoor_windspeed = 1.0  # too still: open again
        self.assertEqual((await self.balance_once())[c1], 25.0)

    async def test_no_command_once_the_louvers_run_out_of_travel(self) -> None:
        """Fully open and still too still is a normal state, not a fault.

        On a calm night no louver position produces the target windspeed. The
        loop opens everything, then falls silent rather than re-commanding
        positions it has already reached: nothing changed, so nothing is said.
        """
        self.declare_observer_command([FULLY_OPEN] * 34)
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_open_time = utils.current_tai() - 1801.0
        self.weather_model.wind_direction = 0.0
        self.weather_model.indoor_windspeed = 0.0  # dead calm inside
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)
        self.model.mode = LouverControlMode.NIGHTTIME_LOUVER_CONTROL
        self.model.enter_mode(LouverControlMode.NIGHTTIME_LOUVER_CONTROL)

        await self.model.balance_louvers()
        await asyncio.sleep(STD_SLEEP)

        self.assertEqual(self.fake_remote.cmd_setLouvers.calls, [])

    # ----- record_command -----

    def test_record_command_keeps_previous_value_for_do_not_move(self) -> None:
        """MTDome keeps a louver's positionCommanded when sent DO_NOT_MOVE."""
        self.model.expected_commanded = [0.0, 80.0] + [40.0] * 32

        self.model.record_command([DO_NOT_MOVE, 50.0] + [DO_NOT_MOVE] * 32)

        self.assertEqual(self.model.expected_commanded, [0.0, 50.0] + [40.0] * 32)

    # ----- azimuth callback -----

    async def test_azimuth_callback_updates_dome_azimuth(self) -> None:
        """azimuth_callback should update dome_azimuth from positionActual."""
        self.assertIsNone(self.model.dome_azimuth)

        await self.model.azimuth_callback(SimpleNamespace(positionActual=123.45))

        self.assertAlmostEqual(self.model.dome_azimuth, 123.45)

    async def test_azimuth_callback_requests_update(self) -> None:
        """azimuth_callback wakes daytime control but never commands.

        The first reading and any change greater than
        `AZIMUTH_CHANGE_THRESHOLD` request an immediate pass; a sub-threshold
        change does not. The callback itself must not send a command, which is
        what keeps the control loop the single source of setpoints.
        """
        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.0))
        self.assertTrue(self.model.update_requested.is_set())
        self.assertEqual(self.fake_remote.cmd_setLouvers.calls, [])

        self.model.update_requested.clear()
        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.05))
        self.assertFalse(self.model.update_requested.is_set())

        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.5))
        self.assertTrue(self.model.update_requested.is_set())

    async def test_azimuth_callback_does_not_wake_nighttime_control(self) -> None:
        """A moving dome does not make nighttime control step faster.

        Each nighttime pass takes a step, so waking it on every azimuth sample
        would walk the louvers at the telemetry rate. The new azimuth is still
        recorded for the next pass to group the louvers by.
        """
        self.model.mode = LouverControlMode.NIGHTTIME_LOUVER_CONTROL
        await self.model.azimuth_callback(SimpleNamespace(positionActual=10.0))
        await self.model.azimuth_callback(SimpleNamespace(positionActual=20.0))

        self.assertEqual(self.model.dome_azimuth, 20.0)
        self.assertFalse(self.model.update_requested.is_set())

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
        self.declare_observer_command([-1.0] + [100.0] * 33)
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
        self.declare_observer_command([100.0] * 34)
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
        self.declare_observer_command([10.0] * 34)
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
        self.declare_observer_command([80.0] * 34)
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
        self.declare_observer_command([0.0] * 34)
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
        self.declare_observer_command([80.0] * 34)
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
        cycle1 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(cycle1[f1_index], 50.0)

        # Echo EAS's own command back as the new telemetry, exactly as MTDome
        # would report it.
        self.declare_observer_command(list(cycle1))

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
        self.declare_observer_command([80.0] * 34)
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
        cycle1 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]

        # The observer now commands F1 to 30, which differs from EAS's last
        # command of 50 and so is taken as a fresh observer request.
        echoed = list(cycle1)
        echoed[f1_index] = 30.0
        self.declare_observer_command(echoed)

        # Cycle 2: F1 still faces the sun; min(30, 50) == 30 confirms the
        # baseline was updated to 30.
        await self.model.adjust_louvers(sun_az)
        await spin_until(lambda: len(self.fake_remote.cmd_setLouvers.calls) >= 2)

        cycle2 = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(cycle2[f1_index], 30.0)

    # ----- leaving daytime control -----

    async def test_leaving_daytime_keeps_the_observer_baseline(self) -> None:
        """The observer baseline outlives daytime control.

        After the restore, EAS expects MTDome to report the observer's own
        command. The transition depends only on sun altitude, not on dome
        state.
        """
        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        self.model.louver_observer_command = [80.0] * 34
        self.model.expected_commanded = [50.0] * 34

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()

        self.assertIs(self.model.mode, LouverControlMode.IDLE)
        self.assertEqual(self.model.louver_observer_command, [80.0] * 34)
        self.assertEqual(self.model.expected_commanded, [80.0] * 34)

    async def test_leaving_daytime_opens_to_observer_command(self) -> None:
        """At sundown each opened louver is restored to the observer command.

        A sun-facing louver may have been capped below the observer's command
        during the day. Since the daytime logic does not run again until
        sunrise, the observer baseline is commanded one last time at sundown so
        louvers the observer opened are not left capped overnight. Louvers the
        observer did not open (<= 0) stay uncommanded (-1).
        """
        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        self.model.louver_observer_command = [-1.0] + [100.0] * 33
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))

        position = self.fake_remote.cmd_setLouvers.calls[-1]["position"]
        self.assertEqual(position[0], -1.0)
        self.assertTrue(all(p == 100.0 for p in position[1:]))

    def set_up_night_walk(self) -> tuple[int, int]:
        """Run nighttime control for one step, from an observer command.

        The observer opened A1 (upwind) to 80 and C1 (downwind) to 40. The
        inside wind is below the threshold, so nighttime control steps C1 to
        65, which MTDome then reports as commanded.

        Returns
        -------
        `tuple` [`int`, `int`]
            Indices of A1 and C1.
        """
        a1, c1 = (find_louver(name).index for name in ("A1", "C1"))
        command = [DO_NOT_MOVE] * 34
        command[a1], command[c1] = 80.0, 40.0
        self.start_nighttime_control(command)
        self.weather_model.indoor_windspeed = 1.0
        return a1, c1

    async def test_sunrise_returns_to_observer_command(self) -> None:
        """Night to day mirrors day to night.

        Daytime control applies its rules to the observer's standing command,
        not to where the nighttime walk left the louvers: C1 goes back from 65
        to the observer's 40, and A1, facing the sun, is capped at 50.
        """
        a1, c1 = self.set_up_night_walk()
        self.assertEqual((await self.balance_once())[c1], 65.0)

        calls = self.fake_remote.cmd_setLouvers.calls
        sent_before = len(calls)
        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(10.0, azimuth=0.0)):
            await self.model.update_louvers()
        await spin_until(lambda: len(calls) > sent_before)

        self.assertIs(self.model.mode, LouverControlMode.DAYTIME_LOUVER_CONTROL)
        position = calls[-1]["position"]
        self.assertEqual(position[c1], 40.0, "C1 returns to the observer's command")
        self.assertEqual(position[a1], 50.0, "sun-facing A1 is capped")

    async def test_observer_commands_between_night_and_day_are_kept(self) -> None:
        """An observer command while EAS is idle becomes the standing command.

        Nighttime control drops to IDLE when the louvers stop venting. The
        observer then moves C1 to 30, and daytime control takes 30, not the
        nighttime target or the earlier command, as the observer's request.
        """
        a1, c1 = self.set_up_night_walk()
        await self.balance_once()

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
        self.assertIs(self.model.mode, LouverControlMode.IDLE)

        assert self.dome_model.louvers_telemetry is not None
        command = list(self.dome_model.louvers_telemetry.positionCommanded)
        command[c1] = 30.0
        self.declare_observer_command(command)

        calls = self.fake_remote.cmd_setLouvers.calls
        sent_before = len(calls)
        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(10.0, azimuth=0.0)):
            await self.model.update_louvers()
        await spin_until(lambda: len(calls) > sent_before)

        self.assertEqual(calls[-1]["position"][c1], 30.0)
        self.assertEqual(self.model.louver_observer_command[c1], 30.0)

    def set_up_direct_day_to_night(self) -> tuple[int, int, int, int]:
        """Put the model in daytime control just before a direct handover.

        The observer opened every louver to 80. Daytime control capped A1
        (upwind) and C1 (downwind) at 50, so ``positionCommanded`` telemetry
        shows 50 for them. The louvers have been open longer than the vent
        delay, so the next pass after sundown goes straight to nighttime
        control, with the dome at azimuth 0 and wind out of the north.

        Returns
        -------
        `tuple` [`int`, `int`, `int`, `int`]
            Indices of A1 (capped, upwind), B1 (uncapped, upwind), C1 (capped,
            downwind) and F1 (uncapped, downwind).
        """
        a1, b1, c1, f1 = (find_louver(name).index for name in ("A1", "B1", "C1", "F1"))
        eas_command = [80.0] * 34
        eas_command[a1] = eas_command[c1] = 50.0

        self.dome_model.louvers_telemetry = SimpleNamespace(positionCommanded=list(eas_command))
        self.fake_remote.cmd_setLouvers.observer_command = [80.0] * 34
        self.model.mode = LouverControlMode.DAYTIME_LOUVER_CONTROL
        self.model.louver_observer_command = [80.0] * 34
        self.model.expected_commanded = eas_command
        self.model.dome_azimuth = 0.0
        self.dome_model.louvers_open_time = utils.current_tai() - 1801.0
        self.weather_model.wind_direction = 0.0
        self.fake_remote.evt_summaryState.set_state(salobj.State.ENABLED)
        return a1, b1, c1, f1

    async def test_direct_day_to_night_starts_from_observer_command(self) -> None:
        """A direct handover steps from the observer's command, not the cap.

        Telemetry still shows the daytime caps at sundown, so nighttime control
        must seed from the cached observer command. Its first command must
        also carry the restore of the capped louvers it does not step, since a
        separate restore command would be superseded before it was sent.
        """
        a1, b1, c1, f1 = self.set_up_direct_day_to_night()
        self.weather_model.indoor_windspeed = 1.0  # below threshold: downwind opens

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
        await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
        await asyncio.sleep(STD_SLEEP)

        self.assertIs(self.model.mode, LouverControlMode.NIGHTTIME_LOUVER_CONTROL)
        calls = self.fake_remote.cmd_setLouvers.calls
        self.assertEqual(len(calls), 1, "the restore goes out in the first nighttime command")
        position = calls[0]["position"]
        self.assertEqual(position[c1], 100.0, "capped downwind C1 steps from 80, not from 50")
        self.assertEqual(position[f1], 100.0, "uncapped downwind F1 steps from 80")
        self.assertEqual(position[a1], 80.0, "capped upwind A1 is restored to 80")
        self.assertEqual(position[b1], DO_NOT_MOVE, "uncapped upwind B1 is already at 80")
        self.assertEqual(self.model.louver_observer_command, [80.0] * 34)

    async def test_direct_day_to_night_restores_without_wind_data(self) -> None:
        """The restore does not wait for wind measurements and is sent once."""
        a1, b1, c1, f1 = self.set_up_direct_day_to_night()
        self.weather_model.indoor_windspeed = math.nan

        with mock.patch("lsst.ts.eas.louver_model.get_sun", return_value=make_sun_mock(-10.0)):
            await self.model.update_louvers()
            await spin_until(lambda: bool(self.fake_remote.cmd_setLouvers.calls))
            await self.model.update_louvers()
        await asyncio.sleep(STD_SLEEP)

        calls = self.fake_remote.cmd_setLouvers.calls
        self.assertEqual(len(calls), 1, "a pass with nothing to restore and no wind data sends nothing")
        position = calls[0]["position"]
        self.assertEqual([position[i] for i in (a1, b1, c1, f1)], [80.0, DO_NOT_MOVE, 80.0, DO_NOT_MOVE])

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
