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
import math
import unittest
from collections import deque
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from lsst.ts import salobj
from lsst.ts.eas import CONFIG_SCHEMA, delay_controller, tma_model


class TestWithinTolerance(unittest.TestCase):
    def test_none_and_nonfinite(self) -> None:
        """Inputs should return False when None or non-finite."""
        tolerance_positive = 0.1
        tolerance_negative = 0.1
        current = 1.0
        finite_target = 1.0
        self.assertFalse(
            delay_controller.within_tolerance(
                target=None,
                current=current,
                tolerance_positive=tolerance_positive,
                tolerance_negative=tolerance_negative,
            )
        )
        self.assertFalse(
            delay_controller.within_tolerance(
                target=finite_target,
                current=None,
                tolerance_positive=tolerance_positive,
                tolerance_negative=tolerance_negative,
            )
        )
        self.assertFalse(
            delay_controller.within_tolerance(
                target=math.nan,
                current=current,
                tolerance_positive=tolerance_positive,
                tolerance_negative=tolerance_negative,
            )
        )
        self.assertFalse(
            delay_controller.within_tolerance(
                target=finite_target,
                current=math.inf,
                tolerance_positive=tolerance_positive,
                tolerance_negative=tolerance_negative,
            )
        )

    def test_asymmetric_tolerance(self) -> None:
        """Asymmetric tolerances should be applied correctly."""
        target = 10.0
        tolerance_positive = 0.2
        tolerance_negative = 0.05
        current_within = 10.1
        current_outside = 9.8
        alternate_negative_tolerance = 0.1
        self.assertTrue(
            delay_controller.within_tolerance(
                target=target,
                current=current_within,
                tolerance_positive=tolerance_positive,
                tolerance_negative=tolerance_negative,
            )
        )
        self.assertFalse(
            delay_controller.within_tolerance(
                target=target,
                current=current_outside,
                tolerance_positive=tolerance_positive,
                tolerance_negative=alternate_negative_tolerance,
            )
        )


class TestDelayPolicies(unittest.TestCase):
    def test_time_delay_policy_readiness(self) -> None:
        """Time-delay policy should become ready after the delay."""
        delay_seconds = 10.0
        open_time = 100.0
        now_before = 109.9
        now_after = 110.0
        policy = delay_controller.make_time_delay_policy(delay_seconds=delay_seconds)
        self.assertFalse(policy.is_ready(None, None, open_time, now_before))
        self.assertTrue(policy.is_ready(None, None, open_time, now_after))

    def test_wait_for_convergence_policy(self) -> None:
        """Convergence policy should be ready when within tolerance."""
        tolerance_positive = 0.2
        tolerance_negative = 0.2
        target_setpoint = 10.0
        current_within = 10.1
        current_outside = 10.4
        open_time = 100.0
        now = 200.0
        policy = delay_controller.make_wait_for_convergence_policy(
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
            timeout_seconds=None,
        )
        self.assertTrue(policy.is_ready(target_setpoint, current_within, open_time, now))
        self.assertFalse(policy.is_ready(target_setpoint, current_outside, open_time, now))

    def test_wait_for_convergence_policy_timeout(self) -> None:
        """Convergence policy should honor timeout and no-timeout behavior."""
        target_setpoint = 10.0  # doesn't matter for this test
        setpoint = 10.4  # doesn't matter for this test
        tolerance = 0.2
        start_time = 100.0
        timeout = 10.0
        far_future = start_time + timeout * 1e6
        halfway_time = start_time + timeout / 2
        boundary_time = start_time + timeout
        beyond_time = start_time + timeout * 2

        policy = delay_controller.make_wait_for_convergence_policy(
            tolerance_positive=tolerance,
            tolerance_negative=tolerance,
            timeout_seconds=timeout,
        )
        self.assertFalse(policy.is_ready(target_setpoint, setpoint, start_time, halfway_time))
        self.assertTrue(policy.is_ready(target_setpoint, setpoint, start_time, boundary_time))
        self.assertTrue(policy.is_ready(target_setpoint, setpoint, start_time, beyond_time))

        no_timeout = delay_controller.make_wait_for_convergence_policy(
            tolerance_positive=tolerance,
            tolerance_negative=tolerance,
            timeout_seconds=None,
        )
        self.assertFalse(no_timeout.is_ready(target_setpoint, setpoint, start_time, far_future))

    def test_make_delay_policy_from_config(self) -> None:
        """Delay policies should construct from configuration dictionaries."""
        time_delay_config = {"mode": "time_delay", "delay": 5.0}
        open_time = 100.0
        now = 105.0
        policy = delay_controller.make_delay_policy_from_config(time_delay_config)
        self.assertTrue(policy.is_ready(None, None, open_time, now))

        target_setpoint = 10.0
        current_within = 10.3
        open_time = 100.0
        now = 101.0
        policy = delay_controller.make_delay_policy_from_config(
            {
                "mode": "wait_for_convergence",
                "tolerance_positive": 0.2,
                "tolerance_negative": 0.2,
                "timeout": 1.0,
            }
        )
        self.assertTrue(policy.is_ready(target_setpoint, current_within, open_time, now))

        target_setpoint = 10.0
        current_outside = 11.0
        open_time = 100.0
        now = 101.0
        policy = delay_controller.make_delay_policy_from_config(
            {
                "mode": "evening_controlled_convergence",
                "tolerance_positive": 0.2,
                "tolerance_negative": 0.2,
                "slew_rate": 2.0,
            }
        )
        self.assertFalse(policy.is_ready(target_setpoint, current_outside, open_time, now))

    def test_make_delay_policy_from_config_invalid(self) -> None:
        """Invalid delay policy configuration should be rejected."""
        with self.assertRaises(ValueError):
            dict_with_missing_mode = {"delay": 5.0}
            delay_controller.make_delay_policy_from_config(dict_with_missing_mode)
        with self.assertRaises(ValueError):
            dict_with_unknown_mode = {"mode": "unknown"}
            delay_controller.make_delay_policy_from_config(dict_with_unknown_mode)


class TestEveningControlledConvergence(unittest.IsolatedAsyncioTestCase):
    async def test_on_hold_slews_toward_target(self) -> None:
        """Controlled convergence should slew setpoints toward the target."""
        tolerance_positive = 0.2
        tolerance_negative = 0.2
        slew_rate = 360.0
        cadence = 10.0
        target = 12.0
        initial_setpoint = 10.0
        near_target_setpoint = 11.5
        expected_after_first = 11.0
        policy = delay_controller.make_evening_controlled_convergence_policy(
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
            slew_rate=slew_rate,
        )
        applied: list[float] = []

        async def apply_setpoints(value: float) -> None:
            applied.append(value)

        await policy.on_hold(
            target_setpoint=target,
            last_m1m3ts_setpoint=initial_setpoint,
            cadence=cadence,
            apply_setpoints_callback=apply_setpoints,
        )
        self.assertAlmostEqual(applied[-1], expected_after_first, places=6)

        await policy.on_hold(
            target_setpoint=target,
            last_m1m3ts_setpoint=near_target_setpoint,
            cadence=cadence,
            apply_setpoints_callback=apply_setpoints,
        )
        self.assertAlmostEqual(applied[-1], target, places=6)

        await policy.on_hold(
            target_setpoint=target,
            last_m1m3ts_setpoint=None,
            cadence=cadence,
            apply_setpoints_callback=apply_setpoints,
        )
        self.assertAlmostEqual(applied[-1], target, places=6)

    async def test_on_hold_requires_callback(self) -> None:
        """Controlled convergence should send an apply callback."""
        tolerance_positive = 0.2
        tolerance_negative = 0.2
        slew_rate = 360.0
        cadence = 10.0
        target = 12.0
        initial_setpoint = 10.0
        policy = delay_controller.make_evening_controlled_convergence_policy(
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
            slew_rate=slew_rate,
        )
        with self.assertRaises(ValueError):
            await policy.on_hold(
                target_setpoint=target,
                last_m1m3ts_setpoint=initial_setpoint,
                cadence=cadence,
                apply_setpoints_callback=None,
            )


class TestDelayControllerConfig(unittest.TestCase):
    def test_delay_mode_configs_validate(self) -> None:
        """Delay mode configs should validate against config schemas."""
        config_dir = Path(__file__).resolve().parent / "config"
        config_files = [
            config_dir / "delay_time.yaml",
            config_dir / "delay_wait_for_convergence.yaml",
            config_dir / "delay_evening_controlled_convergence.yaml",
        ]
        config_validator = salobj.DefaultingValidator(CONFIG_SCHEMA)
        tma_validator = salobj.DefaultingValidator(tma_model.TmaModel.get_config_schema())

        for config_path in config_files:
            with self.subTest(config_path=config_path.name):
                data = yaml.safe_load(config_path.read_text())
                config = config_validator.validate(data)
                tma_validator.validate(config["tma"])


class FakeDomeModel:
    def __init__(self) -> None:
        self.is_closed: bool | None = True
        self.on_open: deque[asyncio.Event] = deque()

    def open(self) -> None:
        """Change dome state to opened."""
        self.is_closed = False
        for event in list(self.on_open):
            event.set()
        self.on_open.clear()


class TestDelayControllerIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_open_transitions_to_waiting(self) -> None:
        """Controller should transition to WAITING after the dome opens."""
        delay_seconds = 1.0
        poll_interval = 0.01
        controller = delay_controller.DelayController(
            policy=delay_controller.make_time_delay_policy(delay_seconds=delay_seconds),
            log=logging.getLogger(),
        )
        dome = FakeDomeModel()

        with mock.patch.object(delay_controller, "POLL_INTERVAL", poll_interval):
            task = asyncio.create_task(controller.wait_for_open(dome))
            await asyncio.sleep(0)
            dome.open()
            await task

        self.assertEqual(controller.state, delay_controller.DelayState.WAITING)
        self.assertIsNotNone(controller.open_time)

    async def test_gate_returns_false_until_ready(self) -> None:
        """Gate should return False until the policy allows tracking."""
        delay_seconds = 10.0
        open_time = 100.0
        now_before = 105.0
        now_after = 110.0
        cadence = 1.0
        controller = delay_controller.DelayController(
            policy=delay_controller.make_time_delay_policy(delay_seconds=delay_seconds),
            log=logging.getLogger(),
        )
        controller.state = delay_controller.DelayState.WAITING
        controller.open_time = open_time

        with mock.patch.object(delay_controller, "current_tai", return_value=now_before):
            ready = await controller.gate(
                target_setpoint=None,
                last_m1m3ts_setpoint=None,
                cadence=cadence,
                apply_setpoints_callback=None,
            )
            self.assertFalse(ready)

        with mock.patch.object(delay_controller, "current_tai", return_value=now_after):
            ready = await controller.gate(
                target_setpoint=None,
                last_m1m3ts_setpoint=None,
                cadence=cadence,
                apply_setpoints_callback=None,
            )
            self.assertTrue(ready)
            self.assertEqual(controller.state, delay_controller.DelayState.READY)

    async def test_gate_calls_on_hold_while_waiting(self) -> None:
        """Gate should invoke on_hold while waiting."""
        target_setpoint = 10.0
        last_m1m3ts_setpoint = 9.0
        cadence = 5.0
        open_time = 100.0
        calls: list[float] = []

        async def on_hold(
            target_setpoint: float | None,
            last_m1m3ts_setpoint: float | None,
            cadence: float,
            apply_setpoints_callback: delay_controller.ApplySetpointsCallback | None,
        ) -> None:
            calls.append(cadence)

        policy = delay_controller.DelayPolicy(is_ready=lambda *_args: False, on_hold=on_hold)
        controller = delay_controller.DelayController(policy=policy, log=logging.getLogger())
        controller.state = delay_controller.DelayState.WAITING
        controller.open_time = open_time

        await controller.gate(
            target_setpoint=target_setpoint,
            last_m1m3ts_setpoint=last_m1m3ts_setpoint,
            cadence=cadence,
            apply_setpoints_callback=None,
        )
        self.assertEqual(calls, [cadence])

    async def test_gate_applies_setpoints_for_controlled_convergence(self) -> None:
        """Controlled convergence should apply slewed setpoints."""
        tolerance_positive = 0.2
        tolerance_negative = 0.2
        slew_rate = 360.0
        target_setpoint = 12.0
        last_m1m3ts_setpoint = 10.0
        cadence = 10.0
        open_time = 100.0
        now = 101.0
        expected_applied = 11.0
        policy = delay_controller.make_evening_controlled_convergence_policy(
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
            slew_rate=slew_rate,
        )
        controller = delay_controller.DelayController(policy=policy, log=logging.getLogger())
        controller.state = delay_controller.DelayState.WAITING
        controller.open_time = open_time

        applied: list[float] = []

        async def apply_setpoints(value: float) -> None:
            applied.append(value)

        with mock.patch.object(delay_controller, "current_tai", return_value=now):
            ready = await controller.gate(
                target_setpoint=target_setpoint,
                last_m1m3ts_setpoint=last_m1m3ts_setpoint,
                cadence=cadence,
                apply_setpoints_callback=apply_setpoints,
            )

        self.assertFalse(ready)
        self.assertAlmostEqual(applied[-1], expected_applied, places=6)

    async def test_gate_requires_callback_for_controlled_convergence(self) -> None:
        """Controlled convergence should require a callback."""
        tolerance_positive = 0.2
        tolerance_negative = 0.2
        slew_rate = 360.0
        target_setpoint = 12.0
        last_m1m3ts_setpoint = 10.0
        cadence = 10.0
        open_time = 100.0
        policy = delay_controller.make_evening_controlled_convergence_policy(
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
            slew_rate=slew_rate,
        )
        controller = delay_controller.DelayController(policy=policy, log=logging.getLogger())
        controller.state = delay_controller.DelayState.WAITING
        controller.open_time = open_time

        with self.assertRaises(ValueError):
            await controller.gate(
                target_setpoint=target_setpoint,
                last_m1m3ts_setpoint=last_m1m3ts_setpoint,
                cadence=cadence,
                apply_setpoints_callback=None,
            )

    def test_reset(self) -> None:
        """Reset should return the controller to IDLE with no open time."""
        delay_seconds = 1.0
        open_time = 100.0
        controller = delay_controller.DelayController(
            policy=delay_controller.make_time_delay_policy(delay_seconds=delay_seconds),
            log=logging.getLogger(),
        )
        controller.state = delay_controller.DelayState.READY
        controller.open_time = open_time
        controller.reset()
        self.assertEqual(controller.state, delay_controller.DelayState.IDLE)
        self.assertIsNone(controller.open_time)

    async def test_gate_with_no_open_time(self) -> None:
        """Gate should skip on_hold when open time is unset."""
        target_setpoint = 10.0
        last_m1m3ts_setpoint = 9.0
        cadence = 1.0
        called = False

        async def on_hold(*_args: Any) -> None:
            nonlocal called
            called = True

        policy = delay_controller.DelayPolicy(is_ready=lambda *_args: False, on_hold=on_hold)
        controller = delay_controller.DelayController(policy=policy, log=logging.getLogger())

        ready = await controller.gate(
            target_setpoint=target_setpoint,
            last_m1m3ts_setpoint=last_m1m3ts_setpoint,
            cadence=cadence,
            apply_setpoints_callback=None,
        )
        self.assertFalse(ready)
        self.assertFalse(called)

    async def test_controlled_convergence_evening_run(self) -> None:
        """Evening run should converge to target over repeated cycles."""
        tolerance_positive = 0.2
        tolerance_negative = 0.2
        slew_rate = 2.0
        target = 10.0
        starting_setpoint = 15.0
        cadence = 300.0
        max_iterations = 40
        open_time = 0.0
        policy = delay_controller.make_evening_controlled_convergence_policy(
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
            slew_rate=slew_rate,
        )
        controller = delay_controller.DelayController(policy=policy, log=logging.getLogger())
        controller.state = delay_controller.DelayState.WAITING
        controller.open_time = open_time

        last_setpoint = starting_setpoint
        applied: list[float] = []

        async def apply_setpoints(value: float) -> None:
            applied.append(value)

        ready = False
        for _ in range(max_iterations):
            ready = await controller.gate(
                target_setpoint=target,
                last_m1m3ts_setpoint=last_setpoint,
                cadence=cadence,
                apply_setpoints_callback=apply_setpoints,
            )
            if ready:
                break
            last_setpoint = applied[-1]

        self.assertTrue(ready)
        step = slew_rate * cadence / 3600.0
        n_steps = math.ceil((starting_setpoint - (target + tolerance_positive)) / step)
        expected = [starting_setpoint - step * (i + 1) for i in range(n_steps)]
        self.assertEqual(len(applied), len(expected))
        for got, exp in zip(applied, expected, strict=True):
            self.assertAlmostEqual(got, exp, places=6)
        self.assertTrue(
            delay_controller.within_tolerance(
                target=target,
                current=last_setpoint,
                tolerance_positive=tolerance_positive,
                tolerance_negative=tolerance_negative,
            )
        )
