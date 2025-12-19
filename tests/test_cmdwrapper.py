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
from typing import Any, Callable

from lsst.ts import salobj
from lsst.ts.eas.cmdwrapper import close_command_tasks, command_wrapper


class FakeEvtSummaryState:
    def __init__(self) -> None:
        self._msg: SimpleNamespace | None = None

    def set_state(self, state: salobj.State | None) -> None:
        self._msg = None if state is None else SimpleNamespace(summaryState=state)

    def get(self) -> SimpleNamespace | None:
        return self._msg


class FakeRemote:
    def __init__(self) -> None:
        self.evt_summaryState = FakeEvtSummaryState()
        self.salinfo = SimpleNamespace(name="FAKE", index=0)
        self.cmd_test: "FakeCommand" | None = None


class FakeCommand:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._raise: Exception | None = None
        self._block: asyncio.Event | None = None
        self.topic_info = SimpleNamespace(attr_name="cmd_test")

    def raise_on_call(self, ex: Exception) -> None:
        self._raise = ex

    def block_on_call(self, ev: asyncio.Event) -> None:
        self._block = ev

    async def set_start(self, **kwargs: Any) -> None:
        if self._block is not None:
            await self._block.wait()
        if self._raise is not None:
            raise self._raise
        self.calls.append(kwargs)


def make_model(
    remote: FakeRemote,
    log: logging.Logger,
    *,
    timeout: float | None = None,
    dormant_time: float = 0.01,
) -> Any:
    class FakeModel:
        def __init__(self, remote: FakeRemote, log: logging.Logger) -> None:
            self.remote = remote
            self.log = log
            self.command_exception_callback: Callable[[Exception], None] | None = None

        @command_wrapper(
            remote_attr="remote",
            command_attr="cmd_test",
            timeout=timeout,
            dormant_time=dormant_time,
        )
        async def send(self, **kwargs: Any) -> dict[str, Any]:
            return kwargs

        @command_wrapper(
            remote_attr="remote",
            command_attr="cmd_test",
            timeout=timeout,
            dormant_time=dormant_time,
        )
        async def send_batch(self) -> list[dict[str, Any]]:
            return [{"alpha": 1}, {"beta": 2}]

    return FakeModel(remote, log)


async def spin_until(pred: Callable[[], bool], *, timeout: float = 0.5, step: float = 0.01) -> None:
    """Poll `pred()` until true or timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return
        await asyncio.sleep(step)
    raise TimeoutError("Condition not met before timeout")


class TestCommandWrapper(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.remote = FakeRemote()
        self.command = FakeCommand()
        self.log = logging.getLogger("CommandWrapper")
        self.remote.cmd_test = self.command
        self.fake_model = make_model(self.remote, self.log)

    async def asyncTearDown(self) -> None:
        await close_command_tasks(self.fake_model)

    async def test_sends_once_enabled(self) -> None:
        self.remote.evt_summaryState.set_state(None)

        await self.fake_model.send(foo=1)

        # Not enabled yet: should not have sent
        self.assertEqual(self.command.calls, [])

        # Now enable and wait for send
        self.remote.evt_summaryState.set_state(salobj.State.ENABLED)
        await spin_until(lambda: len(self.command.calls) == 1)

        self.assertEqual(self.command.calls, [{"foo": 1}])

    async def test_skips_when_sending_disabled(self) -> None:
        self.remote.evt_summaryState.set_state(salobj.State.ENABLED)
        self.fake_model.allow_send = lambda: False

        with self.assertLogs("CommandWrapper", level="DEBUG") as cm:
            await self.fake_model.send(foo=1)
            await asyncio.sleep(0)

            task_attr = self.fake_model.send._command_task_attr
            task = getattr(self.fake_model, task_attr)
            self.assertTrue(task.done())
            self.assertIsNone(task.result())
            self.assertEqual(self.command.calls, [])
            self.assertTrue(any("not allowed" in msg.lower() for msg in cm.output))

    async def test_supersedes_prior_call(self) -> None:
        self.remote.evt_summaryState.set_state(None)

        with self.assertLogs("CommandWrapper", level="WARNING") as cm:
            await self.fake_model.send(foo=1)
            await asyncio.sleep(0)
            await self.fake_model.send(foo=2)  # supersedes foo=1 before enabled

            self.remote.evt_summaryState.set_state(salobj.State.ENABLED)
            await spin_until(lambda: len(self.command.calls) == 1)

            self.assertEqual(self.command.calls, [{"foo": 2}])
            self.assertTrue(any("superseded" in msg.lower() for msg in cm.output))

    async def test_sends_multiple_for_list_result(self) -> None:
        self.remote.evt_summaryState.set_state(salobj.State.ENABLED)
        await self.fake_model.send_batch()
        await spin_until(lambda: len(self.command.calls) == 2)

        self.assertEqual(self.command.calls, [{"alpha": 1}, {"beta": 2}])

    async def test_logs_exception_from_command(self) -> None:
        error_message = "a total disaster occurred"
        self.remote.evt_summaryState.set_state(salobj.State.ENABLED)
        self.command.raise_on_call(RuntimeError(error_message))

        with self.assertLogs("CommandWrapper", level="WARNING") as cm:
            await self.fake_model.send(foo=1)
            # Give the background task a moment to run
            await asyncio.sleep(0.05)

            # Command should not have recorded a successful call
            self.assertEqual(self.command.calls, [])
            self.assertTrue(any(error_message in msg.lower() for msg in cm.output))

    async def test_close_cancels_inflight_task(self) -> None:
        self.remote.evt_summaryState.set_state(None)
        fake_model = make_model(self.remote, self.log, dormant_time=10.0)
        await fake_model.send(foo=1)
        task_attr = fake_model.send._command_task_attr
        task = getattr(fake_model, task_attr)
        self.assertIsNotNone(task)
        self.assertFalse(task.done())

        await close_command_tasks(fake_model)
        self.assertIsNone(getattr(fake_model, task_attr))
        self.assertTrue(task.done())

    async def test_exception_callback_invoked(self) -> None:
        self.remote.evt_summaryState.set_state(salobj.State.ENABLED)

        exc = RuntimeError("boom")
        self.command.raise_on_call(exc)

        seen: list[Exception] = []

        def exception_callback(e: Exception) -> None:
            seen.append(e)

        self.fake_model.command_exception_callback = exception_callback

        with self.assertLogs("CommandWrapper", level="WARNING"):
            await self.fake_model.send(foo=123)
            await asyncio.sleep(0.05)

        # Command should not succeed
        self.assertEqual(self.command.calls, [])

        # Callback should have been called once with the same exception
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], exc)

    async def test_timeout_logs_and_does_not_send(self) -> None:
        # Never enable the remote
        self.remote.evt_summaryState.set_state(None)

        fake_model = make_model(self.remote, self.log, timeout=0.05, dormant_time=0.01)

        callback_called = False

        def exception_callback(_: Exception) -> None:
            nonlocal callback_called
            callback_called = True

        fake_model.command_exception_callback = exception_callback

        with self.assertLogs("CommandWrapper", level="WARNING") as cm:
            await fake_model.send(foo=456)
            # Allow enough time for timeout + logging
            await asyncio.sleep(0.1)

            # Command should never have been sent
            self.assertEqual(self.command.calls, [])

            # Timeout should produce a warning
            self.assertTrue(
                any("timed out" in msg.lower() for msg in cm.output),
                cm.output,
            )

            # Timeout is not an exception, so callback must NOT be called
            self.assertFalse(callback_called)

        await close_command_tasks(fake_model)
