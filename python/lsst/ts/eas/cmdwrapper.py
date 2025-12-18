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
from typing import Any, Callable

from lsst.ts import salobj
from lsst.ts.utils import current_tai

ExceptionCallback = Callable[[Exception], None]

DEFAULT_DORMANT_TIME = 60.0


class CommandWrapper:
    """
    Fire-and-forget wrapper for issuing a single SAL command with ENABLED
    gating and supersession semantics.

    This class allows issuing a command to a CSC, ensuring the command will be
    sent when the CSC is ENABLED. If a new command is issued, while waiting for
    the CSC to be enabled, the previous one is discarded. Errors from the
    command are logged, and a callback is used if defined.

    Parameters
    ----------
    log : `logging.Logger`
        The Logger object for logging.
    remote : `~salobj.Remote`
        The SAL remote to control with the command.
    command : `~salobj.topics.RemoteCommand`
        The RemoteCommand object with which to issue the command.

    Attributes
    ----------
    timeout : `float` | `None`
        Maximum time, in seconds, to wait for the remote to become ENABLED.
        If None, wait indefinitely.
    dormant_time : `float`
        Sleep interval, in seconds, between ENABLED state checks.
    exception_callback : `ExceptionCallback` | `None`
        A callback that will be called if the command raises an exception.
    """

    def __init__(
        self,
        log: logging.Logger,
        remote: salobj.Remote,
        command: salobj.topics.RemoteCommand,
    ) -> None:
        self.exception_callback: ExceptionCallback | None = None
        self.timeout: float | None = None
        self.task: asyncio.Task | None = None
        self.dormant_time = DEFAULT_DORMANT_TIME
        self.log = log
        self.remote = remote
        self.command = command

    async def close(self) -> None:
        """
        Cancel any in-flight command task and wait for it to finish.

        This is intended to be called during shutdown to ensure that background
        wait-and-send tasks are cleaned up and do not continue running.
        """
        if not self.task:
            return

        task = self.task
        self.task = None

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.exception("Unexpected exception while sending CSC command.")

    def __str__(self) -> str:
        remote_name = self.remote.salinfo.name
        index = self.remote.salinfo.index
        command_name = self.command.topic_info.attr_name

        if index is not None and index > 0:
            return f"{remote_name}:{index}.{command_name}"
        else:
            return f"{remote_name}.{command_name}"

    def __repr__(self) -> str:
        return f"<CommandWrapper {self}>"

    async def _run(self, kwargs_list: list[dict[str, Any]]) -> None:
        """
        Wait for the remote to become ENABLED and then issue the command.

        This coroutine runs in the background. It exits quietly if superseded,
        propagates external cancellation, and logs any exceptions raised while
        issuing the command.

        Parameters
        ----------
        kwargs_list : `list` [`dict` [`str`, `Any`]]
            List of sets of keyword arguments to pass to the underlying SAL
            command `set_start`. Each item in the list represents one call
            to `set_start`.
        """
        deadline = None if self.timeout is None else current_tai() + self.timeout

        try:
            while deadline is None or deadline > current_tai():
                summary_state = self.remote.evt_summaryState.get()
                if summary_state is not None and summary_state.summaryState == salobj.State.ENABLED:
                    for kwargs in kwargs_list:
                        await self.command.set_start(**kwargs)
                    return
                else:
                    await asyncio.sleep(self.dormant_time)

            # Deadline elapsed.
            self.log.warning(f"Command {self} timed out waiting for CSC to be enabled.")

        except asyncio.CancelledError as e:
            if e.args and e.args[0] == "superseded":
                self.log.warning(f"Prior call to {self} was superseded.")
            raise
        except Exception as e:
            self.log.warning(
                f"Error returned by {self}:\n{e}",
                exc_info=True,
            )

            # Call the exception callback if needed.
            cb = self.exception_callback
            if cb is not None:
                try:
                    cb(e)
                except Exception:
                    self.log.exception(f"{self}: exception_callback raised.")

    async def set_start(self, **kwargs: Any) -> None:
        """
        Schedule a command invocation, superseding any previous pending
        invocation.

        This method starts a background task that waits for the remote to
        become ENABLED and then issues the command. If a prior invocation is
        still pending, it is cancelled and treated as superseded.

        Parameters
        ----------
        kwargs : `dict` [`str`, `Any`]
            Keyword arguments to pass to the underlying SAL command
            `set_start`.
        """
        await self.set_start_multi([kwargs])

    async def set_start_multi(self, kwargs_list: list[dict[str, Any]]) -> None:
        """
        Schedule one or more command invocations, superseding any previous
        pending invocation.

        This method starts a background task that waits for the remote to
        become ENABLED and then issues the command once for each set of keyword
        arguments in ``kwargs_list``. If a prior invocation is still pending,
        it is cancelled and treated as superseded.

        Parameters
        ----------
        kwargs_list : `list` [`dict` [`str`, `Any`]]
            Sequence of keyword-argument mappings to pass to the underlying SAL
            command ``set_start``, in the order they should be issued.
        """
        old_task = self.task
        new_task = asyncio.create_task(self._run(kwargs_list))
        self.task = new_task

        if old_task and not old_task.done():
            old_task.cancel("superseded")
            try:
                await old_task
            except asyncio.CancelledError:
                pass
