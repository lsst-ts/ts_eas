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

__all__ = ["command_wrapper", "close_command_tasks", "DEFAULT_DORMANT_TIME"]

import asyncio
import functools
import logging
from typing import Any, Callable, Coroutine

from lsst.ts import salobj
from lsst.ts.utils import current_tai

ExceptionCallback = Callable[[Exception], None]

DEFAULT_DORMANT_TIME = 60.0
TASK_ATTR_REGISTRY = "_command_wrapper_task_attrs"


def _command_label(remote: salobj.Remote, command: salobj.topics.RemoteCommand) -> str:
    """Return a display label for the remote command.

    Parameters
    ----------
    remote : `~salobj.Remote`
        The SAL remote associated with the command.
    command : `~salobj.topics.RemoteCommand`
        The command object used to generate the label.

    Returns
    -------
    `str`
        Human-readable label for logging and diagnostics.
    """
    remote_name = remote.salinfo.name
    index = remote.salinfo.index
    command_name = command.topic_info.attr_name
    if index is not None and index > 0:
        return f"{remote_name}:{index}.{command_name}"
    return f"{remote_name}.{command_name}"


async def _run_command(
    *,
    log: logging.Logger,
    remote: salobj.Remote,
    command: salobj.topics.RemoteCommand,
    kwargs_list: list[dict[str, Any]],
    timeout: float | None,
    dormant_time: float,
    exception_callback: ExceptionCallback | None,
) -> None:
    """Wait for ENABLED, issue command calls, and handle errors.

    Parameters
    ----------
    log : `logging.Logger`
        Logger to use for warnings and exceptions.
    remote : `~salobj.Remote`
        Remote whose summary state gates command issuance.
    command : `~salobj.topics.RemoteCommand`
        Command to invoke once the remote is ENABLED.
    kwargs_list : `list` [`dict` [`str`, `Any`]]
        Sequence of keyword argument mappings for each command call.
    timeout : `float` | `None`
        Maximum time (seconds) to wait for ENABLED, or None to wait
        indefinitely.
    dormant_time : `float`
        Sleep interval between summary state checks.
    exception_callback : `ExceptionCallback` | `None`
        Optional callback invoked when the command raises an exception.
    """
    label = _command_label(remote, command)
    deadline = None if timeout is None else current_tai() + timeout

    try:
        while deadline is None or deadline > current_tai():
            summary_state = remote.evt_summaryState.get()
            if summary_state is not None and summary_state.summaryState == salobj.State.ENABLED:
                for kwargs in kwargs_list:
                    await command.set_start(**kwargs)
                return
            await asyncio.sleep(dormant_time)

        log.warning(f"Command {label} timed out waiting for CSC to be enabled.")

    except asyncio.CancelledError as e:
        if e.args and e.args[0] == "superseded":
            log.warning(f"Prior call to {label} was superseded.")
        raise
    except Exception as e:
        log.warning(
            f"Error returned by {label}:\n{e}",
            exc_info=True,
        )
        if exception_callback is not None:
            try:
                exception_callback(e)
            except Exception:
                log.exception(f"{label}: exception_callback raised.")


def _normalize_kwargs_list(result: Any) -> list[dict[str, Any]] | None:
    """Normalize a decorator result into a list of kwargs dicts.

    Parameters
    ----------
    result : `Any`
        Value returned by the wrapped method.

    Returns
    -------
    `list` [`dict` [`str`, `Any`]] | `None`
        Normalized list of kwargs mappings, or None to skip execution.

    Raises
    ------
    `TypeError`
        If the decorator returned neither `dict` nor `list` nor `None`.
    """
    if result is None:
        return None
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return result
    raise TypeError("Command wrapper expected a dict or list of dicts.")


def command_wrapper(
    *,
    remote_attr: str,
    command_attr: str,
    timeout: float | None = None,
    dormant_time: float = DEFAULT_DORMANT_TIME,
    exception_callback_attr: str = "command_exception_callback",
) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, None]]]:
    """Decorate an async method to issue a command when the CSC is ENABLED.

    Parameters
    ----------
    remote_attr : `str`
        Attribute name on the instance that resolves to a `~salobj.Remote`.
    command_attr : `str`
        Attribute name on the remote that resolves to the command topic.
    timeout : `float` | `None`
        Maximum time to wait for ENABLED, or None to wait indefinitely.
    dormant_time : `float`
        Sleep interval between summary state checks.
    exception_callback_attr : `str`
        Attribute name on the instance containing an exception callback.

    Returns
    -------
    `Callable`
        Decorator that wraps async methods with the command behavior.
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, None]]:
        """Wrap an async method with gating and supersession behavior.

        Parameters
        ----------
        func : `Callable`
            Async method that returns command kwargs.

        Returns
        -------
        `Callable`
            Wrapped async method that schedules the command task.
        """
        task_attr = f"_{func.__name__}_command_task"

        @functools.wraps(func)
        async def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
            """Execute the wrapped method and schedule the command task.

            Parameters
            ----------
            self : `Any`
                Instance on which the method is invoked.
            *args : `tuple`
                Positional arguments forwarded to the wrapped method.
            **kwargs : `dict`
                Keyword arguments forwarded to the wrapped method.
            """
            kwargs_list = _normalize_kwargs_list(await func(self, *args, **kwargs))
            if not kwargs_list:
                return

            remote = getattr(self, remote_attr)
            command = getattr(remote, command_attr)
            log = getattr(self, "log", logging.getLogger(__name__))
            exception_callback = getattr(self, exception_callback_attr, None)

            old_task = getattr(self, task_attr, None)
            new_task = asyncio.create_task(
                _run_command(
                    log=log,
                    remote=remote,
                    command=command,
                    kwargs_list=kwargs_list,
                    timeout=timeout,
                    dormant_time=dormant_time,
                    exception_callback=exception_callback,
                )
            )
            setattr(self, task_attr, new_task)

            # Track task attribute names per instance so they can be cleaned up
            # when superseded or when close_command_tasks is called.
            registry = getattr(self, TASK_ATTR_REGISTRY, None)
            if registry is None:
                registry = []
                setattr(self, TASK_ATTR_REGISTRY, registry)
            if task_attr not in registry:
                registry.append(task_attr)

            if old_task and not old_task.done():
                old_task.cancel("superseded")
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass

        setattr(wrapped, "_command_task_attr", task_attr)
        return wrapped

    return decorator


async def close_command_tasks(instance: Any) -> None:
    """Cancel and await any in-flight command tasks created by decorators.

    Parameters
    ----------
    instance : `Any`
        Object that owns command tasks created by `command_wrapper`.
    """
    task_attrs = getattr(instance, TASK_ATTR_REGISTRY, [])
    log = getattr(instance, "log", logging.getLogger(__name__))

    for task_attr in list(task_attrs):
        task = getattr(instance, task_attr, None)
        if not task:
            continue
        setattr(instance, task_attr, None)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Unexpected exception while sending CSC command.")
