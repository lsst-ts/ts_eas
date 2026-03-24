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

__all__ = [
    "DelayController",
    "DelayPolicy",
    "DelayState",
    "make_delay_policy_from_config",
    "make_evening_controlled_convergence_policy",
    "make_time_delay_policy",
    "make_wait_for_convergence_policy",
]

import asyncio
import enum
import logging
import math
from dataclasses import dataclass
from typing import Awaitable, Callable

import astropy.units as u

from lsst.ts.utils import current_tai

from .dome_model import DomeModel

# Polling frequency to wait for an initial state report from the dome.
POLL_INTERVAL = 10.0

# A callback type for reporting back that new setpoints should be applied.
ApplySetpointsCallback = Callable[[float], Awaitable[None]]


class DelayState(enum.Enum):
    """State of the delay controller.

    The controller starts in ``IDLE`` and transitions to ``WAITING`` after
    the dome opens. It becomes ``READY`` once the policy conditions are met.
    """

    IDLE = enum.auto()
    WAITING = enum.auto()
    READY = enum.auto()


@dataclass(frozen=True)
class DelayPolicy:
    """Encapsulate the logic for determining readiness and hold behavior.

    Parameters
    ----------
    is_ready : `Callable`
        Callable that returns `True` when tracking can resume. Takes
        arguments of
         * `target_setpoint` : `float`
           The target setpoint dictated by environment sensors, °C.
         * `last_m1m3ts_setpoint` : `float`
           The last setpoint applied to M1M3TS, °C.
         * `open_time` : `float`
           The time at which the dome opened, TAI seconds.
         * `now` : `float`
           The current time, TAI seconds.
    on_hold : `Callable`
        Coroutine invoked while tracking is held. This function handles
        activities managed by the controller while it is in the WAITING
        state. Takes arguments of
         * `target_setpoint` : `float`
           The target setpoint dictated by environment sensors, °C.
         * `last_m1m3ts_setpoint` : `float`
           The last setpoint applied to M1M3TS, °C.
         * `cadence` : `float`
           Cadence between checks, seconds.
         * `apply_setpoints_callback` : `ApplySetpointsCallback` | None
           Callback for applying setpoints, or `None` if not applicable.
    """

    is_ready: Callable[[float | None, float | None, float, float], bool]
    on_hold: Callable[[float | None, float | None, float, ApplySetpointsCallback | None], Awaitable[None]]


def within_tolerance(
    *,
    target: float | None,
    current: float | None,
    tolerance_positive: float,
    tolerance_negative: float,
) -> bool:
    """Convenience function to check whether a target/current pair are within
    asymmetric tolerances.

    Parameters
    ----------
    target : `float` | `None`
        Desired setpoint.
    current : `float` | `None`
        Current or last-applied setpoint.
    tolerance_positive : `float`
        Maximum allowed positive difference, in degrees Celsius.
    tolerance_negative : `float`
        Maximum allowed negative difference, in degrees Celsius.

    Returns
    -------
    `bool`
        `True` if the values are within tolerance, otherwise `False`.
    """

    if target is None or current is None:
        return False
    if not math.isfinite(target) or not math.isfinite(current):
        return False

    diff = current - target  # positive if current > target
    if diff >= 0:
        return diff <= tolerance_positive
    return -diff <= tolerance_negative


def make_convergence_is_ready(
    *,
    tolerance_positive: float,
    tolerance_negative: float,
    timeout_seconds: float | None,
) -> Callable[[float | None, float | None, float, float], bool]:
    """Create a readiness predicate based on tolerance and optional timeout.

    Parameters
    ----------
    tolerance_positive : `float`
        Allowed positive deviation, in degrees Celsius.
    tolerance_negative : `float`
        Allowed negative deviation, in degrees Celsius.
    timeout_seconds : `float` | `None`
        Maximum time to wait after opening before resuming tracking. A
        value of `None` disables the timeout.

    Returns
    -------
    `Callable`
        Callable that returns `True` when the tolerance criteria are met,
        or when the optional timeout has elapsed.
    """

    def is_ready(
        target_setpoint: float | None,
        last_m1m3ts_setpoint: float | None,
        open_time: float,
        now: float,
    ) -> bool:
        """Signal ready based on tolerance criterion, with timeout."""
        if within_tolerance(
            target=target_setpoint,
            current=last_m1m3ts_setpoint,
            tolerance_positive=tolerance_positive,
            tolerance_negative=tolerance_negative,
        ):
            return True
        if timeout_seconds is None:
            return False
        return (now - open_time) >= timeout_seconds

    return is_ready


def make_time_delay_policy(*, delay_seconds: float) -> DelayPolicy:
    """Create a fixed time-delay policy.

    Parameters
    ----------
    delay_seconds : `float`
        Time in seconds to wait after dome opening.

    Returns
    -------
    `DelayPolicy`
        Policy that becomes ready after the specified delay.
    """

    def is_ready(
        target_setpoint: float | None,
        last_m1m3ts_setpoint: float | None,
        open_time: float,
        now: float,
    ) -> bool:
        """Return a decision based solely on the clock."""
        return (now - open_time) >= delay_seconds

    async def on_hold(
        target_setpoint: float | None,
        last_m1m3ts_setpoint: float | None,
        cadence: float,
        apply_setpoints_cb: ApplySetpointsCallback | None,
    ) -> None:
        """Do nothing while in WAITING."""
        return None

    return DelayPolicy(is_ready=is_ready, on_hold=on_hold)


def make_wait_for_convergence_policy(
    *,
    tolerance_positive: float,
    tolerance_negative: float,
    timeout_seconds: float | None,
) -> DelayPolicy:
    """Create a convergence policy with an optional timeout.

    Parameters
    ----------
    tolerance_positive : `float`
        Allowed positive deviation, in degrees Celsius.
    tolerance_negative : `float`
        Allowed negative deviation, in degrees Celsius.
    timeout_seconds : `float` | `None`
        Maximum time to wait after opening before resuming tracking. A
        value of `None` disables the timeout.

    Returns
    -------
    `DelayPolicy`
        Policy that waits for convergence or a timeout.
    """

    is_ready = make_convergence_is_ready(
        tolerance_positive=tolerance_positive,
        tolerance_negative=tolerance_negative,
        timeout_seconds=timeout_seconds,
    )

    async def on_hold(
        target_setpoint: float | None,
        last_m1m3ts_setpoint: float | None,
        cadence: float,
        apply_setpoints_cb: ApplySetpointsCallback | None,
    ) -> None:
        """Do nothing while WAITING."""
        return None

    return DelayPolicy(is_ready=is_ready, on_hold=on_hold)


def make_evening_controlled_convergence_policy(
    *,
    tolerance_positive: float,
    tolerance_negative: float,
    slew_rate: float,
    timeout_seconds: float | None = None,
) -> DelayPolicy:
    """Create a convergence policy that slews toward the target.

    Parameters
    ----------
    tolerance_positive : `float`
        Allowed positive deviation, in degrees Celsius.
    tolerance_negative : `float`
        Allowed negative deviation, in degrees Celsius.
    slew_rate : `float`
        Slew rate in degrees Celsius per hour.
    timeout_seconds : `float` | `None`, optional
        Maximum time to wait after opening before resuming tracking. A
        value of `None` disables the timeout.

    Returns
    -------
    `DelayPolicy`
        Policy that slews toward the target while converging.
    """

    is_ready = make_convergence_is_ready(
        tolerance_positive=tolerance_positive,
        tolerance_negative=tolerance_negative,
        timeout_seconds=timeout_seconds,
    )

    async def on_hold(
        target_setpoint: float | None,
        last_m1m3ts_setpoint: float | None,
        cadence: float,
        apply_setpoints_callback: ApplySetpointsCallback | None,
    ) -> None:
        """Slew the setpoint temperature while WAITING."""

        if apply_setpoints_callback is None:
            raise ValueError("apply_setpoints_callback must be set.")

        if target_setpoint is None or not math.isfinite(target_setpoint):
            return
        if last_m1m3ts_setpoint is None or not math.isfinite(last_m1m3ts_setpoint):
            await apply_setpoints_callback(target_setpoint)
            return

        rate_c_per_hour = slew_rate * u.deg_C / u.hour
        rate_c_per_sec = rate_c_per_hour.to(u.deg_C / u.second)
        step = float(rate_c_per_sec.value) * cadence

        if target_setpoint > last_m1m3ts_setpoint:
            new_setpoint = min(target_setpoint, last_m1m3ts_setpoint + step)
        else:
            new_setpoint = max(target_setpoint, last_m1m3ts_setpoint - step)

        await apply_setpoints_callback(new_setpoint)

    return DelayPolicy(is_ready=is_ready, on_hold=on_hold)


def make_delay_policy_from_config(config: dict) -> DelayPolicy:
    """Create a delay policy from a parsed configuration dict.

    Parameters
    ----------
    config : `dict`
        Configuration mapping for the delay mode. Expected to contain a
        ``mode`` key and the associated parameters.

    Returns
    -------
    `DelayPolicy`
        Policy instance configured for the requested delay mode.
    """

    if "mode" not in config:
        raise ValueError("Delay policy configuration must include a 'mode' key.")

    mode = config["mode"]
    if mode == "time_delay":
        return make_time_delay_policy(delay_seconds=config["delay"])
    if mode == "wait_for_convergence":
        return make_wait_for_convergence_policy(
            tolerance_positive=config["tolerance_positive"],
            tolerance_negative=config["tolerance_negative"],
            timeout_seconds=config.get("timeout"),
        )
    if mode == "evening_controlled_convergence":
        return make_evening_controlled_convergence_policy(
            tolerance_positive=config["tolerance_positive"],
            tolerance_negative=config["tolerance_negative"],
            slew_rate=config["slew_rate"],
            timeout_seconds=config.get("timeout"),
        )

    raise ValueError(f"Unknown delay policy mode: {mode!r}")


class DelayController:
    """Gate setpoint updates after dome opening.

    Parameters
    ----------
    policy : `DelayPolicy`
        Policy instance controlling the delay behavior.
    log : `logging.Logger`
        Logger for status messages.
    """

    def __init__(
        self,
        *,
        policy: DelayPolicy,
        log: logging.Logger,
    ) -> None:
        self.policy = policy
        self.log = log
        self.state = DelayState.IDLE
        self.open_time: float | None = None

    async def wait_for_open(self, dome_model: DomeModel) -> None:
        """Wait until dome telemetry indicates the dome is open.

        If the dome is already open when monitoring begins, the controller
        records the start time as the effective open time. In this case,
        delay policies measure elapsed time from when EAS first observes the
        dome in the open state, not from the physical dome opening event.

        Parameters
        ----------
        dome_model : object
            Dome model providing ``is_closed`` and ``on_open``.
        """

        while dome_model.is_closed is None:
            await asyncio.sleep(POLL_INTERVAL)

        while dome_model.is_closed is not False:
            event = asyncio.Event()
            on_open_event = (event, 0.0)
            dome_model.on_open.append(on_open_event)

            # Recheck after registering the waiter so an open transition
            # between the loop condition and append cannot be lost.
            if dome_model.is_closed is False:
                try:
                    dome_model.on_open.remove(on_open_event)
                except ValueError:
                    pass
                break

            self.log.debug("Waiting for dome open (delay controller).")
            await event.wait()

        self.open_time = current_tai()
        self.state = DelayState.WAITING

    def reset(self) -> None:
        """Reset state after dome closes."""

        self.state = DelayState.IDLE
        self.open_time = None

    def is_ready(self) -> bool:
        """Return whether the controller has released the delay."""

        return self.state == DelayState.READY

    async def gate(
        self,
        *,
        target_setpoint: float | None,
        last_m1m3ts_setpoint: float | None,
        cadence: float,
        apply_setpoints_callback: ApplySetpointsCallback | None,
    ) -> bool:
        """Check if tracking can resume and perform hold actions if not.

        Parameters
        ----------
        target_setpoint : `float` | `None`
            Desired setpoint to resume tracking toward.
        last_m1m3ts_setpoint : `float` | `None`
            Most recent setpoint applied to M1M3TS.
        cadence : `float`
            Cadence in seconds between checks.
        apply_setpoints_callback : `Callable` or `None`
            Coroutine to apply a new M1M3TS setpoint, or `None` if not
            applicable.

        Returns
        -------
        `bool`
            `True` if tracking can resume; otherwise `False`.
        """

        if self.state == DelayState.READY:
            return True

        if self.open_time is None:
            return False

        if self.policy.is_ready(target_setpoint, last_m1m3ts_setpoint, self.open_time, current_tai()):
            self.state = DelayState.READY
            return True

        await self.policy.on_hold(
            target_setpoint,
            last_m1m3ts_setpoint,
            cadence,
            apply_setpoints_callback,
        )
        return False
