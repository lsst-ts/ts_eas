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

__all__ = ["DomeModel"]

import asyncio
from collections import deque

from lsst.ts import salobj, utils

DORMANT_TIME = 300  # Time to wait while sleeping, seconds
MAX_TELEMETRY_AGE = 300  # Time at which apertureShutter telemetry expires, seconds
DOME_OPEN_THRESHOLD = 50  # Dome open percentage at which the dome is considered "open"


class DomeModel:
    """A model for MTDome.

    Track whether and when the dome has been opened.

    Parameters
    ---------
    log : `~logging.Logger`
        A logger for log messages.
    """

    def __init__(self) -> None:
        self.monitor_start_event = asyncio.Event()

        # Most recent tel_apertureShutter
        self.aperture_shutter_telemetry: salobj.BaseMsgType | None = None
        self.on_open: deque[tuple[asyncio.Event, float]] = deque()
        self.delayed_events: dict[asyncio.Event, asyncio.Handle] = dict()
        self.was_closed: bool | None = None

    def cancel_pending_events(self) -> None:
        """Cancels all pending handles scheduled to set events.

        Any events waiting to be set will be set at this time.
        The handles associated with the waiting events will be
        cancelled.
        """
        if not self.delayed_events:
            return

        delayed_events = dict(self.delayed_events)
        self.delayed_events.clear()
        for event, handle in delayed_events.items():
            handle.cancel()
            event.set()

    async def aperture_shutter_callback(
        self, aperture_shutter_telemetry: salobj.BaseMsgType
    ) -> None:
        """Callback for MTDome.tel_apertureShutter.

        Data from this telemetry item is used to determine whether the dome
        is opened or closed.

        Parameters
        ----------
        aperture_shutter_telemetry: `~lsst.ts.salobj.BaseMsgType`
            A newly received apertureShutter telemetry item.
        """
        self.aperture_shutter_telemetry = aperture_shutter_telemetry
        is_closed = self.is_closed

        if self.was_closed is not False and is_closed is False:
            events_to_signal = list(self.on_open)
            self.on_open.clear()
            loop = asyncio.get_running_loop()

            for event, delay in events_to_signal:

                def fire_event() -> None:
                    event.set()
                    self.delayed_events.pop(event)

                handle = loop.call_later(delay, fire_event)
                self.delayed_events[event] = handle

        elif is_closed and self.delayed_events:
            self.cancel_pending_events()

        self.was_closed = is_closed

    @property
    def is_closed(self) -> bool | None:
        """Return true if the dome is currently closed.

        If the current state of the dome is unknown, None is returned.
        """
        if self.aperture_shutter_telemetry is None:
            return None

        send_timestamp = self.aperture_shutter_telemetry.private_sndStamp
        telemetry_age = utils.current_tai() - send_timestamp
        if telemetry_age > MAX_TELEMETRY_AGE:
            return None

        return (
            self.aperture_shutter_telemetry.positionActual[0] < DOME_OPEN_THRESHOLD
            and self.aperture_shutter_telemetry.positionActual[1] < DOME_OPEN_THRESHOLD
        )
