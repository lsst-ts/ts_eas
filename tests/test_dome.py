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
import unittest
from types import SimpleNamespace

from lsst.ts import eas, utils

STD_SLEEP = 0.2


def get_open_dome_telemetry() -> SimpleNamespace:
    """Returns telemetry indicating an open dome."""
    return SimpleNamespace(
        private_sndStamp=utils.current_tai(),
        positionActual=[100.0, 100.0],
    )


def get_closed_dome_telemetry() -> SimpleNamespace:
    """Returns telemetry indicating an closed dome."""
    return SimpleNamespace(
        private_sndStamp=utils.current_tai(),
        positionActual=[0.0, 0.0],
    )


class TestDomeModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.model = eas.dome_model.DomeModel()
        # Always start with dome closed.
        await self.model.aperture_shutter_callback(get_closed_dome_telemetry())

    async def test_cancel_pending_events_cancels_and_sets_events(self) -> None:
        event = asyncio.Event()

        self.model.on_open.append((event, 1000.0))
        await self.model.aperture_shutter_callback(get_open_dome_telemetry())

        handle = self.model.delayed_events.get(event)
        self.model.cancel_pending_events()

        # Handle should be cancelled
        self.assertTrue(handle.cancelled())
        # Event should have been set
        self.assertTrue(event.is_set())
        # Internal bookkeeping cleared
        self.assertFalse(self.model.delayed_events)

    async def test_aperture_shutter_callback_schedules_delayed_events(self) -> None:
        event = asyncio.Event()
        delay = 0.01
        self.model.on_open.append((event, delay))

        # Open the dome
        await self.model.aperture_shutter_callback(get_open_dome_telemetry())

        # on_open should be cleared
        self.assertFalse(self.model.on_open)

        # One handle per event
        self.assertEqual(len(self.model.delayed_events), 1)

        # Event should be set after the delay
        await asyncio.wait_for(event.wait(), timeout=0.2)
        self.assertTrue(event.is_set())

    async def test_aperture_shutter_callback_cancels_pending_on_close(self) -> None:
        event = asyncio.Event()
        delay = 1000.0
        self.model.on_open.append((event, delay))

        # Open the dome...
        await self.model.aperture_shutter_callback(get_open_dome_telemetry())

        self.assertEqual(len(self.model.delayed_events), 1)
        await asyncio.sleep(0)

        # Close the dome before the delay expires...
        await self.model.aperture_shutter_callback(get_closed_dome_telemetry())
        await asyncio.sleep(0)

        self.assertTrue(event.is_set())
        self.assertFalse(self.model.delayed_events)

    async def test_delayed_events_cleared_as_tasks_complete(self) -> None:
        self.model = eas.dome_model.DomeModel()
        await self.model.aperture_shutter_callback(get_closed_dome_telemetry())

        event = asyncio.Event()
        delay = 0.01
        self.model.on_open.append((event, delay))

        # Open the dome...
        await self.model.aperture_shutter_callback(get_open_dome_telemetry())
        await asyncio.sleep(STD_SLEEP)

        self.assertTrue(event.is_set())
        self.assertFalse(self.model.delayed_events)
