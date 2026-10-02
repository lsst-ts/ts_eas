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
import unittest
from types import SimpleNamespace

from lsst.ts import eas, utils

STD_SLEEP = 0.2


def get_open_shutter_telemetry() -> SimpleNamespace:
    """Returns telemetry indicating an open shutter."""
    return SimpleNamespace(
        private_sndStamp=utils.current_tai(),
        positionActual=[100.0, 100.0],
    )


def get_closed_shutter_telemetry() -> SimpleNamespace:
    """Returns telemetry indicating an closed shutter."""
    return SimpleNamespace(
        private_sndStamp=utils.current_tai(),
        positionActual=[0.0, 0.0],
    )


def get_open_louver_telemetry() -> SimpleNamespace:
    """Returns telemetry indicating at least one louver is open."""
    return SimpleNamespace(
        private_sndStamp=utils.current_tai(),
        # One open, others closed
        positionActual=[0.0] * 10 + [100.0] + [0.0] * 23,
    )


def get_closed_louver_telemetry() -> SimpleNamespace:
    """Returns telemetry indicating all louvers are closed."""
    return SimpleNamespace(
        private_sndStamp=utils.current_tai(),
        positionActual=[0.0] * 34,
    )


class TestDomeModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.model = eas.dome_model.DomeModel(
            log=logging.getLogger(),
            dome_open_threshold=50.0,
        )
        # Always start with dome closed.
        await self.model.aperture_shutter_callback(get_closed_shutter_telemetry())
        await self.model.louvers_callback(get_closed_louver_telemetry())

    async def test_cancel_pending_events_sets_events(self) -> None:
        event = asyncio.Event()

        self.model.on_open.append(event)
        await self.model.aperture_shutter_callback(get_open_shutter_telemetry())

        # Event should have been set
        self.assertTrue(event.is_set())

    async def test_aperture_shutter_callback_schedules_delayed_events(self) -> None:
        event = asyncio.Event()
        self.model.on_open.append(event)

        # Open the dome
        await self.model.aperture_shutter_callback(get_open_shutter_telemetry())

        # on_open should be cleared
        self.assertFalse(self.model.on_open)

        # Event should be set
        await asyncio.wait_for(event.wait(), timeout=0.2)
        self.assertTrue(event.is_set())

    async def test_multiple_pending_events_all_fired_on_open(self) -> None:
        """All events registered in on_open should fire when the dome opens."""
        event1 = asyncio.Event()
        event2 = asyncio.Event()

        self.model.on_open.append(event1)
        self.model.on_open.append(event2)

        await self.model.aperture_shutter_callback(get_open_shutter_telemetry())
        await asyncio.sleep(STD_SLEEP)

        self.assertTrue(event1.is_set(), "event1 was not set")
        self.assertTrue(event2.is_set(), "event2 was not set")

    async def test_is_closed_true_when_shutter_and_louvers_closed(self) -> None:
        """Dome is closed when shutter AND all louvers are closed."""
        await self.model.aperture_shutter_callback(get_closed_shutter_telemetry())
        await self.model.louvers_callback(get_closed_louver_telemetry())

        self.assertTrue(self.model.is_closed)

    async def test_is_closed_false_when_shutter_open_even_if_louvers_closed(
        self,
    ) -> None:
        """Dome is open if shutter is open, regardless of louvers."""
        await self.model.aperture_shutter_callback(get_open_shutter_telemetry())
        await self.model.louvers_callback(get_closed_louver_telemetry())

        self.assertFalse(self.model.is_closed)

    async def test_is_closed_false_when_any_louver_open_even_if_shutter_closed(
        self,
    ) -> None:
        """Dome is open if any louver is open, even with shutter closed."""
        await self.model.aperture_shutter_callback(get_closed_shutter_telemetry())
        await self.model.louvers_callback(get_open_louver_telemetry())

        self.assertFalse(self.model.is_closed)

    async def test_is_closed_none_when_louver_telemetry_missing(self) -> None:
        """State is unknown when louver telemetry is missing."""
        self.model.louvers_telemetry = None
        await self.model.aperture_shutter_callback(get_closed_shutter_telemetry())

        self.assertIsNone(self.model.is_closed)

    async def test_louvers_open_reports_any_louver_open(self) -> None:
        """`louvers_open` is true when any louver is at or above threshold.

        Nighttime louver control starts a fixed delay after the louvers open,
        so it needs a louver-only notion of "open" -- distinct from
        `is_closed`, which also accounts for the shutter.
        """
        await self.model.louvers_callback(get_closed_louver_telemetry())
        self.assertFalse(self.model.louvers_open)

        await self.model.louvers_callback(get_open_louver_telemetry())
        self.assertTrue(self.model.louvers_open)

    async def test_louvers_open_time_records_transition(self) -> None:
        """`louvers_open_time` stamps the closed->open louver transition.

        The timestamp is set once when the louvers open, held steady while they
        stay open so the venting delay measures from the opening rather than
        from the latest telemetry, and cleared when they close again.
        """
        self.assertIsNone(self.model.louvers_open_time)

        await self.model.louvers_callback(get_open_louver_telemetry())
        first_open_time = self.model.louvers_open_time
        self.assertIsNotNone(first_open_time)

        await self.model.louvers_callback(get_open_louver_telemetry())
        self.assertEqual(self.model.louvers_open_time, first_open_time)

        await self.model.louvers_callback(get_closed_louver_telemetry())
        self.assertIsNone(self.model.louvers_open_time)

    async def test_is_closed_none_when_shutter_telemetry_missing(self) -> None:
        """State is unknown when shutter telemetry is missing."""
        self.model.aperture_shutter_telemetry = None
        await self.model.louvers_callback(get_closed_louver_telemetry())

        self.assertIsNone(self.model.is_closed)
