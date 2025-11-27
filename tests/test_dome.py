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
        self.model = eas.dome_model.DomeModel()
        # Always start with dome closed.
        await self.model.aperture_shutter_callback(get_closed_shutter_telemetry())
        await self.model.louvers_callback(get_closed_louver_telemetry())

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

    async def test_is_closed_none_when_shutter_telemetry_missing(self) -> None:
        """State is unknown when shutter telemetry is missing."""
        self.model.aperture_shutter_telemetry = None
        await self.model.louvers_callback(get_closed_louver_telemetry())

        self.assertIsNone(self.model.is_closed)
