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
import heapq
import pathlib
import unittest
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from astropy.time import Time
from astropy.utils import iers
from lsst.ts.eas.diurnal_timer import DiurnalTimer

TEST_DIR = pathlib.Path(__file__).parent


class TestDiurnalTimer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Disable IERS
        iers.conf.auto_max_age = None

        # Load twilight times from file as timezone-aware datetimes
        self.tz = ZoneInfo("America/Santiago")
        self.twilight_times: list[datetime] = []
        with open(TEST_DIR / "twilight_2025_isot.txt") as f:
            for line in f:
                dt = datetime.fromisoformat(line.strip()).replace(tzinfo=timezone.utc)
                self.twilight_times.append(dt)

        # Generate all local noon times in 2025. Careful how you handle
        # daylight saving time :-P
        self.noon_times = [
            datetime.combine(
                date(2025, 1, 1) + timedelta(days=i), time(12, 0), tzinfo=self.tz
            )
            for i in range(365)
        ]

        self.noon_seen: list[datetime] = []
        self.twilight_seen: list[datetime] = []

        self.fake_now: datetime = datetime(2024, 12, 31, 21, tzinfo=timezone.utc)
        self.sleep_queue: list[tuple[datetime, asyncio.Event]] = []
        self._sleep_lock = asyncio.Lock()
        self._real_sleep = asyncio.sleep
        self.ready_to_sleep = asyncio.Event()

        self.noon_watcher_finished = False
        self.twilight_watcher_finished = False

    def fake_time_now(self) -> Time:
        return Time(self.fake_now)

    async def fake_sleep(self, seconds: float) -> None:
        print(f"fake_sleep({seconds})")

        # Schedule the next wakeup
        async with self._sleep_lock:
            # Set the time to wake up:
            wake_time = self.fake_now + timedelta(seconds=seconds)
            wake_event = asyncio.Event()

            heapq.heappush(self.sleep_queue, (wake_time, wake_event))

        # Wait to be woken up...
        await wake_event.wait()

    async def _scheduler(self) -> None:
        """Drive simulated time forward by waking up sleepers in order."""
        # Start waking up sleep
        while not (self.noon_watcher_finished and self.twilight_watcher_finished):
            await self._real_sleep(0.01)  # Give other things a chance to run
            async with self._sleep_lock:
                if len(self.sleep_queue) == 0:
                    continue

                self.sleep_queue.sort()
                wake_time, wake_event = heapq.heappop(self.sleep_queue)
                self.fake_now = wake_time
                wake_event.set()  # Ready to wake up

                # Wait for processing...
                await self.ready_to_sleep.wait()
                self.ready_to_sleep.clear()

    def assert_datetime_lists_close(
        self,
        expected: list[datetime],
        actual: list[datetime],
        tolerance_seconds: float = 60,
        label: str = "",
    ) -> None:
        self.assertEqual(
            len(expected),
            len(actual),
            f"{label} length mismatch: expected {len(expected)}, got {len(actual)}",
        )

        for i, (e, a) in enumerate(zip(expected, actual)):
            delta = abs((e - a).total_seconds())
            self.assertLessEqual(
                delta,
                tolerance_seconds,
                f"{label} mismatch at index {i}: expected {e}, got {a}, delta = {delta:.2f} seconds",
            )

    async def test_diurnaltimer_notifies_all(self) -> None:
        timer = DiurnalTimer(sun_altitude="astronomical")

        async def watch_noon() -> None:
            while not self.noon_watcher_finished:
                await self._real_sleep(0.001)
                self.ready_to_sleep.set()
                if len(self.noon_seen) < len(self.noon_times):
                    async with timer.noon_condition:
                        await timer.noon_condition.wait()
                    if not timer.is_running:
                        return
                    self.noon_seen.append(self.fake_now)
                else:
                    self.noon_watcher_finished = True

        async def watch_twilight() -> None:
            while not self.twilight_watcher_finished:
                await self._real_sleep(0.001)
                self.ready_to_sleep.set()
                if len(self.twilight_seen) < len(self.twilight_times):
                    async with timer.twilight_condition:
                        await timer.twilight_condition.wait()
                    if not timer.is_running:
                        return
                    self.twilight_seen.append(self.fake_now)
                else:
                    self.twilight_watcher_finished = True

        with patch("astropy.time.Time.now", side_effect=self.fake_time_now), patch(
            "asyncio.sleep", side_effect=self.fake_sleep
        ):
            async with timer:
                await self._real_sleep(1)
                await asyncio.wait_for(
                    asyncio.gather(
                        watch_noon(),
                        watch_twilight(),
                        self._scheduler(),
                    ),
                    timeout=600,
                )

        self.assert_datetime_lists_close(
            self.noon_times, self.noon_seen, label="Noon event"
        )
        self.assert_datetime_lists_close(
            self.twilight_times, self.twilight_seen, label="Twilight event"
        )
