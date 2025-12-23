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
from typing import Any, Type
from unittest.mock import _patch, patch
from zoneinfo import ZoneInfo

from astropy.time import Time
from astropy.utils import iers

from lsst.ts.eas.diurnal_timer import DiurnalTimer

TEST_DIR = pathlib.Path(__file__).parent


class SimulatedClock:
    """Async test harness that simulates time for asyncio code.

    Parameters
    ----------
    start : datetime
        Initial simulated time.
    watchers : int
        Number of users of the clock. This number of calls must be made to
        `watcher_ready()`.
    """

    def __init__(
        self,
        start: datetime,
        *,
        watchers: int,
    ) -> None:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        self.now: datetime = start

        # Min-heap of (wake_time, wake_event)
        self._heap: list[tuple[datetime, asyncio.Event]] = []
        self._heap_lock = asyncio.Lock()

        # Real sleep we can use to yield without changing simulated time
        self._real_sleep = asyncio.sleep

        # Patches
        self._sleep_patcher: _patch | None = None
        self._time_now_patcher: _patch | None = None

        # Scheduler task + lifecycle
        self._scheduler_task: asyncio.Task | None = None
        self.is_running = False

        # yield duration for cooperative scheduling
        self._yield = 0.0

        # Track how many watchers are ready
        self._waiting_count = watchers
        self._waiting_lock = asyncio.Lock()
        self._all_ready = asyncio.Event()
        if self._waiting_count == 0:
            self._all_ready.set()

    async def __aenter__(self) -> "SimulatedClock":
        # Patch asyncio.sleep to our fake
        self._sleep_patcher = patch("asyncio.sleep", side_effect=self._fake_sleep)
        self._sleep_patcher.start()

        self._time_now_patcher = patch("astropy.time.Time.now", side_effect=self.time_now)
        self._time_now_patcher.start()

        # Start scheduler
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler(), name="SimClockScheduler")
        return self

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        # Stop scheduler
        self.is_running = False
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass  # Expected

        # Unpatch
        if self._sleep_patcher is not None:
            self._sleep_patcher.stop()
        if self._time_now_patcher is not None:
            self._time_now_patcher.stop()

        # Drain heap: wake any stragglers so tests don't hang
        async with self._heap_lock:
            for _, ev in self._heap:
                ev.set()
            self._heap.clear()

    async def watcher_ready(self) -> None:
        if self._all_ready.is_set():
            return
        async with self._waiting_lock:
            if not self._all_ready.is_set():
                self._waiting_count -= 1
                if self._waiting_count <= 0:
                    self._all_ready.set()

    def time_now(self) -> Time:
        """Replacement for astropy.time.Time.now()"""
        return Time(self.now)

    async def _fake_sleep(self, seconds: float) -> None:
        """Replacement for asyncio.sleep that enqueues a wake on the heap."""
        if seconds < 0:
            seconds = 0.0
        wake_ev = asyncio.Event()
        async with self._heap_lock:
            wake_time = self.now + timedelta(seconds=seconds)
            heapq.heappush(self._heap, (wake_time, wake_ev))
        # Wait for the scheduler to pop and set your event
        await wake_ev.wait()

    async def _scheduler(self) -> None:
        """Advance simulated time in discrete 'ticks'."""
        await self._all_ready.wait()

        while self.is_running:
            # Wait until someone is sleeping
            while self.is_running and not self._heap:
                await self._real_sleep(1)

            if not self.is_running:
                break

            # Pop the earliest batch atomically
            batch: list[tuple[datetime, asyncio.Event]] = []
            async with self._heap_lock:
                if not self._heap:
                    continue
                earliest = self._heap[0][0]
                while self._heap and self._heap[0][0] == earliest:
                    batch.append(heapq.heappop(self._heap))
                # Advance simulated time to the wake instant
                self.now = earliest

            # Wake all sleepers scheduled for this instant
            for _, ev in batch:
                ev.set()

            # Give awakened tasks a turn (compute, notify, enqueue next sleeps)
            await self._real_sleep(0.001)


class TestDiurnalTimer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Disable IERS
        iers.conf.auto_max_age = None

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
        # Load twilight times from file as timezone-aware datetimes
        self.tz = ZoneInfo("America/Santiago")
        self.twilight_times: list[datetime] = []
        self.sunrise_times: list[datetime] = []
        with open(TEST_DIR / "twilight_2025_isot.txt") as f:
            for line in f:
                dt = datetime.fromisoformat(line.strip()).replace(tzinfo=timezone.utc)
                self.twilight_times.append(dt)
        with open(TEST_DIR / "sunrise_2025_isot.txt") as f:
            for line in f:
                dt = datetime.fromisoformat(line.strip()).replace(tzinfo=timezone.utc)
                self.sunrise_times.append(dt)

        # Generate all local noon times in 2025. Careful how you handle
        # daylight saving time :-P
        self.noon_times = [
            datetime.combine(date(2025, 1, 1) + timedelta(days=i), time(12, 0), tzinfo=self.tz)
            for i in range(365)
        ]

        self.noon_seen: list[datetime] = []
        self.twilight_seen: list[datetime] = []
        self.sunrise_seen: list[datetime] = []

        timer = DiurnalTimer(sun_altitude="astronomical")

        async def watch_noon(clock: SimulatedClock) -> None:
            while clock.is_running:
                async with timer.noon_condition:
                    await clock.watcher_ready()
                    await timer.noon_condition.wait()

                self.noon_seen.append(clock.now)
                if len(self.noon_seen) >= 365:
                    return

        async def watch_twilight(clock: SimulatedClock) -> None:
            while clock.is_running:
                async with timer.twilight_condition:
                    await clock.watcher_ready()
                    await timer.twilight_condition.wait()

                self.twilight_seen.append(clock.now)
                if len(self.twilight_seen) >= 365:
                    return

        async def watch_sunrise(clock: SimulatedClock) -> None:
            while clock.is_running:
                async with timer.sunrise_condition:
                    await clock.watcher_ready()
                    await timer.sunrise_condition.wait()

                self.sunrise_seen.append(clock.now)
                if len(self.sunrise_seen) >= 365:
                    return

        start_time = datetime(2024, 12, 31, 21, tzinfo=timezone.utc)
        async with SimulatedClock(start_time, watchers=3) as clock:
            async with timer:
                await asyncio.wait_for(
                    asyncio.gather(
                        watch_noon(clock),
                        watch_twilight(clock),
                        watch_sunrise(clock),
                    ),
                    timeout=600,
                )

        self.assert_datetime_lists_close(self.noon_times, self.noon_seen, label="Noon event")
        self.assert_datetime_lists_close(self.twilight_times, self.twilight_seen, label="Twilight event")
        self.assert_datetime_lists_close(self.sunrise_times, self.sunrise_seen, label="Sunrise event")

    async def test_is_night_start_in_daytime(self) -> None:
        timer = DiurnalTimer(sun_altitude="astronomical")
        cases = [
            ("2025-08-03 23:15:00", False, "just before twilight"),
            ("2025-08-04 01:00:00", True, "between twilight and midnight"),
            ("2025-08-04 11:15:00", True, "early morning before dawn"),
            ("2025-08-04 11:35:00", False, "just after dawn"),
        ]

        start_time = datetime(2025, 8, 3, 16, tzinfo=timezone.utc)
        async with SimulatedClock(start_time, watchers=1):
            async with timer:
                for when_str, expected, label in cases:
                    with self.subTest(label=label):
                        t = Time(when_str, scale="utc")
                        self.assertIs(timer.is_night(t), expected)

    async def test_is_night_start_at_night(self) -> None:
        timer = DiurnalTimer(sun_altitude="astronomical")
        cases = [
            ("2025-08-04 01:00:00", True, "between twilight and midnight"),
            ("2025-08-04 11:15:00", True, "early morning before dawn"),
            ("2025-08-04 11:35:00", False, "just after dawn"),
            ("2025-08-04 23:15:00", False, "just before twilight next day"),
            ("2025-08-04 23:45:00", True, "just after twilight next day"),
        ]

        start_time = datetime(2025, 8, 4, 0, tzinfo=timezone.utc)
        async with SimulatedClock(start_time, watchers=1):
            async with timer:
                for when_str, expected, label in cases:
                    with self.subTest(label=label):
                        t = Time(when_str, scale="utc")
                        self.assertIs(timer.is_night(t), expected)
