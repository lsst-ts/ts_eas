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
import datetime
from typing import Any, Callable, Optional, Type
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf
from scipy.optimize import brentq

__all__ = ["DiurnalTimer"]

iers_conf.auto_download = False
iers_conf.iers_degraded_accuracy = "ignore"
iers.IERS_Auto.iers_table = iers.IERS_B.open()
iers.conf.auto_max_age = None

OBSERVATORY_LOCATION = EarthLocation(
    lat=-30.24074167 * u.deg, lon=-70.7366833 * u.deg, height=2750 * u.m
)
OBSERVATORY_TIME_ZONE = ZoneInfo("America/Santiago")

# Standard definition of sunrise / sunset with solar elevation -0.833 degrees.
SOLAR_ELEVATION_AT_SUNSET = -0.833


def get_local_noon_time() -> Time:
    """Returns the next local noon as an astropy.time.Time object.

    If the current time is past local noon today, returns noon tomorrow.
    The difference from noon to noon is 24 hours except when there's
    a time change because of daylight saving time.

    Returns
    -------
    `~astropy.time.Time`
        The next occurrence of local noon in observatory time zone.
    """

    now = Time.now()
    now += 1 * u.min

    # Convert current time to local datetime with correct DST info
    local_now = now.to_datetime(timezone=OBSERVATORY_TIME_ZONE)

    # Construct a local datetime at 12:00 (with correct DST handling)
    local_noon = datetime.datetime.combine(
        local_now.date(), datetime.time(12, 0), tzinfo=OBSERVATORY_TIME_ZONE
    ).astimezone(OBSERVATORY_TIME_ZONE)

    local_noon_time = Time(local_noon)

    # If it's already past noon today, use tomorrow
    if now >= local_noon_time:
        next_day = local_now.date() + datetime.timedelta(days=1)
        local_noon = datetime.datetime.combine(
            next_day, datetime.time(12, 0), tzinfo=OBSERVATORY_TIME_ZONE
        ).astimezone(OBSERVATORY_TIME_ZONE)
        local_noon_time = Time(local_noon)

    return local_noon_time


def get_sun_altitude_deg(t: Time) -> float:
    """Returns the sun's altitude at the given time.

    The altitude is based on the specified observatory
    location and does not account for atmospheric
    refraction.

    Returns
    -------
    float
        The sun's altitude (deg) above the horizon at time t.
    """
    sun = get_sun(t)
    altaz = sun.transform_to(AltAz(obstime=t, location=OBSERVATORY_LOCATION))
    return altaz.alt.deg


def get_crossing_time(
    target_alt: float, going_up: bool = False, search_from: Time | None = None
) -> Time:
    """Computes the time when the sun will reach target_alt.

    Parameters
    ----------
    target_alt : float
        The sun altitude at the twilight time to search for, in degrees.
    search_from : `~astropy.time.Time` | None
        The time to start searching from. The returned value will be
        the first instance of the sun crossing the target altitude
        after this time.
    going_up : bool
        True if searching for the crossing of the target altitude
        while the sun altitude is increasing ("sun is rising"),
        of False if the sun should be decreasing in altitude at the
        returned time.

    Returns
    -------
    `~astropy.time.Time`
        The time when the sun will next cross target_alt.
    """

    def f(t_sec: float) -> float:
        t = Time(t_sec, format="unix")
        return get_sun_altitude_deg(t) - target_alt

    # Start by searching for a good window for the root finder.
    if search_from is None:
        search_from = Time.now() + 1 * u.min
    t0 = search_from + 1 * u.s
    sun_altitude_before = get_sun_altitude_deg(t0)
    for i in range(25):  # Start by searching the next 25 hours.
        t1 = t0 + 1 * u.hour
        sun_altitude_after = get_sun_altitude_deg(t1)

        if going_up:
            if sun_altitude_before < target_alt and sun_altitude_after > target_alt:
                break
        else:
            if sun_altitude_before > target_alt and sun_altitude_after < target_alt:
                break

        # Nope, keep looking
        sun_altitude_before = sun_altitude_after
        t0 = t1

    t0 = t0.unix
    t1 = t1.unix

    t_cross = brentq(f, t0, t1)
    return Time(t_cross, format="unix")


class DiurnalTimer:
    """A class representing a timer for noon, evening twilight, and sunrise.

    Parameters
    ----------
    sun_altitude : str | float
        The sun altitude (degrees) below the horizon that is considered the
        end of twilight. This can be a number between 0 and -90 or it can
        be one of "civil" (-6 degrees), "nautical" (-12 degrees), or
        "astronomical"(-18 degrees).
    """

    def __init__(self, sun_altitude: str | float = -18.0):
        if sun_altitude == "civil":
            sun_altitude = -6.0
        elif sun_altitude == "nautical":
            sun_altitude = -12.0
        elif sun_altitude == "astronomical":
            sun_altitude = -18.0
        elif isinstance(sun_altitude, str):
            raise RuntimeError(
                "Allowed string values for "
                "sun_altitude: civil, nautical, astronomical"
            )

        if not (-90 <= sun_altitude <= 0):
            raise RuntimeError("sun_altitude not in range -90 to 0")

        self.sun_altitude = sun_altitude
        self.twilight_time: Time | None = None
        self.sunrise_time: Time | None = None

        self._noon_loop_ready = asyncio.Event()
        self._twilight_loop_ready = asyncio.Event()
        self._sunrise_loop_ready = asyncio.Event()

        self.noon_condition = asyncio.Condition()
        self.twilight_condition = asyncio.Condition()
        self.sunrise_condition = asyncio.Condition()

        self.is_running = False
        self._tasks: list[asyncio.Task] = []

    @staticmethod
    def _seconds_until(target: Time, now: Time | None = None) -> float:
        """Returns time from `now` until `target` in seconds.

        If the target has already passed, zero is returned to avoid
        passing a negative value to `sleep`.

        Parameters
        ----------
        target: `~astropy.time.Time`
            The future event that is to be timed.

        now: `~astropy.time.Time`
            The starting time, or None if `Time.now()` should be used.

        Returns
        -------
        float
            Number of seconds from `now` until `target`.
        """
        now = now or Time.now()
        dt = (target - now).sec
        return max(60.0, dt)

    async def _run_loop(
        self,
        *,
        name: str,
        compute_next: Callable[[], Time],
        set_time_attr: Callable[[Time], None],
        condition: asyncio.Condition,
        ready_event: asyncio.Event,
    ) -> None:
        """
        Background loop to schedule and notify a diurnal event.

        This coroutine repeatedly computes the next occurrence of an event
        (e.g., noon, twilight, sunrise), waits until that time, and then
        notifies all tasks waiting on the associated condition. The loop
        continues until the timer is stopped or the task is cancelled.

        Parameters
        ----------
        name : str
            A human-readable label for the loop.
        compute_next : Callable[[], `~astropy.time.Time`]
            Function that computes the next scheduled time for this event.
            Must return a `Time` strictly after the current time.
        set_time_attr : Callable[[`~astropy.time.Time`], None]
            Callback invoked with the computed time, to update the attribute.
        condition : asyncio.Condition
            Condition variable that is notified whenever the event occurs.
        ready_event : asyncio.Event
            Event set after the first successful computation.
        """
        while self.is_running:
            next_time = compute_next()
            set_time_attr(next_time)
            wait_seconds = self._seconds_until(next_time, Time.now())

            if not ready_event.is_set():
                ready_event.set()

            await asyncio.sleep(wait_seconds)

            async with condition:
                condition.notify_all()

    def _next_noon(self) -> Time:
        """Returns time of next noon."""
        return get_local_noon_time()

    def _next_twilight(self) -> Time:
        """Returns time of next end of evening twilight."""
        return get_crossing_time(self.sun_altitude, going_up=False)

    def _next_sunrise(self) -> Time:
        """Returns time of next sunrise."""
        return get_crossing_time(
            SOLAR_ELEVATION_AT_SUNSET,
            going_up=True,
        )

    async def start(self) -> None:
        """Starts the timers and begins notifying for the conditions."""
        if self.is_running:
            return

        self.is_running = True
        self._tasks = [
            asyncio.create_task(
                self._run_loop(
                    name="noon",
                    compute_next=self._next_noon,
                    set_time_attr=lambda t: None,  # no public noon attr
                    condition=self.noon_condition,
                    ready_event=self._noon_loop_ready,
                )
            ),
            asyncio.create_task(
                self._run_loop(
                    name="twilight",
                    compute_next=self._next_twilight,
                    set_time_attr=lambda t: setattr(self, "twilight_time", t),
                    condition=self.twilight_condition,
                    ready_event=self._twilight_loop_ready,
                )
            ),
            asyncio.create_task(
                self._run_loop(
                    name="sunrise",
                    compute_next=self._next_sunrise,
                    set_time_attr=lambda t: setattr(self, "sunrise_time", t),
                    condition=self.sunrise_condition,
                    ready_event=self._sunrise_loop_ready,
                )
            ),
        ]

        await asyncio.gather(
            self._noon_loop_ready.wait(),
            self._twilight_loop_ready.wait(),
            self._sunrise_loop_ready.wait(),
        )

    async def stop(self) -> None:
        """Stops the timers and discontinues notifications."""
        if not self.is_running:
            return
        self.is_running = False

        self._noon_loop_ready.clear()
        self._twilight_loop_ready.clear()
        self._sunrise_loop_ready.clear()

        # End tasks
        for task in self._tasks:
            task.cancel()

        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

        # One last notify so that we're not hanging.
        async with self.noon_condition:
            self.noon_condition.notify_all()

        async with self.twilight_condition:
            self.twilight_condition.notify_all()

        async with self.sunrise_condition:
            self.sunrise_condition.notify_all()

    def get_twilight_time(self, after: Time) -> Time:
        """Returns the time of next end of twilight for the given time.

        Parameters
        ----------
        after : `~astropy.time.Time`
            Time returned will be time corresponding to the
            end of twilight on or after this time.
        """
        return get_crossing_time(self.sun_altitude, going_up=False, search_from=after)

    def sun_altitude_at(self, t_utc: float) -> float:
        """Returns the sun altitude in degrees at the specified time.

        Parameters
        ----------
        t_utc : float
            The time of interest to look up sun altitude, specified
            in UTC seconds in the UNIX epoch.

        Returns
        -------
        float
            Elevation above the local observatory horizon of the center of the
            sun, in degrees.
        """
        return get_sun_altitude_deg(Time(t_utc, format="unix"))

    def seconds_until_twilight(self, time: Time) -> float:
        """Returns the number of seconds remaining until twilight.

        Parameters
        ----------
        time : `~astropy.time.Time`
            The time from which to calculate the time until twilight.

        Raises
        ------
        `ValueError`
            If the time calculated is negative, or is greater than
            (a little more than) one day.
        """
        if self.twilight_time is None:
            raise RuntimeError(
                "seconds_until_twilight called before initialization finished."
            )

        t = (self.twilight_time - time).sec
        if t < 0 or t > 25 * 3600:
            raise ValueError(f"Time until twilight unexpectedly out of range: {t}")
        return t

    def seconds_until_sunrise(self, time: Time) -> float:
        """Returns the number of seconds remaining until sunrise.

        Parameters
        ----------
        time : `~astropy.time.Time`
            The time from which to calculate the time until sunrise.

        Raises
        ------
        `ValueError`
            If the time calculated is negative, or is greater than
            (a little more than) one day.
        """
        if self.sunrise_time is None:
            raise RuntimeError(
                "seconds_until_sunrise called before initialization finished."
            )

        t = (self.sunrise_time - time).sec
        if t < 0 or t > 25 * 3600:
            raise ValueError(f"Time until sunrise unexpectedly out of range: {t}")
        return t

    def is_night(self, time: Time) -> bool:
        """Returns True if `time` is in the EAS-defined night.

        If the time is after twilight and before sunrise, it is
        considered to be night.

        Parameters
        ----------
        time : `~astropy.time.Time`
            The time to use in the determination.

        Returns
        -------
        bool
            True if and only if `time` is between the next twilight and
            sunrise.
        """
        if self.sunrise_time is None or self.twilight_time is None:
            raise RuntimeError("is_night called before initialization finished.")

        if self.sunrise_time > self.twilight_time:
            return self.twilight_time <= time <= self.sunrise_time
        else:
            return time < self.sunrise_time or time > self.twilight_time

    async def __aenter__(self) -> "DiurnalTimer":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        await self.stop()
