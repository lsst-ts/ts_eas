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
from typing import Any, Optional, Type
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from scipy.optimize import brentq

__all__ = ["DiurnalTimer"]

OBSERVATORY_LOCATION = EarthLocation.of_site("Cerro Pachon")
OBSERVATORY_TIME_ZONE = ZoneInfo("America/Santiago")


def get_local_noon_time() -> Time:
    """Returns the next local noon as an astropy.time.Time object.

    If the current time is past local noon today, returns noon tomorrow.
    The difference from noon to noon is 24 hours except when there's
    a time change because of daylight saving time.

    Returns
    -------
    Time
        The next occurrence of local noon in observatory time zone.
    """

    now = Time.now()

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


def get_crossing_time(target_alt: float) -> Time:
    """Computes the time when the sun will reach target_alt.

    Parameters
    ----------
    target_alt : float
        The sun altitude at the twilight time to search for, in degrees.

    Returns
    -------
    Time
        The time when the sun will next cross target_alt.
    """

    def f(t_sec: float) -> float:
        t = Time(t_sec, format="unix")
        return get_sun_altitude_deg(t) - target_alt

    # Start by searching for a good window for the root finder.
    t0 = Time.now() + 1 * u.s
    sun_altitude_before = get_sun_altitude_deg(t0)
    for i in range(25):  # Start by searching the next 25 hours.
        t1 = t0 + 1 * u.hour
        sun_altitude_after = get_sun_altitude_deg(t1)

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
    """A class representing a timer to wait for noon and for evening twilight.

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
            raise RuntimeError("sun_altitude not in range -90 to 90")

        self.sun_altitude = sun_altitude

        self.noon_condition = asyncio.Condition()
        self.twilight_condition = asyncio.Condition()

        self.is_running = False
        self._tasks: list[asyncio.Task] = []

    async def _run_noon_loop(self) -> None:
        """Notifies noon_condition whenever it's noon.

        This method loops, notifying for noon_condition each day.
        """
        while self.is_running:
            local_noon = get_local_noon_time()
            wait_seconds = (local_noon - Time.now()).sec
            await asyncio.sleep(wait_seconds)

            async with self.noon_condition:
                self.noon_condition.notify_all()

    async def _run_twilight_loop(self) -> None:
        """Notifies twilight_condition whenever it's noon.

        This method loops, notifying for twilight_condition each day.
        """
        while self.is_running:
            # Search for time of twilight in the next 25 hours. Not necessarily
            # valid in the polar regions, but fine for
            # Rubin.
            twilight_time = get_crossing_time(self.sun_altitude)
            wait_seconds = (twilight_time - Time.now()).sec
            await asyncio.sleep(wait_seconds)

            async with self.twilight_condition:
                self.twilight_condition.notify_all()

    def start(self) -> None:
        """Starts the timers and begins notifying for the conditions."""
        if self.is_running:
            return
        self.is_running = True
        self._tasks = [
            asyncio.create_task(self._run_noon_loop()),
            asyncio.create_task(self._run_twilight_loop()),
        ]

    async def stop(self) -> None:
        """Stops the timers and discontinues notificaitons."""
        if not self.is_running:
            return
        self.is_running = False

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

    async def __aenter__(self) -> "DiurnalTimer":
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        await self.stop()
