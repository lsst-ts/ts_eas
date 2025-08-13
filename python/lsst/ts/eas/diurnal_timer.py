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

# Accounting for the diameter of the solar disk, average atmospheric
# refraction at the horizon, and horizon dip at the elevation of
# Rubin Observatory, the average elevation of the sun at the moment
# of sunset should be roughly 2.22 degrees below horizon.
SOLAR_ELEVATION_AT_SUNSET = -2.22


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


def get_crossing_time(target_alt: float, going_up: bool = False) -> Time:
    """Computes the time when the sun will reach target_alt.

    Parameters
    ----------
    target_alt : float
        The sun altitude at the twilight time to search for, in degrees.
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
    t0 = Time.now() + 1 * u.s
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
            raise RuntimeError("sun_altitude not in range -90 to 0")

        self.sun_altitude = sun_altitude
        self.twilight_time: Time | None = None

        self._noon_loop_ready = asyncio.Event()
        self._twilight_loop_ready = asyncio.Event()

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
            self._noon_loop_ready.set()

            await asyncio.sleep(wait_seconds)

            async with self.noon_condition:
                self.noon_condition.notify_all()

    async def _run_twilight_loop(self) -> None:
        """Notifies twilight_condition whenever it's noon.

        This method loops, notifying for twilight_condition each day.
        """
        while self.is_running:
            # Search for time of twilight and sunrise in the next 25 hours.
            # Not necessarily valid in the polar regions, but fine for Rubin.
            self.twilight_time = get_crossing_time(self.sun_altitude, going_up=False)
            self.sunrise_time = get_crossing_time(
                SOLAR_ELEVATION_AT_SUNSET, going_up=True
            )
            wait_seconds = self.seconds_until_twilight(Time.now())
            self._twilight_loop_ready.set()

            await asyncio.sleep(wait_seconds)

            async with self.twilight_condition:
                self.twilight_condition.notify_all()

    async def start(self) -> None:
        """Starts the timers and begins notifying for the conditions."""
        if self.is_running:
            return
        self.is_running = True
        self._tasks = [
            asyncio.create_task(self._run_noon_loop()),
            asyncio.create_task(self._run_twilight_loop()),
        ]

        await asyncio.gather(
            self._noon_loop_ready.wait(),
            self._twilight_loop_ready.wait(),
        )

    async def stop(self) -> None:
        """Stops the timers and discontinues notificaitons."""
        if not self.is_running:
            return
        self.is_running = False

        self._noon_loop_ready.clear()
        self._twilight_loop_ready.clear()

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
        t = (self.twilight_time - time).sec
        if t < 0 or t > 25 * 3600:
            raise ValueError("Time until twilight unexpectedly out of range: {t}")
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
