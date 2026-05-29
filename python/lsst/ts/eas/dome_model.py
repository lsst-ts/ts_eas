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
import logging
from collections import deque
from typing import Any, Callable

import yaml
from astropy import units as u
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time

from lsst.ts import salobj, utils

try:
    from lsst.ts.xml.tables.mtdome import LouverTable
except ImportError:
    # TODO: OSW-2359 Remove this backward compatibility once the louver table
    # is available in the ts_xml conda package.
    from .louver_table import LouverTable

from .cmdwrapper import close_command_tasks, command_wrapper
from .diurnal_timer import OBSERVATORY_LOCATION

DORMANT_TIME = 30  # Time to wait while sleeping, seconds
MAX_TELEMETRY_AGE = 300  # Time at which apertureShutter telemetry expires, seconds

# Minimum dome-azimuth change (degrees) that retriggers louver adjustment
# from the azimuth callback.
AZIMUTH_CHANGE_THRESHOLD = 0.1

# Atmospheric pressure used in the sun-altitude refraction correction.
SUN_ALTITUDE_PRESSURE = 700 * u.hPa


# Number of degrees in a complete circle.
CIRCLE = 360.0


class DomeModel:
    """A model for MTDome.

    Track whether and when the dome has been opened, and adjust louver
    positions based on the sun azimuth.

    Parameters
    ----------
    log : `~logging.Logger`
        A logger for log messages.
    dome_open_threshold : `float`
        Percent opening of a dome slit or louver beyond which the dome is
        considered "open."
    louver_sun_angle : `float`
        Minimum angular separation (degrees) between a louver's azimuth and
        the sun's azimuth at which the louver is considered to be facing the
        sun.
    louver_exposed_command : `float`
        Commanded percent-open position for a louver that faces the sun.
    louver_shaded_command : `float`
        Commanded percent-open position for a louver that does not face the
        sun.
    sun_altitude_threshold : `float`
        Sun altitude (degrees) above which louver positions are adjusted
        based on sun azimuth.
    dome_remote : `~lsst.ts.salobj.Remote`
        SAL remote for the MTDome CSC. Used to send louver commands.
    features_to_disable : `list` [`str`]
        List of feature names to disable. If ``"day_louvers"`` is present,
        sun position is not computed and louvers are not adjusted during the
        day.
    allow_send : `Callable` [[], `bool`] | None, optional
        Callable that returns ``True`` when commands may be sent (i.e., when
        EAS is in the ENABLED state). If ``None``, commands are always
        permitted.
    """

    def __init__(
        self,
        *,
        log: logging.Logger,
        dome_open_threshold: float,
        louver_sun_angle: float,
        louver_exposed_command: float,
        louver_shaded_command: float,
        sun_altitude_threshold: float,
        dome_remote: salobj.Remote,
        features_to_disable: list[str],
        allow_send: Callable[[], bool] | None = None,
    ) -> None:
        self.monitor_start_event = asyncio.Event()

        # Most recent tel_apertureShutter
        self.aperture_shutter_telemetry: salobj.BaseMsgType | None = None

        # Most recent tel_louvers
        self.louvers_telemetry: salobj.BaseMsgType | None = None

        # Most recent actual azimuth of dome (degrees east of north)
        self.dome_azimuth: float | None = None

        # Minimum angle between louver azimuth and sun azimuth (degrees)
        # at which a louver is considered to be "facing" the sun.
        self.louver_sun_angle: float = louver_sun_angle

        # Desired command position for a louver facing the sun.
        self.louver_exposed_command: float = louver_exposed_command

        # Desired command position for a louver that is not facing the sun.
        self.louver_shaded_command: float = louver_shaded_command

        # Sun altitude (degrees) above which louvers are adjusted.
        self.sun_altitude_threshold: float = sun_altitude_threshold

        self.on_open: deque[asyncio.Event] = deque()
        self.was_closed: bool | None = None

        self.log = log
        self.dome_open_threshold = dome_open_threshold
        self.dome_remote = dome_remote
        self.features_to_disable = features_to_disable
        self.allow_send = allow_send

    @classmethod
    def get_config_schema(cls) -> str:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for Dome EAS configuration.
type: object
properties:
  dome_open_threshold:
    description: Percent opening of a dome slit or louver beyond which the dome is considered "open."
    type: number
    default: 50.0
  louver_sun_angle:
    description: >-
      Minimum angular separation (degrees) between a louver's azimuth and the
      sun's azimuth at which the louver is considered to be facing the sun.
    type: number
    default: 60.0
  louver_exposed_command:
    description: Commanded percent-open position for a louver that faces the sun.
    type: number
    default: 50.0
  louver_shaded_command:
    description: Commanded percent-open position for a louver that does not face the sun.
    type: number
    default: 100.0
  sun_altitude_threshold:
    description: >-
      Sun altitude (degrees) above which louver positions are adjusted based
      on sun azimuth.
    type: number
    default: -1.0
required:
  - dome_open_threshold
  - louver_sun_angle
  - louver_exposed_command
  - louver_shaded_command
additionalProperties: false
"""
        )

    async def aperture_shutter_callback(self, aperture_shutter_telemetry: salobj.BaseMsgType) -> None:
        """Callback for MTDome.tel_apertureShutter.

        Data from this telemetry item is used to determine whether the dome
        is opened or closed.

        Parameters
        ----------
        aperture_shutter_telemetry: `~lsst.ts.salobj.BaseMsgType`
            A newly received apertureShutter telemetry item.
        """
        self.aperture_shutter_telemetry = aperture_shutter_telemetry
        self.refresh_telemetry()

    async def louvers_callback(self, louvers_telemetry: salobj.BaseMsgType) -> None:
        self.louvers_telemetry = louvers_telemetry
        self.refresh_telemetry()

    async def azimuth_callback(self, azimuth_telemetry: salobj.BaseMsgType) -> None:
        """Callback for MTDome.tel_azimuth.

        Data from this telemetry item is used to track the current dome
        azimuth, which is required to compute per-louver sun angles. When
        the dome azimuth changes by more than `AZIMUTH_CHANGE_THRESHOLD`
        (or is being read for the first time), louver positions are
        recomputed against the current sun position.

        Parameters
        ----------
        azimuth_telemetry : `~lsst.ts.salobj.BaseMsgType`
            A newly received azimuth telemetry item.
        """
        new_azimuth = azimuth_telemetry.positionActual
        previous_azimuth = self.dome_azimuth
        self.dome_azimuth = new_azimuth

        if previous_azimuth is None or abs(new_azimuth - previous_azimuth) > AZIMUTH_CHANGE_THRESHOLD:
            await self.update_louvers_for_sun()

    def refresh_telemetry(self) -> None:
        is_closed = self.is_closed

        if self.was_closed is not False and is_closed is False:
            events_to_signal = list(self.on_open)
            self.on_open.clear()

            for event in events_to_signal:
                event.set()

        self.was_closed = is_closed

    def set_pending_events(self) -> None:
        """Sets all events in the `on_open` deque.

        Any events waiting to be set will be set at this time.
        This gives waiting coroutines an opportunity to close
        gracefully.
        """
        events_to_signal = list(self.on_open)
        self.on_open.clear()

        for event in events_to_signal:
            event.set()

    @property
    def is_closed(self) -> bool | None:
        """Return true if the dome is currently closed.

        If the current state of the dome is unknown, None is returned.
        """
        if self.aperture_shutter_telemetry is None or self.louvers_telemetry is None:
            return None

        send_timestamp = min(
            self.aperture_shutter_telemetry.private_sndStamp,
            self.louvers_telemetry.private_sndStamp,
        )
        telemetry_age = utils.current_tai() - send_timestamp
        if telemetry_age > MAX_TELEMETRY_AGE:
            return None

        shutters_closed = (
            self.aperture_shutter_telemetry.positionActual[0] < self.dome_open_threshold
            and self.aperture_shutter_telemetry.positionActual[1] < self.dome_open_threshold
        )
        louvers_closed = all(
            position < self.dome_open_threshold for position in self.louvers_telemetry.positionActual
        )

        return shutters_closed and louvers_closed

    @command_wrapper(remote_attr="dome_remote", command_attr="cmd_setLouvers")
    async def set_louvers(self, position: list[float]) -> dict[str, Any] | None:
        """Send a setLouvers command to the MTDome CSC.

        Parameters
        ----------
        position : `list` [`float`]
            Desired percent-open position for each of the 34 louvers.
            A value of 0 is fully closed, 100 is fully open, and -1
            means do not move the louver.

        Returns
        -------
        `dict` [`str`, `Any`] | None
            Keyword arguments forwarded to ``cmd_setLouvers.set_start``.
        """
        return {"position": position}

    async def adjust_louvers(self, sun_azimuth: float) -> None:
        """Adjust positions of louvers.

        If a louver is actively being commanded (`positionCommanded` >= 0) then
        the commanded position of the louver should be adjusted based on the
        azimuth of the sun: if the louvers face the sun (within
        `louver_sun_angle`) its commanded position should be set to
        `exposed_lover_command`. If the louver does not face the sun, its
        commanded position should be set to `shaded_louver_command` If the
        louver is not commanded, it should remain uncommanded.

        Parameters
        ----------
        sun_azimuth : `float`
            Current sun azimuth in degrees, north = 0, east = 90.
        """
        if self.louvers_telemetry is None or self.dome_azimuth is None:
            return

        louver_azimuth = [(self.dome_azimuth + louver.azimuth) % CIRCLE for louver in LouverTable]
        sun_distance = [
            min((sun_azimuth - az) % CIRCLE, (az - sun_azimuth) % CIRCLE) for az in louver_azimuth
        ]
        louver_command = [
            (
                -1.0
                if cmd < 0
                else self.louver_exposed_command
                if sd < self.louver_sun_angle
                else self.louver_shaded_command
            )
            for cmd, sd in zip(self.louvers_telemetry.positionCommanded[: len(LouverTable)], sun_distance)
        ]
        await self.set_louvers(position=louver_command)

    async def update_louvers_for_sun(self) -> None:
        """Adjust louvers against the current sun position.

        Computes the sun's current altitude and azimuth at the observatory
        location and, if the sun is above `sun_altitude_threshold` and the
        ``day_louvers`` feature is not disabled, calls `adjust_louvers`
        with the sun azimuth.
        """
        if "day_louvers" in self.features_to_disable:
            return

        t = Time.now()
        sun = get_sun(t)
        altaz = sun.transform_to(
            AltAz(
                obstime=t,
                location=OBSERVATORY_LOCATION,
                pressure=SUN_ALTITUDE_PRESSURE,
            )
        )
        if altaz.alt.deg > self.sun_altitude_threshold:
            await self.adjust_louvers(altaz.az.deg)

    async def monitor(self) -> None:
        """Monitor the sun position and adjust louvers accordingly.

        Periodically checks whether the sun is above the horizon. When it
        is, `adjust_louvers` is called with the current sun azimuth so
        that each active louver is commanded to an appropriate position.

        Signals `monitor_start_event` before entering the polling loop so
        that the EAS CSC can synchronize startup across all monitor tasks.
        """
        self.log.debug("DomeModel.monitor")

        self.monitor_start_event.set()
        try:
            while True:
                try:
                    await self.update_louvers_for_sun()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.log.exception("In DomeModel louver control loop.")

                await asyncio.sleep(DORMANT_TIME)
        finally:
            self.monitor_start_event.clear()

    async def close(self) -> None:
        """Cancel any in-flight command tasks."""
        await close_command_tasks(self)
