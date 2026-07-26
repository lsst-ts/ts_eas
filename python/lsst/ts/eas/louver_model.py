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

__all__ = ["LouverControlMode", "LouverModel"]

import asyncio
import enum
import logging
from typing import Any, Callable

import yaml
from astropy import units as u
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time

from lsst.ts import salobj

try:
    from lsst.ts.xml.tables.mtdome import LouverTable
except ImportError:
    # TODO: OSW-2359 Remove this backward compatibility once the louver table
    # is available in the ts_xml conda package.
    from .louver_table import LouverTable

from .cmdwrapper import close_command_tasks, command_wrapper
from .diurnal_timer import OBSERVATORY_LOCATION
from .dome_model import DomeModel

DORMANT_TIME = 30  # Time to wait while sleeping, seconds

# Minimum dome-azimuth change (degrees) that retriggers louver adjustment
# from the azimuth callback.
AZIMUTH_CHANGE_THRESHOLD = 0.1

# Tolerance (percent-open) within which a reported positionCommanded is
# considered equal to the position EAS last commanded. A larger difference is
# treated as a freshly sent command rather than EAS's own command echoed back.
LOUVER_COMMAND_TOLERANCE = 0.1

# Atmospheric pressure used in the sun-altitude refraction correction.
SUN_ALTITUDE_PRESSURE = 700 * u.hPa

# Number of degrees in a complete circle.
CIRCLE = 360.0


class LouverControlMode(enum.Enum):
    """The active source of louver setpoints.

    Louver setpoints come from exactly one place at a time. The control loop
    in `LouverModel.monitor` selects a mode on every pass and takes its
    setpoints from that mode alone, so there is a single path from sensor
    input to a ``setLouvers`` command.

    Attributes
    ----------
    IDLE
        No louver control. EAS leaves louver positions to the observer.
    DAYTIME_LOUVER_CONTROL
        Sun avoidance. A louver facing the sun is capped at
        `LouverModel.louver_exposed_command`.
    """

    IDLE = enum.auto()
    DAYTIME_LOUVER_CONTROL = enum.auto()


class LouverModel:
    """A model for MTDome louver control.

    Adjust louver positions according to the currently selected
    `LouverControlMode`. All louver commands originate in `monitor`; telemetry
    callbacks only ask the loop to run sooner, so there is one control loop
    rather than several competing sources of setpoints.

    Parameters
    ----------
    log : `~logging.Logger`
        A logger for log messages.
    dome_model : `DomeModel`
        The dome model, which owns MTDome louver telemetry.
    dome_remote : `~lsst.ts.salobj.Remote`
        SAL remote for the MTDome CSC. Used to send louver commands.
    louver_sun_angle : `float`
        Minimum angular separation (degrees) between a louver's azimuth and
        the sun's azimuth at which the louver is considered to be facing the
        sun.
    louver_exposed_command : `float`
        Maximum commanded percent-open position for a louver that faces the
        sun. A sun-facing louver is never opened beyond this value, nor beyond
        the position the observer commanded.
    sun_altitude_threshold : `float`
        Sun altitude (degrees) above which louver positions are adjusted
        based on sun azimuth.
    features_to_disable : `list` [`str`]
        List of feature names to disable. If ``"day_louvers"`` is present,
        louvers are not adjusted during the day.
    allow_send : `Callable` [[], `bool`] | None, optional
        Callable that returns ``True`` when commands may be sent (i.e., when
        EAS is in the ENABLED state). If ``None``, commands are always
        permitted.
    """

    def __init__(
        self,
        *,
        log: logging.Logger,
        dome_model: DomeModel,
        dome_remote: salobj.Remote,
        louver_sun_angle: float,
        louver_exposed_command: float,
        sun_altitude_threshold: float,
        features_to_disable: list[str],
        allow_send: Callable[[], bool] | None = None,
    ) -> None:
        self.monitor_start_event = asyncio.Event()

        # Set by telemetry callbacks to ask the control loop to run before its
        # cadence would otherwise expire. Callbacks never command louvers
        # themselves; they only wake the loop.
        self.update_requested = asyncio.Event()

        # The setpoint source currently in effect.
        self.mode = LouverControlMode.IDLE

        # Most recent actual azimuth of dome (degrees east of north)
        self.dome_azimuth: float | None = None

        # Minimum angle between louver azimuth and sun azimuth (degrees)
        # at which a louver is considered to be "facing" the sun.
        self.louver_sun_angle: float = louver_sun_angle

        # Maximum command position for a louver facing the sun.
        self.louver_exposed_command: float = louver_exposed_command

        # Sun altitude (degrees) above which louvers are adjusted.
        self.sun_altitude_threshold: float = sun_altitude_threshold

        # Observer-commanded position per louver. EAS caps a sun-facing louver
        # at `louver_exposed_command` but never opens a louver beyond what the
        # observer requested, so the observer's request must be remembered
        # separately. This list stores observed louver commands issued by the
        # operator so that they can be re-issued when the louver returns to a
        # shaded position. If no command has been observed, the value is None.
        self.louver_operator_command: list[float] | None = None

        # Position EAS last intended for each louver (with -1 "do not move"
        # resolved to the value carried forward). Used to distinguish EAS's own
        # commands echoed back in positionCommanded from fresh observer
        # commands. None until the first daytime adjustment; reset at sundown.
        self.louver_eas_command: list[float] | None = None

        self.log = log
        self.dome_model = dome_model
        self.dome_remote = dome_remote
        self.features_to_disable = features_to_disable
        self.allow_send = allow_send

    @classmethod
    def get_config_schema(cls) -> str:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for Louver EAS configuration.
type: object
properties:
  louver_sun_angle:
    description: >-
      Minimum angular separation (degrees) between a louver's azimuth and the
      sun's azimuth at which the louver is considered to be facing the sun.
    type: number
    default: 60.0
  louver_exposed_command:
    description: >-
      Maximum commanded percent-open position for a louver that faces the sun.
      A sun-facing louver is never opened beyond this value, nor beyond the
      position the observer commanded.
    type: number
    default: 50.0
  sun_altitude_threshold:
    description: >-
      Sun altitude (degrees) above which louver positions are adjusted based
      on sun azimuth.
    type: number
    default: -1.0
required:
  - louver_sun_angle
  - louver_exposed_command
additionalProperties: false
"""
        )

    @property
    def cadence(self) -> float:
        """Time (seconds) to wait between control loop passes."""
        return DORMANT_TIME

    async def azimuth_callback(self, azimuth_telemetry: salobj.BaseMsgType) -> None:
        """Callback for MTDome.tel_azimuth.

        Data from this telemetry item is used to track the current dome
        azimuth, which is required to compute per-louver sun angles. When the
        dome azimuth changes by more than `AZIMUTH_CHANGE_THRESHOLD` (or is
        being read for the first time), the control loop is asked to run
        immediately rather than waiting out its cadence.

        Parameters
        ----------
        azimuth_telemetry : `~lsst.ts.salobj.BaseMsgType`
            A newly received azimuth telemetry item.
        """
        new_azimuth = azimuth_telemetry.positionActual
        previous_azimuth = self.dome_azimuth
        self.dome_azimuth = new_azimuth

        if previous_azimuth is None or abs(new_azimuth - previous_azimuth) > AZIMUTH_CHANGE_THRESHOLD:
            self.update_requested.set()

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

    def get_sun_altaz(self) -> tuple[float, float]:
        """Return the sun's current altitude and azimuth.

        Returns
        -------
        `tuple` [`float`, `float`]
            Altitude and azimuth of the sun in degrees at the observatory
            location, corrected for refraction.
        """
        t = Time.now()
        sun = get_sun(t)
        altaz = sun.transform_to(
            AltAz(
                obstime=t,
                location=OBSERVATORY_LOCATION,
                pressure=SUN_ALTITUDE_PRESSURE,
            )
        )
        return altaz.alt.deg, altaz.az.deg

    def select_mode(self, sun_altitude: float) -> LouverControlMode:
        """Return the setpoint source that applies to the current conditions.

        Parameters
        ----------
        sun_altitude : `float`
            Current sun altitude in degrees above the horizon.

        Returns
        -------
        `LouverControlMode`
            The mode whose setpoints should be applied.
        """
        if "day_louvers" in self.features_to_disable:
            return LouverControlMode.IDLE

        if sun_altitude > self.sun_altitude_threshold:
            return LouverControlMode.DAYTIME_LOUVER_CONTROL

        return LouverControlMode.IDLE

    async def exit_mode(self, mode: LouverControlMode) -> None:
        """Perform the actions required when leaving `mode`.

        Leaving `~LouverControlMode.DAYTIME_LOUVER_CONTROL` releases the
        daytime cap, so each louver the observer opened is commanded to its
        observer-commanded position one last time and the cached
        observer-commanded position is cleared. This happens at sundown, and
        also if ``day_louvers`` is disabled while the sun is up, so that
        louvers are never left capped with nothing to lift the cap.

        Parameters
        ----------
        mode : `LouverControlMode`
            The mode being left.
        """
        if mode is LouverControlMode.DAYTIME_LOUVER_CONTROL and self.louver_operator_command is not None:
            await self.set_louvers(self.louver_operator_command)
            self.louver_operator_command = None
            self.louver_eas_command = None

    async def adjust_louvers(self, sun_azimuth: float) -> None:
        """Adjust positions of louvers.

        EAS never opens a louver beyond the position the observer commanded. A
        louver the observer has opened is capped at `louver_exposed_command`
        while it faces the sun (within `louver_sun_angle`) and is otherwise
        left at the observer's commanded position. A louver the observer has
        not opened remains uncommanded.

        Because EAS commands louvers through the same path the observer uses,
        the `positionCommanded` telemetry is overwritten by EAS's own caps and
        no longer reflects the observer's intent. The standing observer command
        is therefore tracked separately in `self.louver_operator_command`.
        A change in `positionCommanded` that differs from what EAS last sent
        (`self.louver_eas_command`) is taken to be a fresh observer command and
        updates the baseline.

        Parameters
        ----------
        sun_azimuth : `float`
            Current sun azimuth in degrees, north = 0, east = 90.
        """
        louvers_telemetry = self.dome_model.louvers_telemetry
        if louvers_telemetry is None or self.dome_azimuth is None:
            return

        position_commanded = list(louvers_telemetry.positionCommanded[: len(LouverTable)])

        # Reconcile the observer baseline against the latest telemetry.
        if self.louver_operator_command is None or self.louver_eas_command is None:
            # First daytime adjustment: adopt the reported commands as the
            # observer's standing request.
            self.louver_operator_command = list(position_commanded)
        else:
            for i, commanded in enumerate(position_commanded):
                if commanded <= 0 or abs(commanded - self.louver_eas_command[i]) > LOUVER_COMMAND_TOLERANCE:
                    # The observer moved this louver (closed it, or commanded a
                    # position EAS did not). Adopt it as the new baseline.
                    self.louver_operator_command[i] = commanded

        louver_azimuth = [(self.dome_azimuth + louver.azimuth) % CIRCLE for louver in LouverTable]
        sun_distance = [
            min((sun_azimuth - az) % CIRCLE, (az - sun_azimuth) % CIRCLE) for az in louver_azimuth
        ]

        # Settle on what commands to send:
        louver_command = [
            (
                -1.0
                if base <= 0  # <-- No command if the louver is closed.
                else min(base, self.louver_exposed_command)  # Min of command or `louver_exposed_command`...
                if sd < self.louver_sun_angle  # ...if the louver is exposed to the sun...
                else base  # ... or the observer's commanded position otherwise.
            )
            for base, sd in zip(self.louver_operator_command, sun_distance)
        ]

        # Copy the non-negative values in `louver_command` to
        # `self.louver_eas_command` for later reference.
        self.louver_eas_command = [
            previous if command < 0 else command
            for command, previous in zip(
                louver_command,
                self.louver_eas_command if self.louver_eas_command is not None else louver_command,
            )
        ]

        await self.set_louvers(position=louver_command)

    async def update_louvers(self) -> None:
        """Run one pass of louver control.

        Select the mode that applies to current conditions, handle any mode
        change, and apply the setpoints belonging to the selected mode.
        """
        sun_altitude, sun_azimuth = self.get_sun_altaz()

        new_mode = self.select_mode(sun_altitude)
        if new_mode is not self.mode:
            self.log.info(f"Louver control mode {self.mode.name} -> {new_mode.name}")
            await self.exit_mode(self.mode)
            self.mode = new_mode

        match self.mode:
            case LouverControlMode.DAYTIME_LOUVER_CONTROL:
                await self.adjust_louvers(sun_azimuth)
            case LouverControlMode.IDLE:
                pass

    async def wait_for_next_cycle(self) -> None:
        """Wait for the cadence to expire or for an update to be requested."""
        try:
            await asyncio.wait_for(self.update_requested.wait(), timeout=self.cadence)
        except asyncio.TimeoutError:
            pass

    async def monitor(self) -> None:
        """Adjust louvers according to the selected control mode.

        This is the only source of ``setLouvers`` commands. Each pass selects
        a `LouverControlMode` and applies that mode's setpoints, then waits
        for either the cadence to expire or a telemetry callback to request an
        earlier pass.

        Signals `monitor_start_event` before entering the polling loop so
        that the EAS CSC can synchronize startup across all monitor tasks.
        """
        self.log.debug("LouverModel.monitor")

        self.monitor_start_event.set()
        try:
            while True:
                # Clear before doing the work, so that a request arriving while
                # the pass is in flight schedules another pass rather than
                # being swallowed by this one.
                self.update_requested.clear()

                try:
                    await self.update_louvers()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.log.exception("In LouverModel control loop.")

                await self.wait_for_next_cycle()
        finally:
            self.monitor_start_event.clear()

    async def close(self) -> None:
        """Cancel any in-flight command tasks."""
        await close_command_tasks(self)
