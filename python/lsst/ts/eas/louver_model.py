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
import math
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
from .dome_model import DomeModel
from .weather_model import WeatherModel

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

# Maximum angle (degrees) between a louver's normal and the wind direction for
# that louver to count as upwind.
UPWIND_ANGLE = 90.0

# Percent-open limits for a louver.
FULLY_CLOSED = 0.0
FULLY_OPEN = 100.0

# Value that tells MTDome to leave a louver where it is.
DO_NOT_MOVE = -1.0


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
    NIGHTTIME_LOUVER_CONTROL
        Inside-windspeed balancing. Upwind and downwind louver groups are
        adjusted to hold the inside windspeed near a target.
    """

    IDLE = enum.auto()
    DAYTIME_LOUVER_CONTROL = enum.auto()
    NIGHTTIME_LOUVER_CONTROL = enum.auto()


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
    weather_model : `WeatherModel`
        The weather model, which owns the outside wind direction and the
        inside anemometer windspeeds used by nighttime louver control.
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
    nighttime_vent_delay : `float`
        Time (seconds) after the louvers open before nighttime louver control
        begins, allowing the dome to vent first.
    inside_windspeed_threshold : `float`
        Target windspeed (m/s) inside the dome during nighttime louver
        control.
    inside_windspeed_deadband : `float`
        Half-width (m/s) of the band around ``inside_windspeed_threshold``
        within which no louver adjustment is made.
    upwind_interval : `float`
        Percent-open step applied to the upwind louvers per adjustment.
    downwind_interval : `float`
        Percent-open step applied to the downwind louvers per adjustment.
    wind_interval : `float`
        Time window (seconds) over which the outside wind direction is
        averaged.
    anemometer_interval : `float`
        Time window (seconds) over which the inside anemometer windspeeds are
        averaged.
    nighttime_adjustment_interval : `float`
        Time (seconds) between nighttime louver adjustments. Also the control
        loop cadence while nighttime control is active.
    features_to_disable : `list` [`str`]
        List of feature names to disable. If ``"day_louvers"`` is present,
        louvers are not adjusted during the day. If ``"night_louvers"`` is
        present, louvers are not adjusted during the night.
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
        weather_model: WeatherModel,
        dome_remote: salobj.Remote,
        louver_sun_angle: float,
        louver_exposed_command: float,
        sun_altitude_threshold: float,
        nighttime_vent_delay: float,
        inside_windspeed_threshold: float,
        inside_windspeed_deadband: float,
        upwind_interval: float,
        downwind_interval: float,
        wind_interval: float,
        anemometer_interval: float,
        nighttime_adjustment_interval: float,
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

        # Most recent reported azimuth of dome (degrees east of north)
        self.dome_azimuth: float | None = None

        # Minimum angle between louver azimuth and sun azimuth (degrees)
        # at which a louver is considered to be "facing" the sun.
        self.louver_sun_angle: float = louver_sun_angle

        # Maximum allowed command position for a louver facing the sun.
        self.louver_exposed_command: float = louver_exposed_command

        # Sun altitude (degrees) above which louvers are adjusted.
        self.sun_altitude_threshold: float = sun_altitude_threshold

        # Venting time (seconds) after the louvers open before nighttime
        # control begins.
        self.nighttime_vent_delay: float = nighttime_vent_delay

        # Target inside windspeed (m/s) and the half-width of the band around
        # it within which no adjustment is made.
        self.inside_windspeed_threshold: float = inside_windspeed_threshold
        self.inside_windspeed_deadband: float = inside_windspeed_deadband

        # Step increment size applied to each group per adjustment
        # during nighttime control.
        self.upwind_interval: float = upwind_interval
        self.downwind_interval: float = downwind_interval

        # Trailing windows (seconds) over which the wind measurements are
        # averaged, and the cadence (seconds) between nighttime adjustments.
        self.wind_interval: float = wind_interval
        self.anemometer_interval: float = anemometer_interval
        self.nighttime_adjustment_interval: float = nighttime_adjustment_interval

        # The position nighttime control is currently asking for, per louver.
        # Seeded from the observer's command on entry and then stepped one
        # interval at a time, so successive adjustments accumulate and the
        # louvers can travel as far as the inside windspeed requires. This is
        # the loop's only persistent state; because it *is* the commanded
        # position and is clamped to 0-100, there is no accumulated error
        # hiding behind the actuator limit and nothing to unwind when the wind
        # reverses. None until nighttime control is entered.
        self.night_target_positions: list[float] | None = None

        # Per-louver flag for whether nighttime control may move that louver.
        # A louver the operator did not open is left alone entirely: EAS never
        # opens a louver the operator did not ask for. None until nighttime
        # control is entered.
        self.night_operable: list[bool] | None = None

        # Observer-commanded position per louver. EAS caps a sun-facing louver
        # at `louver_exposed_command` but never opens a louver beyond what the
        # observer requested, so the observer's request must be remembered
        # separately. This list stores observed louver commands issued by the
        # operator so that they can be re-issued when the louver returns to a
        # shaded position. If no command has been observed, the value is None.
        self.louver_operator_command: list[float] | None = None

        # Exactly what EAS last sent for each louver, including the -1 "do not
        # move" entries. Used to distinguish EAS's own commands echoed back in
        # positionCommanded from fresh observer commands. MTDome reports -1 for
        # a louver it holds no command for, so storing the -1 verbatim is what
        # makes the echo compare equal and a subsequent observer command
        # compare different. None until the first daytime adjustment; reset at
        # sundown.
        self.louver_eas_command: list[float] | None = None

        self.log = log
        self.dome_model = dome_model
        self.weather_model = weather_model
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
  nighttime_vent_delay:
    description: >-
      Time (seconds) after the louvers open before nighttime louver control
      begins, allowing the dome to vent first.
    type: number
    default: 1800.0
  inside_windspeed_threshold:
    description: Target windspeed (m/s) inside the dome during nighttime louver control.
    type: number
    default: 2.0
  inside_windspeed_deadband:
    description: >-
      Half-width (m/s) of the band around the inside windspeed threshold within
      which no louver adjustment is made.
    type: number
    default: 0.2
  upwind_interval:
    description: Percent-open step applied to the upwind louvers per adjustment.
    type: number
    default: 10.0
  downwind_interval:
    description: Percent-open step applied to the downwind louvers per adjustment.
    type: number
    default: 25.0
  wind_interval:
    description: Time window (s) over which the outside wind direction is averaged.
    type: number
    default: 60.0
  anemometer_interval:
    description: Time window (s) over which the inside anemometer windspeeds are averaged.
    type: number
    default: 15.0
  nighttime_adjustment_interval:
    description: Time (s) between nighttime louver adjustments.
    type: number
    default: 15.0
required:
  - louver_sun_angle
  - louver_exposed_command
additionalProperties: false
"""
        )

    @property
    def cadence(self) -> float:
        """Time (seconds) to wait between control loop passes.

        Nighttime control steps the targets by a fixed interval each pass, so
        its cadence sets how quickly the louvers converge and is configurable.
        Every other mode only needs to keep up with the sun.
        """
        if self.mode is LouverControlMode.NIGHTTIME_LOUVER_CONTROL:
            return self.nighttime_adjustment_interval

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
        if sun_altitude > self.sun_altitude_threshold:
            if "day_louvers" in self.features_to_disable:
                return LouverControlMode.IDLE
            return LouverControlMode.DAYTIME_LOUVER_CONTROL

        # The sun is down.
        if "night_louvers" in self.features_to_disable:
            return LouverControlMode.IDLE
        if self.vent_delay_elapsed():
            return LouverControlMode.NIGHTTIME_LOUVER_CONTROL

        return LouverControlMode.IDLE

    def louver_targets(self, inside_windspeed: float, upwind: list[bool]) -> list[float]:
        """Return the target position for each louver.

        The louver targets are adjusted based on indoor windspeed and the
        outdoor wind direction. The downwind louvers are the ones stepped open
        first and shut last; the upwind louvers are brought in only once every
        downwind louver has run to the end of its travel.

        Parameters
        ----------
        inside_windspeed : `float`
            Current average windspeed (m/s) inside the dome.
        upwind : `list` [`bool`]
            Per-louver flags from `is_upwind`.

        Returns
        -------
        `list` [`float`]
            Target percent-open position per louver, clamped to 0-100.
        """
        if self.night_target_positions is None or self.night_operable is None:
            raise RuntimeError("louver_targets called before the observer baseline was captured.")

        targets = self.night_target_positions
        operable = self.night_operable

        if abs(inside_windspeed - self.inside_windspeed_threshold) <= self.inside_windspeed_deadband:
            # No change because windspeed is within deadband.
            return list(targets)

        downwind = [index for index, is_up in enumerate(upwind) if not is_up and operable[index]]
        upwind_group = [index for index, is_up in enumerate(upwind) if is_up and operable[index]]

        # The two groups are alternatives, never both: adjust the preferred
        # group if that is possible, and only if it is not, adjust the other.
        # Both tests read the current targets, so a group that has been stepped
        # to the end of its travel hands over to the other one.
        if inside_windspeed < self.inside_windspeed_threshold:
            # Too still: let more air in, from the downwind louvers if any of
            # them can open further, and from the upwind ones if none can.
            if any(targets[index] < FULLY_OPEN for index in downwind):
                trim = {index: self.downwind_interval for index in downwind}
            else:
                trim = {index: self.upwind_interval for index in upwind_group}
        else:
            # Too windy: shut the wind out, from the upwind louvers if any of
            # them can close further, and from the downwind ones if none can.
            if any(targets[index] > FULLY_CLOSED for index in upwind_group):
                trim = {index: -self.upwind_interval for index in upwind_group}
            else:
                trim = {index: -self.downwind_interval for index in downwind}

        # Only stepped louvers are clamped. An unstepped entry is passed
        # through verbatim, so the -1 of a louver under no command survives
        # rather than being clamped up to 0.
        return [
            min(FULLY_OPEN, max(FULLY_CLOSED, position + trim[index])) if index in trim else position
            for index, position in enumerate(targets)
        ]

    def nighttime_louver_command(self, targets: list[float], adjusted: set[int]) -> list[float]:
        """Return the positions to command during nighttime control.

        Only the louvers actually being moved on this pass carry a position.
        Every other louver is commanded `DO_NOT_MOVE` and simply stays where
        MTDome last put it.

        Restating an unchanged position instead would reintroduce, one branch
        along, the same defect the deadband hold avoids: whichever group is not
        being stepped would be written back to where it started, undoing an
        earlier adjustment that is still doing its job. Commanding only what is
        being changed keeps every branch free of that.

        Parameters
        ----------
        targets : `list` [`float`]
            Target position per louver, from `louver_targets`.
        adjusted : `set` [`int`]
            Indices of the louvers whose target changed on this pass. A louver
            the operator did not open never appears here, since it is never
            stepped.

        Returns
        -------
        `list` [`float`]
            Desired percent-open position per louver, or `DO_NOT_MOVE`.
        """
        return [target if index in adjusted else DO_NOT_MOVE for index, target in enumerate(targets)]

    def is_upwind(self, wind_direction: float) -> list[bool]:
        """Return, per louver, whether it faces into the wind.

        A louver is upwind when its outward normal lies within
        `UPWIND_ANGLE` degrees of the direction the wind is blowing from, and
        downwind otherwise. Because `LouverTable` gives each louver's azimuth
        relative to the dome, the comparison is made in absolute azimuth, so
        the grouping changes as the dome rotates.

        Note that ESS ``airFlow.direction`` is taken to be meteorological, that
        is, the direction the wind blows *from*. A louver whose normal points
        at that bearing therefore faces into the wind.

        Parameters
        ----------
        wind_direction : `float`
            Direction (degrees east of north) the wind is blowing from.

        Returns
        -------
        `list` [`bool`]
            One flag per louver in `LouverTable`, True when upwind.

        Raises
        ------
        `RuntimeError`
            If the dome azimuth is not yet known.
        """
        if self.dome_azimuth is None:
            raise RuntimeError("is_upwind called before dome azimuth was known.")

        louver_azimuth = [(self.dome_azimuth + louver.azimuth) % CIRCLE for louver in LouverTable]

        return [
            min((wind_direction - azimuth) % CIRCLE, (azimuth - wind_direction) % CIRCLE) <= UPWIND_ANGLE
            for azimuth in louver_azimuth
        ]

    def vent_delay_elapsed(self) -> bool:
        """Return whether the louvers have been open long enough to vent.

        Returns
        -------
        `bool`
            True if the louvers have been open for at least
            `nighttime_vent_delay`
            seconds. False if they are closed, their state is unknown, or the
            delay has not yet elapsed.
        """
        louvers_open_time = self.dome_model.louvers_open_time
        if louvers_open_time is None:
            return False

        return (utils.current_tai() - louvers_open_time) >= self.nighttime_vent_delay

    def enter_mode(self, mode: LouverControlMode) -> None:
        """Perform the actions required when entering `mode`.

        Entering `~LouverControlMode.NIGHTTIME_LOUVER_CONTROL` seeds each
        louver's target from the position the operator commanded it to, so that
        nothing moves at the moment control is handed over, and records which
        louvers nighttime control is allowed to move. A louver reported as -1
        (under no command) or 0 (commanded shut) is not one the operator
        opened, so it takes no part.

        Parameters
        ----------
        mode : `LouverControlMode`
            The mode being entered.
        """
        if mode is LouverControlMode.NIGHTTIME_LOUVER_CONTROL:
            baseline = self.night_baseline()
            self.night_target_positions = list(baseline)
            self.night_operable = [position > FULLY_CLOSED for position in baseline]

    def night_baseline(self) -> list[float]:
        """Return the operator's standing command to start nighttime control.

        Prefers the baseline captured during daytime control: at the
        day-to-night transition the restore command is still in flight, so
        ``positionCommanded`` telemetry would still be showing EAS's own
        daytime caps rather than what the operator asked for. Falls back to
        telemetry when EAS started up after sundown and so never ran the
        daytime logic.

        Returns
        -------
        `list` [`float`]
            Operator-commanded position per louver, empty if not yet known.
        """
        if self.louver_operator_command is not None:
            return list(self.louver_operator_command)

        louvers_telemetry = self.dome_model.louvers_telemetry
        if louvers_telemetry is None:
            return []

        return list(louvers_telemetry.positionCommanded[: len(LouverTable)])

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

        # Record exactly what was sent, including -1 for louvers EAS is not
        # commanding, so it compares equal to the echoed telemetry.
        self.louver_eas_command = list(louver_command)

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
            self.enter_mode(new_mode)

        match self.mode:
            case LouverControlMode.DAYTIME_LOUVER_CONTROL:
                await self.adjust_louvers(sun_azimuth)
            case LouverControlMode.NIGHTTIME_LOUVER_CONTROL:
                await self.balance_louvers()
            case LouverControlMode.IDLE:
                pass

    async def balance_louvers(self) -> None:
        """Adjust louvers to hold the inside windspeed near its threshold.

        Groups the louvers by the current relative wind direction, steps the
        targets one interval toward the threshold, and commands the result. The
        grouping is recomputed every pass, so dome rotation alone moves a
        louver between groups and changes which interval steps it.
        """
        if self.dome_azimuth is None:
            self.log.warning("No dome azimuth; skipping nighttime louver adjustment.")
            return

        wind_direction = self.weather_model.average_wind_direction(window=self.wind_interval)
        inside_windspeed = self.weather_model.average_indoor_windspeed(window=self.anemometer_interval)

        if math.isnan(wind_direction) or math.isnan(inside_windspeed):
            # Grouping is meaningless without a wind direction, and there is
            # nothing to steer toward without an inside windspeed.
            self.log.warning("Wind measurements unavailable; skipping nighttime louver adjustment.")
            return

        if not self.night_target_positions:
            # Neither a daytime baseline nor louver telemetry was available
            # when this mode was entered, so there is nothing to control.
            self.log.warning("No operator louver baseline; skipping nighttime louver adjustment.")
            return

        targets = self.louver_targets(
            inside_windspeed=inside_windspeed, upwind=self.is_upwind(wind_direction)
        )

        adjusted = {
            index
            for index, (previous, target) in enumerate(zip(self.night_target_positions, targets))
            if target != previous
        }
        if not adjusted:
            # Nothing moved, so there is nothing to say. This covers both the
            # deadband and the case where the louvers have run out of travel
            # and the inside windspeed is simply not achievable in the current
            # conditions -- which is a normal state, not a fault.
            return

        self.night_target_positions = targets
        await self.set_louvers(position=self.nighttime_louver_command(targets, adjusted))

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
