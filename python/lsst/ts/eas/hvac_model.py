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

__all__ = ["HvacModel", "HVAC_SLEEP_TIME"]

import asyncio
import logging
import math

from astropy.time import Time
from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .weather_model import WeatherModel

HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state (seconds)
STD_TIMEOUT = 5  # seconds


class HvacModel:
    """A model for HVAC system automation.

    Parameters
    ----------
    domain : `~lsst.ts.salobj.Domain`
        A SAL domain object for obtaining remotes.
    log : `~logging.Logger`
        A logger for log messages.
    diurnal_timer : `DiurnalTimer`
        A timer that signals at noon, sunrise, and at the end of twilight.
    dome_model : `DomeModel`
        A model representing the dome state.
    weather_model : `WeatherModel`
        A model representing weather conditions.
    setpoint_lower_limit : float
        The minimum allowed setpoint for thermal control. If a lower setpoint
        than this is indicated from the ESS temperature readings, this setpoint
        will be used instead.
    wind_threshold : float
        Windspeed limit for the VEC-04 fan. (m/s)
    vec04_hold_time : float
        Minimum time to wait before changing the state of the VEC-04 fan. This
        value is ignored if the dome is opened or closed. (s)
    disable_features: list[str]
        A list of features that should be disabled. The following strings can
        be used:
         * vec04
         * ahu
         * room_setpoint
        Any other values are ignored.
    """

    def __init__(
        self,
        *,
        domain: salobj.Domain,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        dome_model: DomeModel,
        weather_model: WeatherModel,
        setpoint_lower_limit: float = 6,
        wind_threshold: float = 5,
        vec04_hold_time: float = 5 * 60,
        features_to_disable: list[str] = [],
    ) -> None:
        self.domain = domain
        self.log = log
        self.diurnal_timer = diurnal_timer

        self.monitor_start_event = asyncio.Event()

        self.last_vec04_time: float = (
            0  # Last time VEC-04 was changed (UNIX TAI seconds).
        )

        # Configuration parameters:
        self.dome_model = dome_model
        self.weather_model = weather_model
        self.setpoint_lower_limit = setpoint_lower_limit
        self.wind_threshold = wind_threshold
        self.vec04_hold_time = vec04_hold_time
        self.features_to_disable = features_to_disable

        # Glycol chiller parameters:
        self.glycol_band_low = glycol_band_low
        self.glycol_band_high = glycol_band_high
        self.glycol_average_offset = glycol_average_offset
        self.glycol_dew_point_margin = glycol_dew_point_margin
        self.glycol_setpoints_delta = glycol_setpoints_delta
        self.glycol_absolute_minimum = glycol_absolute_minimum

        # Glycol setpoints
        self.glycol_setpoint1: float | None = None
        self.glycol_setpoint2: float | None = None

    @classmethod
    def get_config_schema(cls) -> str:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for EAS HVAC configuration.
type: object
properties:
  setpoint_lower_limit:
    type: number
    default: 6
    description: >-
      The minimum allowed setpoint for thermal control. If a lower setpoint
      than this is indicated from the ESS temperature readings, this setpoint
      will be used instead.
  wind_threshold:
    type: number
    default: 10
    description: Windspeed limit for the VEC-04 fan. (m/s)
  vec04_hold_time:
    type: number
    default: 300
    description: >-
      Minimum time to wait before changing the state of the VEC-04 fan. This
      value is ignored if the dome is opened or closed. (s)
  glycol_band_low:
    type: number
    default: -10
    description: Lower bound of allowed glycol setpoint band relative to ambient (°C).
  glycol_band_high:
    type: number
    default: -5
    description: Upper bound of allowed glycol setpoint band relative to ambient (°C).
  glycol_average_offset:
    type: number
    default: -7.5
    description: Nominal average offset for glycol setpoints relative to ambient (°C).
  glycol_dew_point_margin:
    type: number
    default: 1.0
    description: Safety margin (°C) added to the maximum dew point to avoid condensation.
  glycol_setpoints_delta:
    type: number
    default: 1.0
    description: Temperature difference (°C) between the two glycol chiller setpoints (chiller 1 warmer).
  glycol_absolute_minimum:
    type: number
    default: -10.0
    description: Absolute minimum setpoint (°C) allowed for the colder glycol chiller.
required:
  - setpoint_lower_limit
  - wind_threshold
  - vec04_hold_time
  - glycol_band_low
  - glycol_band_high
  - glycol_average_offset
  - glycol_dew_point_margin
  - glycol_setpoints_delta
  - glycol_absolute_minimum
additionalProperties: false
"""
        )

    async def monitor(self) -> None:
        """Monitor the dome status and windspeed to control the HVAC.

        This monitor does the following:
         * If the dome is open, it turns on the four AHUs.
         * If the dome is closed, it turns off the AHUs.
         * If the dome is open and the wind is calm, it turns on VEC-04.
        """
        self.log.debug("HvacModel.monitor")

        # Give the dome model an opportunity to collect some telemetry...
        await asyncio.sleep(STD_TIMEOUT)

        async with salobj.Remote(
            domain=self.domain,
            name="HVAC",
            include=("enableDevice", "disableDevice", "configAhu"),
        ) as hvac_remote:
            tasks = [self.control_ahus_and_vec04(hvac_remote=hvac_remote)]

            if "room_setpoint" not in self.features_to_disable:
                tasks.extend(
                    [
                        self.wait_for_sunrise(hvac_remote=hvac_remote),
                        self.adjust_glycol_chillers_at_noon(hvac_remote=hvac_remote),
                        self.apply_setpoint_at_night(hvac_remote=hvac_remote),
                        self.monitor_glycol_chillers(hvac_remote=hvac_remote),
                    ]
                )

            hvac_future = asyncio.gather(*tasks)
            self.monitor_start_event.clear()

            try:
                await hvac_future
            except asyncio.CancelledError:
                hvac_future.cancel()
                await asyncio.gather(hvac_future, return_exceptions=True)
                raise

    async def control_ahus_and_vec04(self, *, hvac_remote: salobj.Remote) -> None:
        cached_shutter_closed = None
        cached_wind_threshold = None

        while True:
            # Check the aperture state
            shutter_closed = self.dome_model.is_closed
            if shutter_closed is None:
                await asyncio.sleep(0.1)
                continue

            if (
                "vec04" not in self.features_to_disable
                and not shutter_closed
                and (utils.current_tai() - self.last_vec04_time > self.vec04_hold_time)
            ):
                # Check windspeed threshold
                average_windspeed = self.weather_model.average_windspeed
                wind_threshold = average_windspeed < self.wind_threshold
                self.log.debug(
                    f"VEC-04 operation demanded: {average_windspeed} m/s -> {wind_threshold}"
                )
                if wind_threshold != cached_wind_threshold:
                    cached_wind_threshold = wind_threshold
                    self.last_vec04_time = utils.current_tai()
                    if wind_threshold:
                        self.log.info("Turning on VEC-04 fan!")
                        await hvac_remote.cmd_enableDevice.set_start(
                            device_id=DeviceId.lowerDamperFan03P04
                        )
                    else:
                        self.log.info("Turning off VEC-04 fan!")
                        await hvac_remote.cmd_disableDevice.set_start(
                            device_id=DeviceId.lowerDamperFan03P04
                        )

            if shutter_closed != cached_shutter_closed:
                cached_shutter_closed = shutter_closed
                ahus = (
                    DeviceId.lowerAHU04P05,
                    DeviceId.lowerAHU03P05,
                    DeviceId.lowerAHU02P05,
                    DeviceId.lowerAHU01P05,
                )
                if shutter_closed:
                    if "ahu" not in self.features_to_disable:
                        # Enable the four AHUs
                        self.log.info("Enabling HVAC AHUs!")
                        for device in ahus:
                            await hvac_remote.cmd_enableDevice.set_start(
                                device_id=device
                            )
                            await asyncio.sleep(0.1)

                    if "vec04" not in self.features_to_disable:
                        # Disable the VEC-04 fan
                        self.log.info("Turning off VEC-04 fan!")
                        await hvac_remote.cmd_disableDevice.set_start(
                            device_id=DeviceId.lowerDamperFan03P04
                        )
                        self.last_vec04_time = utils.current_tai()
                else:
                    if "ahu" not in self.features_to_disable:
                        self.log.info("Disabling HVAC AHUs!")
                        for device in ahus:
                            await hvac_remote.cmd_disableDevice.set_start(
                                device_id=device
                            )
                            await asyncio.sleep(0.1)

            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def wait_for_sunrise(self, *, hvac_remote: salobj.Remote) -> None:
        """Wait for sunrise and then set the room temperature.

        Wait for the timer to signal sunrise, and then obtain the
        temperature that was reported last night at the end
        of twilight, and then apply that temperature as at AHU
        setpoint.

        Parameters
        ----------
        hvac_remote : `~lsst.ts.salobj.Remote`
            A SALobj remote representing the HVAC.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.sunrise_condition:
                self.monitor_start_event.set()

                await self.diurnal_timer.sunrise_condition.wait()
                last_twilight_temperature = (
                    await self.weather_model.get_last_twilight_temperature()
                )
                if (
                    self.diurnal_timer.is_running
                    and last_twilight_temperature is not None
                ):
                    if "room_setpoint" not in self.features_to_disable:
                        # Time to set the room setpoint based on last twilight
                        setpoint = max(
                            last_twilight_temperature,
                            self.setpoint_lower_limit,
                        )

                        for device_id in (
                            DeviceId.lowerAHU01P05,
                            DeviceId.lowerAHU02P05,
                            DeviceId.lowerAHU03P05,
                            DeviceId.lowerAHU04P05,
                        ):
                            await hvac_remote.cmd_configLowerAhu.set_start(
                                device_id=device_id,
                                workingSetpoint=setpoint,
                                maxFanSetpoint=math.nan,
                                minFanSetpoint=math.nan,
                                antiFreezeTemperature=math.nan,
                            )

    def compute_glycol_setpoints(
        self, ambient_temperature: float
    ) -> tuple[float, float]:
        """Computes staggered glycol chiller setpoints.

        Computes staggered glycol chiller setpoints based on ambient
        temperature, configured band limits, dew point, and absolute
        minimum constraints.

        The algorithm enforces:
          * Average setpoint nominally `ambient - glycol_average_offset`.
          * Clamped to the band
            `[ambient - glycol_band_low, ambient - glycol_band_high]`.
            Default values `glycol_band_low` of 5, `glycol_band_high` of 10.
          * Raised if necessary to exceed the nightly maximum indoor dew
            point plus a safety margin (`dewpoint_margin`).
          * Split into two staggered setpoints (`setpoint1`, `setpoint2`)
            separated by `glycol_setpoints_delta`, **with chiller 1 warmer**.
          * Absolute minimum enforced on the colder chiller
            (`setpoint2 >= glycol_absolute_minimum`),
            adjusting both setpoints to preserve the delta.

        Parameters
        ----------
        ambient_temperature : float
            Ambient temperature in degrees Celsius. This value
            is used to determine the nominal target band for the glycol
            loop average.

        Returns
        -------
        setpoint1 : float
            Active setpoint for chiller 1 (°C), the warmer of the two.
        setpoint2 : float
            Active setpoint for chiller 2 (°C), the colder of the two.
        """
        # Find the allowed range of the setpoints
        band_low = ambient_temperature + self.glycol_band_low
        band_high = ambient_temperature + self.glycol_band_high

        # Compute a target average setpoint
        target_average = ambient_temperature + self.glycol_average_offset
        average = min(max(target_average, band_low), band_high)

        # Incorporate dewpoint into the calculation - setpoint
        # average should not be lower than the dewpoint (with margin)
        nightly_maximum_dewpoint = self.weather_model.nightly_maximum_dewpoint
        if nightly_maximum_dewpoint is not None:
            average = max(
                average, nightly_maximum_dewpoint + self.glycol_dewpoint_margin
            )

        # Break average and delta into individual setpoints
        setpoint1 = average + 0.5 * self.glycol_setpoints_delta
        setpoint2 = average - 0.5 * self.glycol_setpoints_delta

        # Enforce the absolute minimum temperature (-10°C)
        if setpoint2 < self.glycol_absolute_minimum:
            setpoint2 = self.glycol_absolute_minimum
            setpoint1 = setpoint2 + self.glycol_setpoints_delta

        return setpoint1, setpoint2

    def check_glycol_setpoint(self, ambient_temperature: float) -> bool:
        """Verify whether the chiller setpoints are within the allowed band.

        Parameters
        ----------
        ambient_temperature : float
            Ambient temperature (°C)

        Returns
        -------
        bool
            True if the current setpoints are acceptable, or False otherwise.
        """
        if self.glycol_setpoint1 is None or self.glycol_setpoint2 is None:
            return False

        average_setpoint = 0.5 * (self.glycol_setpoint1 + self.glycol_setpoint2)
        return (
            self.glycol_band_low
            <= average_setpoint - ambient_temperature
            <= self.glycol_band_high
        )

    async def monitor_glycol_chillers(self, *, hvac_remote: salobj.Remote) -> None:
        """Continuously monitor and enforce glycol chiller setpoints.

        This coroutine runs while the diurnal timer is active. On each cycle it
        checks whether the current chiller setpoints are within the allowed
        band relative to the ambient indoor temperature. If the setpoints are
        outside of the band, new setpoints are computed and applied to the
        HVAC CSC.

        Parameters
        ----------
        hvac_remote : `~lsst.ts.salobj.Remote`
            A SALobj remote representing the HVAC.
        """
        while self.diurnal_timer.is_running:
            try:
                self.log.debug("monitor_glycol_chillers")

                # At night, reset the setpoints and do not control
                # the glycol.
                if self.diurnal_timer.is_night(Time.now()):
                    self.glycol_setpoint1 = None
                    self.glycol_setpoint2 = None
                    await asyncio.sleep(HVAC_SLEEP_TIME)
                    continue

                # After the setpoints are chosen at noon, monitor
                # the system and adjust setpoints if needed.
                ambient_temperature = self.weather_model.current_indoor_temperature
                if ambient_temperature is not None and not self.check_glycol_setpoint(
                    ambient_temperature
                ):
                    self.log.debug("Recomputing glycol setpoints.")
                    glycol_setpoint1, glycol_setpoint2 = self.compute_glycol_setpoints(
                        ambient_temperature
                    )

                    if all(
                        (
                            glycol_setpoint1 is not None,
                            not math.isnan(glycol_setpoint1),
                            glycol_setpoint2 is not None,
                            not math.isnan(glycol_setpoint2),
                        )
                    ):
                        self.glycol_setpoint1 = glycol_setpoint1
                        self.glycol_setpoint2 = glycol_setpoint2

                await hvac_remote.cmd_configChiller.set_start(
                    device_id=DeviceId.chiller01P01,
                    activeSetpoint=glycol_setpoint1,
                )
                await hvac_remote.cmd_configChiller.set_start(
                    device_id=DeviceId.chiller02P01,
                    activeSetpoint=glycol_setpoint2,
                )
            except Exception:
                self.log.exception("In HVAC glycol control loop")

            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def adjust_glycol_chillers_at_noon(
        self, *, hvac_remote: salobj.Remote
    ) -> None:
        """Waits for noon and then sets the glycol chillers.

        Waits for the timer to signal noon, and then obtains the minimum
        temperature that was reported last night, and then applies an
        appropriate temperature as the glycol setpoint.

        Parameters
        ----------
        hvac_remote : `~lsst.ts.salobj.Remote`
            A SALobj remote representing the HVAC.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                self.monitor_start_event.set()

                await self.diurnal_timer.noon_condition.wait()
                if not self.diurnal_timer.is_running:
                    return

                nightly_minimum_temperature = (
                    self.weather_model.nightly_minimum_temperature
                )
                if nightly_minimum_temperature is None:
                    self.log.error("Nightly minimum temperature was not available.")
                    continue

                self.glycol_setpoint1, self.glycol_setpoint2 = (
                    self.compute_glycol_setpoints(nightly_minimum_temperature)
                )

                if self.glycol_setpoint1 is None or self.glycol_setpoint2 is None:
                    self.log.error("Failed to calculate noon glycol setpoints.")
                    continue

                await hvac_remote.cmd_configChiller.set_start(
                    device_id=DeviceId.chiller01P01,
                    activeSetpoint=self.glycol_setpoint1,
                )
                await hvac_remote.cmd_configChiller.set_start(
                    device_id=DeviceId.chiller02P01,
                    activeSetpoint=self.glycol_setpoint2,
                )

    async def apply_setpoint_at_night(self, *, hvac_remote: salobj.Remote) -> None:
        """Control the HVAC setpoint during the night.

        At night time (defined by `DiurnalTimer.is_night`) the HVAC
        AHU setpoint should be applied based on the outside temperature,
        if the dome is closed. If the dome is open, the HVAC AHUs
        should not be enabled, and the setpoint should not matter.

        Parameters
        ----------
        hvac_remote : `~lsst.ts.salobj.Remote`
            A SALobj remote representing the HVAC.
        """
        warned_no_temperature = False

        while self.diurnal_timer.is_running:
            if self.diurnal_timer.is_night(Time.now()) and self.dome_model.is_closed:
                setpoint = max(
                    self.weather_model.current_temperature,
                    self.setpoint_lower_limit,
                )
                if math.isnan(setpoint):
                    if not warned_no_temperature:
                        self.log.warning(
                            "Failed to collect a temperature sample for HVAC setpoint."
                        )
                        warned_no_temperature = True

                else:
                    # Apply setpoint for each of the 4 AHUs
                    for device_id in (
                        DeviceId.lowerAHU01P05,
                        DeviceId.lowerAHU02P05,
                        DeviceId.lowerAHU03P05,
                        DeviceId.lowerAHU04P05,
                    ):
                        await hvac_remote.cmd_configLowerAhu.set_start(
                            device_id=device_id,
                            workingSetpoint=setpoint,
                            maxFanSetpoint=math.nan,
                            minFanSetpoint=math.nan,
                            antiFreezeTemperature=math.nan,
                        )

            await asyncio.sleep(HVAC_SLEEP_TIME)
