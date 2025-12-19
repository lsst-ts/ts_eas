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
from typing import Any, Callable

import yaml
from astropy.time import Time

from lsst.ts import salobj, utils
from lsst.ts.xml.enums.HVAC import DeviceId

from .cmdwrapper import close_command_tasks, command_wrapper
from .diurnal_timer import DiurnalTimer
from .dome_model import DomeModel
from .weather_model import WeatherModel

HVAC_SLEEP_TIME = 60.0  # How often to check the HVAC state (seconds)
STD_TIMEOUT = 5  # seconds
N_CHILLERS = 2  # EAS controls two HVAC glycol chillers.


class HvacModel:
    """A model for HVAC system automation.

    Parameters
    ----------
    log : `~logging.Logger`
        A logger for log messages.
    diurnal_timer : `DiurnalTimer`
        A timer that signals at noon, sunrise, and at the end of twilight.
    dome_model : `DomeModel`
        A model representing the dome state.
    weather_model : `WeatherModel`
        A model representing weather conditions.
    ahu_setpoint_delta : `float`
        The offset that will be added to the measured temperature in
        selecting a setpoint for the HVAC air handling units (AHUs/UMAs)
        measured in °C.
    setpoint_lower_limit : `float`
        The minimum allowed setpoint for thermal control. If a lower setpoint
        than this is indicated from the ESS temperature readings, this setpoint
        will be used instead.
    wind_threshold : `float`
        Windspeed limit for the VEC-04 fan. (m/s)
    vec04_hold_time : `float`
        Minimum time to wait before changing the state of the VEC-04 fan. This
        value is ignored if the dome is opened or closed. (s)
    glycol_band_low : `float`
        The lower bound (more negative) of the allowed difference between the
        average glycol setpoint and the ambient temperature (°C). This
        represents how far below ambient the setpoint is permitted to drift
        before being considered out of range. This number is expected (but not
        required) to be negative.
    glycol_band_high : `float`
        Upper bound of allowed glycol setpoint band relative to ambient (°C),
        corresponding to `glycol_band_low`. This number is expected (but not
        required) to be negative.
    glycol_average_offset : `float`
        Nominal average offset for glycol setpoints relative to ambient (°C).
        The average of the two glycol setpoints should differ from the
        ambient temperature reported by the ESS by this amount.
    glycol_dew_point_margin : `float`
        Safety margin (°C) added to the maximum dew point to avoid
        condensation.
    glycol_setpoints_delta : `float`
        Temperature difference (°C) between the two glycol chiller setpoints
        (chiller 1 warmer).
    glycol_absolute_minimum : `float`
        Absolute minimum setpoint (°C) allowed for the colder glycol chiller.
    glycol_absolute_maximum : `float`
        Absolute maximum setpoint (°C) allowed for the warmer glycol chiller.
    disable_features: `list[str]`
        A list of features that should be disabled. The following strings can
        be used:
         * vec04
         * ahu
         * room_setpoint
         * glycol_chillers
        Any other values are ignored.
    """

    def __init__(
        self,
        *,
        log: logging.Logger,
        diurnal_timer: DiurnalTimer,
        dome_model: DomeModel,
        weather_model: WeatherModel,
        hvac_remote: salobj.Remote,
        ahu_setpoint_delta: float,
        setpoint_lower_limit: float,
        wind_threshold: float,
        vec04_hold_time: float,
        glycol_band_low: float,
        glycol_band_high: float,
        glycol_average_offset: float,
        glycol_dew_point_margin: float,
        glycol_setpoints_delta: float,
        glycol_absolute_minimum: float,
        glycol_absolute_maximum: float,
        features_to_disable: list[str],
        allow_send: Callable[[], bool] | None = None,
    ) -> None:
        self.log = log
        self.allow_send = allow_send
        self.diurnal_timer = diurnal_timer

        self.monitor_start_event = asyncio.Event()

        self.last_vec04_time: float = 0  # Last time VEC-04 was changed (UNIX TAI seconds).

        # Configuration parameters:
        self.dome_model = dome_model
        self.weather_model = weather_model
        self.ahu_setpoint_delta = ahu_setpoint_delta
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
        self.glycol_absolute_maximum = glycol_absolute_maximum

        # Glycol setpoints
        self.glycol_setpoint1: float | None = None
        self.glycol_setpoint2: float | None = None

        # The remote
        self.hvac_remote = hvac_remote

    @classmethod
    def get_config_schema(cls) -> str:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for EAS HVAC configuration.
type: object
properties:
  ahu_setpoint_delta:
    type: number
    default: -1.0
    description: >-
      The offset that will be applied to the measured temperature in
      selecting a setpoint for the HVAC air handling units (AHUs/UMAs)
      measured in °C.
  setpoint_lower_limit:
    type: number
    default: 6.0
    description: >-
      The minimum allowed setpoint for thermal control. If a lower setpoint
      than this is indicated from the ESS temperature readings, this setpoint
      will be used instead.
  wind_threshold:
    type: number
    default: 10.0
    description: Windspeed limit for the VEC-04 fan (m/s).
  vec04_hold_time:
    type: number
    default: 300.0
    description: >-
      Minimum time to wait before changing the state of the VEC-04 fan. This
      value is ignored if the dome is opened or closed (s).
  glycol_band_low:
    type: number
    default: -10.0
    description: Lower bound of allowed glycol setpoint band relative to ambient (°C).
  glycol_band_high:
    type: number
    default: -5.0
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
  glycol_absolute_maximum:
    type: number
    default: 10.0
    description: Absolute maximum setpoint (°C) allowed for the warmer glycol chiller.
required:
  - ahu_setpoint_delta
  - setpoint_lower_limit
  - wind_threshold
  - vec04_hold_time
  - glycol_band_low
  - glycol_band_high
  - glycol_average_offset
  - glycol_dew_point_margin
  - glycol_setpoints_delta
  - glycol_absolute_minimum
  - glycol_absolute_maximum
additionalProperties: false
"""
        )

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_enableDevice")
    async def enable_devices(self, device_ids: list[DeviceId]) -> list[dict[str, Any]] | None:
        if not device_ids:
            return None
        return [{"device_id": device_id} for device_id in device_ids]

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_disableDevice")
    async def disable_devices(self, device_ids: list[DeviceId]) -> list[dict[str, Any]] | None:
        if not device_ids:
            return None
        return [{"device_id": device_id} for device_id in device_ids]

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_configLowerAhu")
    async def config_lower_ahu(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not commands:
            return None
        return commands

    @command_wrapper(remote_attr="hvac_remote", command_attr="cmd_configChiller")
    async def config_chiller(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not commands:
            return None
        return commands

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

        tasks = [self.control_ahus_and_vec04()]

        if "room_setpoint" not in self.features_to_disable:
            tasks.extend(
                [
                    self.wait_for_sunrise(),
                    self.apply_setpoint_at_night(),
                ]
            )

        if "glycol_chillers" not in self.features_to_disable:
            tasks.extend(
                [
                    self.adjust_glycol_chillers_at_noon(),
                    self.monitor_glycol_chillers(),
                ]
            )

        hvac_future = asyncio.gather(*tasks)
        self.monitor_start_event.set()

        try:
            await hvac_future
        except asyncio.CancelledError:
            hvac_future.cancel()
            await asyncio.gather(hvac_future, return_exceptions=True)
            raise
        finally:
            self.monitor_start_event.clear()

    async def close(self) -> None:
        """Cancel any in-flight command tasks."""
        await close_command_tasks(self)

    async def control_ahus_and_vec04(self) -> None:
        cached_shutter_closed = None
        cached_wind_threshold = None

        while True:
            # Check the aperture state
            shutter_closed = self.dome_model.is_closed
            if shutter_closed is None:
                await asyncio.sleep(0.1)
                continue

            enable_device_list = []
            disable_device_list = []

            if (
                "vec04" not in self.features_to_disable
                and not shutter_closed
                and (utils.current_tai() - self.last_vec04_time > self.vec04_hold_time)
            ):
                # Check windspeed threshold
                average_windspeed = self.weather_model.average_windspeed
                wind_threshold = average_windspeed < self.wind_threshold
                if wind_threshold != cached_wind_threshold:
                    change_message = f"VEC-04 operation demanded: {average_windspeed} m/s -> {wind_threshold}"

                    cached_wind_threshold = wind_threshold
                    self.last_vec04_time = utils.current_tai()
                    if wind_threshold:
                        self.log.info(f"Turning on VEC-04 fan! {change_message}")
                        enable_device_list.append(DeviceId.loadingBayFan04P04)
                    else:
                        self.log.info(f"Turning off VEC-04 fan! {change_message}")
                        disable_device_list.append(DeviceId.loadingBayFan04P04)

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
                        enable_device_list.extend(ahus)

                    if "vec04" not in self.features_to_disable:
                        # Disable the VEC-04 fan
                        self.log.info("Turning off VEC-04 fan!")
                        disable_device_list.append(DeviceId.loadingBayFan04P04)
                        self.last_vec04_time = utils.current_tai()
                else:
                    if "ahu" not in self.features_to_disable:
                        self.log.info("Disabling HVAC AHUs!")
                        disable_device_list.extend(ahus)

            if disable_device_list:
                await self.disable_devices(disable_device_list)
            if enable_device_list:
                await self.enable_devices(enable_device_list)
            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def wait_for_sunrise(self) -> None:
        """Wait for sunrise and then set the room temperature.

        Wait for the timer to signal sunrise, and then obtain the
        temperature that was reported last night at the end
        of twilight, and then apply that temperature as at AHU
        setpoint.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.sunrise_condition:
                await self.diurnal_timer.sunrise_condition.wait()
                last_twilight_temperature = await self.weather_model.get_last_twilight_temperature()
                if self.diurnal_timer.is_running and last_twilight_temperature is not None:
                    if "room_setpoint" not in self.features_to_disable:
                        # Time to set the room setpoint based on last twilight
                        setpoint = max(
                            last_twilight_temperature + self.ahu_setpoint_delta,
                            self.setpoint_lower_limit,
                        )

                        await self.config_lower_ahu(
                            [
                                {
                                    "device_id": device_id,
                                    "workingSetpoint": setpoint,
                                    "maxFanSetpoint": math.nan,
                                    "minFanSetpoint": math.nan,
                                    "antiFreezeTemperature": math.nan,
                                }
                                for device_id in (
                                    DeviceId.lowerAHU01P05,
                                    DeviceId.lowerAHU02P05,
                                    DeviceId.lowerAHU03P05,
                                    DeviceId.lowerAHU04P05,
                                )
                            ]
                        )

    def compute_glycol_setpoints(self, ambient_temperature: float) -> tuple[float, float]:
        """Compute staggered glycol chiller setpoints.

        Compute staggered glycol chiller setpoints based on ambient
        temperature, configured band limits, dew point, and absolute
        minimum and maximum constraint.

        The algorithm enforces:
          * Average setpoint nominally `ambient + glycol_average_offset`
            (float, with `glycol_average_offset` being negative).
          * Raised if necessary to exceed the nightly maximum indoor dew
            point plus a safety margin (`dew_point_margin`: `float`).
          * Split into two staggered setpoints
            (`setpoint1`: `float`, `setpoint2`: `float`)
            separated by `glycol_setpoints_delta`: `float`, **with chiller 1
            warmer**.
          * Absolute minimum enforced on the colder chiller
            (`setpoint2` : `float` >= `glycol_absolute_minimum`: `float`),
            adjusting both setpoints to preserve the delta.
          * Similar constraints for the absolute maximum temperature.

        Parameters
        ----------
        ambient_temperature : `float`
            Ambient temperature in degrees Celsius. This value
            is used to determine the nominal target band for the glycol
            loop average.

        Returns
        -------
        setpoint1 : `float`
            Active setpoint for chiller 1 (°C), the warmer of the two.
        setpoint2 : `float`
            Active setpoint for chiller 2 (°C), the colder of the two.
        """
        # Compute a target average setpoint
        target_average = ambient_temperature + self.glycol_average_offset

        # Incorporate dew point into the calculation - setpoint
        # average should not be lower than the dew point (with margin)
        nightly_maximum_dew_point = self.weather_model.nightly_maximum_indoor_dew_point
        if nightly_maximum_dew_point is not None:
            target_average = max(target_average, nightly_maximum_dew_point + self.glycol_dew_point_margin)

        # Break average and delta into individual setpoints. The two
        # setpoints should have the computed average and differ
        # with each other by `glycol_setpoints_delta`.
        setpoint1, setpoint2 = target_average, target_average
        setpoint1 += self.glycol_setpoints_delta / N_CHILLERS
        setpoint2 -= self.glycol_setpoints_delta / N_CHILLERS

        # Enforce the absolute minimum temperature
        if setpoint2 < self.glycol_absolute_minimum:
            setpoint2 = self.glycol_absolute_minimum
            setpoint1 = setpoint2 + self.glycol_setpoints_delta

        # Enforce the absolute maximum temperature
        if setpoint1 > self.glycol_absolute_maximum:
            setpoint1 = self.glycol_absolute_maximum
            setpoint2 = setpoint1 - self.glycol_setpoints_delta

        return setpoint1, setpoint2

    def check_glycol_setpoint(self, ambient_temperature: float) -> bool:
        """Verify whether the chiller setpoints are within the allowed band.

        Parameters
        ----------
        ambient_temperature : `float`
            Ambient temperature (°C)

        Returns
        -------
        bool
            True if the current setpoints are acceptable, or False otherwise.
        """
        # This test makes mypy happy:
        if self.glycol_setpoint1 is None or self.glycol_setpoint2 is None:
            return False

        # Find the average of the two glycol setpoints.
        average_setpoint = (self.glycol_setpoint1 + self.glycol_setpoint2) / N_CHILLERS

        # The difference between the average and the current ambient
        # reading must not fall below `glycol_band_low` or above
        # `glycol_band_high`.
        return self.glycol_band_low <= average_setpoint - ambient_temperature <= self.glycol_band_high

    async def monitor_glycol_chillers(self) -> None:
        """Continuously monitor and enforce glycol chiller setpoints.

        This coroutine runs while the diurnal timer is active. On each cycle it
        checks whether the current chiller setpoints are within the allowed
        band relative to the ambient indoor temperature. If the setpoints are
        outside of the band, new setpoints are computed and applied to the
        HVAC CSC.
        """
        self.log.debug("monitor_glycol_chillers")
        while self.diurnal_timer.is_running:
            try:
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
                if ambient_temperature is not None and not self.check_glycol_setpoint(ambient_temperature):
                    self.log.debug("Recomputing glycol setpoints.")
                    glycol_setpoint1, glycol_setpoint2 = self.compute_glycol_setpoints(ambient_temperature)

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

                chiller_commands = []
                if self.glycol_setpoint1 is not None:
                    chiller_commands.append(
                        {
                            "device_id": DeviceId.chiller01P01,
                            "activeSetpoint": self.glycol_setpoint1,
                        }
                    )
                if self.glycol_setpoint2 is not None:
                    chiller_commands.append(
                        {
                            "device_id": DeviceId.chiller02P01,
                            "activeSetpoint": self.glycol_setpoint2,
                        }
                    )
                if chiller_commands:
                    await self.config_chiller(chiller_commands)
            except Exception:
                self.log.exception("In HVAC glycol control loop")

            await asyncio.sleep(HVAC_SLEEP_TIME)

    async def adjust_glycol_chillers_at_noon(self) -> None:
        """Wait for noon and then sets the glycol chillers.

        Wait for the timer to signal noon, and then obtain the minimum
        temperature that was reported last night, and then apply an
        appropriate temperature as the glycol setpoint.
        """
        while self.diurnal_timer.is_running:
            async with self.diurnal_timer.noon_condition:
                await self.diurnal_timer.noon_condition.wait()
                if not self.diurnal_timer.is_running:
                    return

                nightly_minimum_temperature = self.weather_model.nightly_minimum_temperature
                if math.isnan(nightly_minimum_temperature):
                    self.log.error("Nightly minimum temperature was not available.")
                    continue

                self.glycol_setpoint1, self.glycol_setpoint2 = self.compute_glycol_setpoints(
                    nightly_minimum_temperature
                )

                if self.glycol_setpoint1 is None or self.glycol_setpoint2 is None:
                    self.log.error("Failed to calculate noon glycol setpoints.")
                    continue

                await self.config_chiller(
                    [
                        {
                            "device_id": DeviceId.chiller01P01,
                            "activeSetpoint": self.glycol_setpoint1,
                        },
                        {
                            "device_id": DeviceId.chiller02P01,
                            "activeSetpoint": self.glycol_setpoint2,
                        },
                    ]
                )

    async def apply_setpoint_at_night(self) -> None:
        """Control the HVAC setpoint during the night.

        At night time (defined by `DiurnalTimer.is_night`) the HVAC
        AHU setpoint should be applied based on the outside temperature,
        if the dome is closed. If the dome is open, the HVAC AHUs
        should not be enabled, and the setpoint should not matter.
        """
        warned_no_temperature = False

        while self.diurnal_timer.is_running:
            if self.diurnal_timer.is_night(Time.now()) and self.dome_model.is_closed:
                setpoint = max(
                    self.weather_model.current_temperature + self.ahu_setpoint_delta,
                    self.setpoint_lower_limit,
                )
                if math.isnan(setpoint):
                    if not warned_no_temperature:
                        self.log.warning("Failed to collect a temperature sample for HVAC setpoint.")
                        warned_no_temperature = True

                else:
                    # Apply setpoint for each of the 4 AHUs
                    await self.config_lower_ahu(
                        [
                            {
                                "device_id": device_id,
                                "workingSetpoint": setpoint,
                                "maxFanSetpoint": math.nan,
                                "minFanSetpoint": math.nan,
                                "antiFreezeTemperature": math.nan,
                            }
                            for device_id in (
                                DeviceId.lowerAHU01P05,
                                DeviceId.lowerAHU02P05,
                                DeviceId.lowerAHU03P05,
                                DeviceId.lowerAHU04P05,
                            )
                        ]
                    )

            await asyncio.sleep(HVAC_SLEEP_TIME)
