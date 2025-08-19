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

__all__ = ["CONFIG_SCHEMA"]

import yaml

CONFIG_SCHEMA = yaml.safe_load(
    """
    $schema: http://json-schema.org/draft-07/schema#
    $id: https://github.com/lsst-ts/ts_eas/blob/main/python/lsst/ts/eas/config_schema.py
    # title must end with one or more spaces followed by the schema version, which must begin with "v"
    title: EAS v8
    description: Schema for EAS configuration files
    type: object
    properties:
      wind_threshold:
        description: Windspeed limit for the VEC-04 fan (m/s)
        type: number
        exclusiveMinimum: 0
      wind_average_window:
        description: Time over which to average windspeed for threshold determination (s)
        type: number
        exclusiveMinimum: 0
      wind_minimum_window:
        description: Minimum amount of time to collect wind data before acting on it. (s)
        type: number
        exclusiveMinimum: 0
      vec04_hold_time:
        description: >
          Minimum time to wait before changing the state of the VEC-04 fan. This
          value is ignored if the dome is opened or closed. (s)
        type: number
        exclusiveMinimum: 0
      m1m3_setpoint_cadence:
        description: >
          Frequency at which to send commands to MTM1M3TS to change the setpoint. (s)
        type: number
        exclusiveMinimum: 0
      features_to_disable:
        description: List of CSC control points to disable
        type: array
        items:
          type: string
      twilight_definition:
        description: >
          Definition of twilight. Can be a number (in degrees) between -90 and 0, corresponding
          to sun elevation, or one of the strings: "civil", "nautical", "astronomical"
        oneOf:
          - type: number
            minimum: -90
            maximum: 0
          - type: string
            enum: ["civil", "nautical", "astronomical"]
      weather_ess_index:
        description: SAL index for the CSC providing weather information
        type: integer
        minimum: 0
      indoor_ess_index:
        description: SAL index for the CSC providing indoor temperature measurements.
        type: integer
        minimum: 0
      ess_timeout:
        description: >
          The amount of time (seconds) of no ESS measurements after which
          the CSC should fault.
        type: number
      glycol_setpoint_delta:
        description: >
          Offset between desired ambient setpoint and M1M3TS glycol setpoint (°C)
        type: number
      heater_setpoint_delta:
        description: >
          Offset between desired ambient setpoint and MTM3TS FCU heater setpoint (°C)
        type: number
      top_end_setpoint_delta:
        description: >
          Offset between measured indoor temperature and the top end setpoint (°C)
        type: number
      setpoint_deadband_heating:
        description: >
          Deadband for M1M3TS heating. If the the new setpoint exceeds the previous
          setpoint by less than this amount, no new command is sent. (°C)
      setpoint_deadband_cooling:
        description: >
          Deadband for M1M3TS cooling. If the new setpoint is lower than the previous
          setpoint by less than this amount, no new command is sent. (°C)
        type: number
        minimum: 0
      maximum_heating_rate:
        description: >
          Maximum allowed rate of increase in the M1M3TS setpoint temperature. Limits
          how quickly the setpoint can rise, in degrees Celsius per hour. (°C/hr)
        type: number
        minimum: 0
      slow_cooling_rate:
        description: >
          Cooling rate to be used shortly before and during the night.
          Limits how quickly the setpoint can fall, in degrees Celsius per hour.
        type: number
        minimum: 0
      fast_cooling_rate:
        description: >
          Cooling rate to be used during the day. Limits how quickly the setpoint
          can fall, in degrees Celsius per hour.
        type: number
        minimum: 0
      setpoint_lower_limit:
        description: >
          The minimum allowed setpoint for thermal control. If a lower setpoint
          than this is indicated from the ESS temperature readings, this setpoint
          will be used instead.
      efd_name:
         description: Name of the EFD instance telemetry should be queried from.
         type: string

    required:
      - wind_threshold
      - wind_average_window
      - wind_minimum_window
      - vec04_hold_time
      - m1m3_setpoint_cadence
      - features_to_disable
      - twilight_definition
      - weather_ess_index
      - indoor_ess_index
      - ess_timeout
      - glycol_setpoint_delta
      - heater_setpoint_delta
      - top_end_setpoint_delta
      - setpoint_deadband_heating
      - setpoint_deadband_cooling
      - maximum_heating_rate
      - slow_cooling_rate
      - fast_cooling_rate
      - setpoint_lower_limit
      - efd_name
    additionalProperties: false
    """
)
