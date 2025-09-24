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
    title: EAS v9
    description: Schema for EAS configuration files.
    type: object
    properties:
      weather:
        description: Weather configuration items.
        type: object
      hvac:
        description: HVAC configuration items.
        type: object
      tma:
        description: TMA configuration items.
        type: object
      features_to_disable:
        description: >-
          List of EAS functionalities to disable. Options are:
            * `room_setpoint`: HVAC setpoints will not be applied.
            * `ahu`: HVAC AHUs will not be enabled / disabled.
            * `vec04`: VEC-04 exhaust fan will not be enabled / disabled.
            * `fanspeed`: MTM1M3TS fans will not be controlled.
            * `m1m3ts`: MTM1M3TS setpoints will not be applied.
            * `require_dome_open`: functionality will operate even when the dome is closed.
        type: array
        items:
          type: string
          enum:
            - room_setpoint
            - ahu
            - vec04
            - fanspeed
            - m1m3ts
            - require_dome_open
      twilight_definition:
        description: >
          Definition of twilight. Can be a number (in degrees) between -90 and 0, corresponding
          to sun elevation, or one of the strings: "civil", "nautical", "astronomical".
        oneOf:
          - type: number
            minimum: -90.0
            maximum: 0.0
          - type: string
            enum: ["civil", "nautical", "astronomical"]

    required:
      - weather
      - hvac
      - tma
      - features_to_disable
      - twilight_definition
    additionalProperties: false
    """
)
