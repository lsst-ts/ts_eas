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
    title: EAS v3
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
    required:
      - wind_threshold
      - wind_average_window
      - wind_minimum_window
      - vec04_hold_time
    additionalProperties: false
    """
)
