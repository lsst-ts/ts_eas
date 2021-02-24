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

__all__ = ["EasCsc"]

from .config_schema import CONFIG_SCHEMA
from . import __version__
from lsst.ts import salobj


class EasCsc(salobj.ConfigurableCsc):
    """Commandable SAL Component for the EAS.

    Parameters
    ----------
    config_dir : `string`
        The configuration directory
    initial_state : `salobj.State`
        The initial state of the CSC
    simulation_mode : `int`
        Simulation mode (1) or not (0)
    """

    valid_simulation_modes = (0, 1)
    version = __version__

    def __init__(
        self, config_dir=None, initial_state=salobj.State.STANDBY, simulation_mode=0,
    ):
        self.config = None
        self._config_dir = config_dir
        super().__init__(
            name="EAS",
            index=0,
            config_schema=CONFIG_SCHEMA,
            config_dir=config_dir,
            initial_state=initial_state,
            simulation_mode=simulation_mode,
        )
        self.eas = None
        self.log.info("__init__")

    async def connect(self):
        """Connect the EAS CSC or start the mock client, if in
        simulation mode.
        """
        self.log.info("Connecting")
        self.log.info(self.config)
        self.log.info(f"self.simulation_mode = {self.simulation_mode}")
        if self.config is None:
            raise RuntimeError("Not yet configured")
        if self.connected:
            raise RuntimeError("Already connected")
        if self.simulation_mode == 1:
            # TODO Add code for simulation case, see DM-26440
            pass
        else:
            # TODO Add code for non-simulation case
            pass
        if self.eas:
            self.eas.connect()

    async def disconnect(self):
        """Disconnect the EAS CSC, if connected.
        """
        self.log.info("Disconnecting")
        if self.eas:
            self.eas.disconnect()

    async def handle_summary_state(self):
        """Override of the handle_summary_state function to connect or
        disconnect to the EAS CSC (or the mock client) when needed.
        """
        self.log.info(f"handle_summary_state {salobj.State(self.summary_state).name}")
        if self.disabled_or_enabled:
            if not self.connected:
                await self.connect()
        else:
            await self.disconnect()

    async def configure(self, config):
        self.config = config

    @property
    def connected(self):
        # TODO Add code to determine if the CSC is connected or not.
        return True

    @staticmethod
    def get_config_pkg():
        return "ts_config_ocs"
