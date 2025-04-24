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

__all__ = ["EasCsc", "run_eas"]

import asyncio
import typing
from types import SimpleNamespace

from lsst.ts import salobj

from . import __version__
from .config_schema import CONFIG_SCHEMA


def run_eas() -> None:
    asyncio.run(EasCsc.amain(index=None))


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
    override : `str`, optional
        Override of settings if ``initial_state`` is `State.DISABLED`
        or `State.ENABLED`.
    """

    valid_simulation_modes = (0, 1)
    version = __version__

    def __init__(
        self,
        config_dir: typing.Optional[str] = None,
        initial_state: salobj.State = salobj.State.STANDBY,
        simulation_mode: int = 0,
        override: str = "",
    ) -> None:
        self.config: typing.Optional[SimpleNamespace] = None
        self._config_dir = config_dir
        super().__init__(
            name="EAS",
            index=0,
            config_schema=CONFIG_SCHEMA,
            config_dir=config_dir,
            initial_state=initial_state,
            simulation_mode=simulation_mode,
            override=override,
        )

    async def handle_summary_state(self) -> None:
        """Override of the handle_summary_state function to
        set up the control loop.
        """
        self.log.debug(f"handle_summary_state {salobj.State(self.summary_state).name}")

        if self.disabled_or_enabled:
            pass

        else:
            pass

    async def configure(self, config: SimpleNamespace) -> None:
        self.config = config

    @staticmethod
    def get_config_pkg() -> str:
        return "ts_config_ocs"
