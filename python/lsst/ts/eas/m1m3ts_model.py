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

__all__ = ["M1M3TSModel"]

import asyncio
import logging

from lsst.ts import salobj, utils

STD_TIMEOUT = 10  # seconds
M1M3_TEMPERATURE_CADENCE = (
    60  # Time over which to collect M1M3 temperature measurements (sec)
)


class M1M3TSModel:
    """A model for the M1M3TS system automation.

    Parameters
    ----------
    domain : salobj.Domain
        A SAL domain object for obtaining remotes.
    log : logging.Logger
        A logger for log messages.
    glycol_setpoint_delta : float
        The difference between the twilight ambient temperature and the
        setpoint to apply for the glycol, e.g., -2 if the glycol should
        be two degrees cooler than ambient.
    heater_setpoint_delta : float
        The difference between the twilight ambient temperature and the
        setpoint to apply for the FCU heaters, e.g., -1 if the FCU heaters
        should be one degree cooler than ambient.
    features_to_disable : list[str]
        A list of features that should be disabled. The following strings can
        be used:
         * m1m3ts
        Any other values are ignored.
    """

    def __init__(
        self,
        *,
        domain: salobj.Domain,
        log: logging.Logger,
        glycol_setpoint_delta: float,
        heater_setpoint_delta: float,
        features_to_disable: list[str] = [],
    ) -> None:
        self.domain = domain
        self.log = log

        # Configuration parameters:
        self.glycol_setpoint_delta = glycol_setpoint_delta
        self.heater_setpoint_delta = heater_setpoint_delta
        self.features_to_disable = features_to_disable

    async def monitor(self) -> None:
        self.log.debug("HvacModel.monitor")

        async with salobj.Remote(
            domain=self.domain,
            name="MTM1M3TS",
        ) as m1m3ts_remote:
            if "m1m3ts" not in self.features_to_disable:
                await self.follow_ess112(m1m3ts_remote=m1m3ts_remote)
            else:
                while True:
                    try:
                        await asyncio.sleep(300)
                    except asyncio.CancelledError:
                        self.log.info("monitor cancelled")
                        raise

    async def follow_ess112(self, m1m3ts_remote: salobj.Remote) -> None:
        self.log.debug("follow_ess112")

        async with salobj.Remote(
            domain=self.domain, name="ESS", index=112, include=("temperature",)
        ) as ess_remote:
            while True:
                temperatures = []
                start_time = utils.current_tai()

                # Collect average ESS:112 temperature for
                # M1M3_TEMPERATURE_CADENCE seconds.
                while utils.current_tai() - start_time < M1M3_TEMPERATURE_CADENCE:
                    try:
                        sample = await ess_remote.tel_temperature.aget(timeout=10)
                        await asyncio.sleep(0)  # Make sure we get CancelledError
                        temperatures.append(sample.temperatureItem[0])
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self.log.warning(f"Failed to get ESS:112 temperature: {e!r}")

                if not temperatures:
                    self.log.error(
                        "No temperature samples were collected. CSC will fault."
                    )
                    raise RuntimeError("No temperature samples were collected.")

                average_temperature = sum(temperatures) / len(temperatures)
                self.log.debug(
                    f"Collected {len(temperatures)} ESS:112 samples with {average_temperature=}."
                )

                glycol_setpoint = average_temperature + self.glycol_setpoint_delta
                heaters_setpoint = average_temperature + self.heater_setpoint_delta
                self.log.debug(
                    f"Setting MTM1MTS: {glycol_setpoint=} {heaters_setpoint=}"
                )
                if hasattr(m1m3ts_remote, "cmd_applySetpoints"):
                    await m1m3ts_remote.cmd_applySetpoints.set_start(
                        glycolSetpoint=glycol_setpoint,
                        heatersSetpoint=heaters_setpoint,
                    )
                else:
                    await m1m3ts_remote.cmd_applySetpoint.set_start(
                        glycolSetpoint=glycol_setpoint,
                        heatersSetpoint=heaters_setpoint,
                    )
