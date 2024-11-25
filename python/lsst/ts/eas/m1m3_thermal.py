#!/usr/bin/env python
# coding: utf-8

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

import asyncio
import logging

from dds import DDSException
from lsst.ts import salobj

SAL_TIMEOUT = 5.0  # SAL telemetry/command timeout
REMOTE_STARTUP_TIME = 5.0  # Time for remotes to get set up
SUMMARY_STATE_TIME = 5.0  # Wait time for a summary state change
FAN_SLEEP_TIME = 30.0  # Time to wait after changing the fans
VALVE_SLEEP_TIME = 60.0  # Time to wait after changing the valve
DDS_RESTART_TIME = 60.0  # Time to wait after a DDS exception


async def run_control_loop(domain: salobj.Domain, log: logging.Logger) -> None:
    """Runs the control loop for the fans and the heaters.

    Parameters
    ----------
    domain: salobj.Domain
        The SALObj domain instance to use.

    log: logging.Logger
        A logger for log messages.
    """
    heaterdemand = [0] * 96
    fandemand = [30] * 96

    try:
        m1m3ts = salobj.Remote(domain, "MTM1M3TS")
        ess = salobj.Remote(domain, "ESS", index=112)

        # Wait for remotes to get set up...
        await asyncio.sleep(REMOTE_STARTUP_TIME)

        mixing = await m1m3ts.tel_mixingValve.next(flush=True, timeout=SAL_TIMEOUT)
        currentvalveposition = mixing.valvePosition
        oldvalveposition = currentvalveposition

        while True:
            try:
                glycol = await m1m3ts.tel_glycolLoopTemperature.next(
                    flush=True, timeout=SAL_TIMEOUT
                )
                mixing = await m1m3ts.tel_mixingValve.next(
                    flush=True, timeout=SAL_TIMEOUT
                )
                fcu = await m1m3ts.tel_thermalData.next(flush=True, timeout=SAL_TIMEOUT)
                currenttemp = (
                    glycol.insideCellTemperature1
                    + glycol.insideCellTemperature2
                    + glycol.insideCellTemperature3
                ) / 3
                currentvalveposition = mixing.valvePosition

                fcu = await m1m3ts.tel_thermalData.next(flush=True, timeout=SAL_TIMEOUT)
                fanspeed = fcu.fanRPM
                fcutemp = fcu.absoluteTemperature

                airtemp = await ess.tel_temperature.next(
                    flush=True, timeout=SAL_TIMEOUT
                )
                targettemp = airtemp.temperatureItem[0]

                log.info(
                    f"""
                    target cell temp (above air temp): {targettemp}
                    current cell temp: {currenttemp}
                    current valve position: {currentvalveposition}
                    current fan speed: {fanspeed[50]}
                    current FCU temp: {fcutemp[50]}
                    """
                )

                # if the FCUs are off, try to turn them on
                if fanspeed[50] > 60000:
                    log.info(
                        f"fans off, turning them on and waiting {FAN_SLEEP_TIME} seconds..."
                    )
                    await salobj.set_summary_state(
                        m1m3ts,
                        salobj.State.STANDBY,
                        timeout=SAL_TIMEOUT,
                    )
                    await asyncio.sleep(SUMMARY_STATE_TIME)
                    await salobj.set_summary_state(
                        m1m3ts,
                        salobj.State.ENABLED,
                        timeout=SAL_TIMEOUT,
                    )
                    await asyncio.sleep(SUMMARY_STATE_TIME)
                    await m1m3ts.cmd_setEngineeringMode.set_start(
                        enableEngineeringMode=True,
                        timeout=SAL_TIMEOUT,
                    )
                    await m1m3ts.cmd_heaterFanDemand.set_start(
                        heaterPWM=heaterdemand,
                        fanRPM=fandemand,
                        timeout=SAL_TIMEOUT,
                    )
                    await asyncio.sleep(FAN_SLEEP_TIME)
                elif fanspeed[50] < 50:
                    log.info(
                        "fans rpms too low, turning them back up and waiting {FAN_SLEEP_TIME} seconds..."
                    )
                    await salobj.set_summary_state(m1m3ts, salobj.State.STANDBY)
                    await asyncio.sleep(SUMMARY_STATE_TIME)
                    await salobj.set_summary_state(m1m3ts, salobj.State.ENABLED)
                    await asyncio.sleep(SUMMARY_STATE_TIME)
                    await m1m3ts.cmd_setEngineeringMode.set_start(
                        enableEngineeringMode=True,
                        timeout=SAL_TIMEOUT,
                    )
                    await m1m3ts.cmd_heaterFanDemand.set_start(
                        heaterPWM=heaterdemand,
                        fanRPM=fandemand,
                        timeout=SAL_TIMEOUT,
                    )
                    await asyncio.sleep(FAN_SLEEP_TIME)

                if currenttemp - targettemp >= 0.05:
                    newvalveposition = min(10.0, oldvalveposition + 5.0)
                    log.info(
                        f"temp high, adjusting mixing valve to: {newvalveposition}"
                    )
                    await m1m3ts.cmd_setMixingValve.set_start(
                        mixingValveTarget=newvalveposition,
                        timeout=SAL_TIMEOUT,
                    )
                    oldvalveposition = newvalveposition
                    log.debug(f"waiting {VALVE_SLEEP_TIME} seconds...")
                    await asyncio.sleep(VALVE_SLEEP_TIME)
                elif currenttemp - targettemp <= -0.05:
                    newvalveposition = max(0.0, oldvalveposition - 5.0)
                    log.info(f"temp low, adjusting mixing valve to: {newvalveposition}")
                    await m1m3ts.cmd_setMixingValve.set_start(
                        mixingValveTarget=newvalveposition, timeout=5
                    )
                    oldvalveposition = newvalveposition
                    log.debug(f"waiting {VALVE_SLEEP_TIME} seconds...")
                    await asyncio.sleep(VALVE_SLEEP_TIME)
                else:
                    log.debug(f"doing nothing, valve position: {currentvalveposition}")
                    log.debug(f"waiting {VALVE_SLEEP_TIME} seconds for update...")
                    await asyncio.sleep(VALVE_SLEEP_TIME)

            except DDSException:
                log.exception("DDS exception in main loop. Trying again.")
                await asyncio.sleep(DDS_RESTART_TIME)

    except asyncio.CancelledError:
        log.info("M1M3 thermal control loop cancelled.")
        raise
    finally:
        m1m3ts.close()
        ess.close()
