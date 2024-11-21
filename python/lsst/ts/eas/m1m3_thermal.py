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
import os

from astropy.time import Time
from lsst.ts import salobj


class ControlLoopManager:
    def __init__(self, domain: salobj.Domain, log: logging.Logger):
        """
        Initializes the ControlLoopManager.

        Parameters
        ----------
        domain: salobj.Domain
            The SALObj domain instance to use.

        log: logging.Logger
            A logger for log messages.
        """
        self.domain = domain
        self.log = log
        self.m1m3ts = salobj.Remote(domain, "MTM1M3TS")
        self.ess = salobj.Remote(domain, "ESS", index=112)
        self.task = None
        self.stop_event = asyncio.Event()

        os.environ["LSST_DDS_RESPONSIVENESS_TIMEOUT"] = "15s"

    async def stop_control_loop(self) -> None:
        """
        Cancels all running control tasks and waits for them to terminate.
        """
        self.log.info("Stopping control loop...")
        self.stop_event.set()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def run_control_loop(self, heaterdemand: list[int], fandemand: list[int]):
        """Runs the control loop for the fans and the heaters.

        Parameters
        ----------
        heaterdemand: list[int]
            The heater power to be applied when the heater is turned on by
            the MTM1M3TS.heaterFanDemand command. The values range from 0-255,
            with 255 being 100%.
        fandemand: list[int]
            Fan RPM demand for the MTM1M3TS.headerFanDemand command, with
            255 being 100%.
        """
        try:
            # Wait for remotes to get set up...
            await asyncio.sleep(5.0)

            mixing = await self.m1m3ts.tel_mixingValve.next()
            currentvalveposition = mixing.valvePosition
            oldvalveposition = currentvalveposition

            while True:
                glycol = await self.m1m3ts.tel_glycolLoopTemperature.next()
                mixing = await self.m1m3ts.tel_mixingValve.next()
                fcu = await self.m1m3ts.tel_thermalData.next()
                currenttemp = (
                    glycol.insideCellTemperature1
                    + glycol.insideCellTemperature2
                    + glycol.insideCellTemperature3
                ) / 3
                currentvalveposition = mixing.valvePosition

                fcu = await self.m1m3ts.tel_thermalData.next()
                fanspeed = fcu.fanRPM
                fcutemp = fcu.absoluteTemperature

                airtemp = await self.ess.tel_temperature.next()
                targettemp = airtemp.temperatureItem[0]

                date = Time.now()
                self.log.info(f'TAITIME_{date.tai.strftime("%Y%m%d_%H%M")}')
                self.log.info(f"target cell temp (above air temp): {targettemp}")
                self.log.info(f"current cell temp: {currenttemp}")
                self.log.info(f"current valve position: {currentvalveposition}")
                self.log.info(f"current fan speed: {fanspeed[50]}")
                self.log.info(f"current FCU temp: {fcutemp[50]}")

                # if the FCUs are off, try to turn them on
                if fanspeed[50] > 60000:
                    self.log.info("fans off, turning them on and waiting 30 seconds...")
                    await salobj.set_summary_state(self.m1m3ts, salobj.State.STANDBY)
                    await asyncio.sleep(5.0)
                    await salobj.set_summary_state(self.m1m3ts, salobj.State.ENABLED)
                    await asyncio.sleep(5.0)
                    await self.m1m3ts.cmd_setEngineeringMode.set_start(
                        enableEngineeringMode=True
                    )
                    await self.m1m3ts.cmd_heaterFanDemand.set_start(
                        heaterPWM=heaterdemand, fanRPM=fandemand
                    )
                    await asyncio.sleep(30.0)
                elif fanspeed[50] < 50:
                    self.log.info(
                        "fans rpms turned down, turning them back up and waiting 30 seconds..."
                    )
                    await salobj.set_summary_state(self.m1m3ts, salobj.State.STANDBY)
                    await asyncio.sleep(5.0)
                    await salobj.set_summary_state(self.m1m3ts, salobj.State.ENABLED)
                    await asyncio.sleep(5.0)
                    await self.m1m3ts.cmd_setEngineeringMode.set_start(
                        enableEngineeringMode=True
                    )
                    await self.m1m3ts.cmd_heaterFanDemand.set_start(
                        heaterPWM=heaterdemand, fanRPM=fandemand
                    )
                    await asyncio.sleep(30.0)
                if currenttemp - targettemp >= 0.05:
                    newvalveposition = min(10.0, oldvalveposition + 5.0)
                    self.log.info(
                        "temp high, adjusting mixing valve to: {newvalveposition}"
                    )
                    await self.m1m3ts.cmd_setMixingValve.set_start(
                        mixingValveTarget=newvalveposition, timeout=5
                    )
                    oldvalveposition = newvalveposition
                    self.log.info("waiting 60 seconds...")
                    await asyncio.sleep(60)
                elif currenttemp - targettemp <= -0.05:
                    newvalveposition = max(0.0, oldvalveposition - 5.0)
                    self.log.info(
                        "temp low, adjusting mixing valve to: {newvalveposition}"
                    )
                    await self.m1m3ts.cmd_setMixingValve.set_start(
                        mixingValveTarget=newvalveposition, timeout=5
                    )
                    oldvalveposition = newvalveposition
                    self.log.info("waiting 60 seconds...")
                    await asyncio.sleep(60)
                else:
                    self.log.info(
                        "doing nothing, valve position: {currentvalveposition}"
                    )
                    self.log.info("waiting 60 seconds for update...")
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            self.log.info("M1M3 thermal control loop cancelled.")

    async def start_control_loop(self):
        """
        Starts the control loop for fans and valves.
        """
        self.log.info("Waiting 30 seconds for the remotes to get set up...")
        await asyncio.sleep(30.0)

        heaterdemand = [0] * 96
        fandemand = [30] * 96

        self.task = asyncio.create_task(self.run_control_loop(heaterdemand, fandemand))

    async def cleanup(self) -> None:
        """
        Cleans up resources.
        """
        await self.m1m3ts.close()
        await self.ess.close()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        log = logging.getLogger()
        with salobj.Domain() as domain:
            manager = ControlLoopManager(domain, log)

            loop.run_until_complete(manager.start_control_loop())

            print("Control loops are running. Press Ctrl+C to stop.")
            loop.run_forever()
    except KeyboardInterrupt:
        print("Stopping...")
        loop.run_until_complete(manager.stop_control_loop())
    finally:
        loop.run_until_complete(manager.cleanup())
