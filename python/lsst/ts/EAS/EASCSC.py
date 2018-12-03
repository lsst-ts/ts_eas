# This file is part of ts_eas.
#
# Developed for the LSST Telescope and Site Systems.
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

__all__ = ["EAS"]
import asyncio
import random
import time

import SALPY_EAS
import lsst.ts.salobj as salobj
#from lsst.ts.salobj import *

class configEAS:
    telemetryInterval = 0.5 #seconds
    eventsInterval = 300.0 #seconds
    
class EASCsc(salobj.BaseCsc):
    """CSC to deliver EAS Telemetery and events.
       it currently issues and takes no commands.

    """
    
    def __init__(self, index=None, initial_state=salobj.base_csc.State.STANDBY):
        if initial_state not in salobj.base_csc.State:
            raise ValueError("intial_state={initial_state} is not a salobj.State enum")

        super().__init__(SALPY_EAS, index)
        self.summary_state = initial_state
        self.conf = configEAS()
        
        #
        # setup event and telemetry loops
        #
        self.telTask = None
        self.evtTask = None
        #
        # set up event data structures
        #
        self.evt_detailedState_data = self.evt_detailedState.DataType()
        self.evt_internalCommand_data = self.evt_internalCommand.DataType()
        self.evt_heartbeat_data = self.evt_heartbeat.DataType()
        self.evt_loopTimeOutOfRange_data = self.evt_loopTimeOutOfRange.DataType()
        self.evt_rejectedCommand_data = self.evt_rejectedCommand.DataType()
        #
        # set up telemetry data structures
        #
        self.tel_timestamp_data = self.tel_timestamp.DataType()
        self.tel_loopTime_data = self.tel_loopTime.DataType()
        self.tel_airTemperature_data = self.tel_airTemperature.DataType()
        self.tel_surfaceTemperature_data = self.tel_surfaceTemperature.DataType()
        self.tel_humidity_data = self.tel_humidity.DataType()
        self.tel_barometricPressure_data = self.tel_barometricPressure.DataType()
        self.tel_airflowDynamics_data = self.tel_airflowDynamics.DataType()
        self.tel_bulkAirflow_data = self.tel_bulkAirflow.DataType()
        self.tel_radioFrequencyInterference_data = self.tel_radioFrequencyInterference.DataType()
        self.tel_currentLeakage_data = self.tel_currentLeakage.DataType()
        self.tel_weatherStation_data = self.tel_weatherStation.DataType()
        self.tel_accelrometers_data = self.tel_accelrometers.DataType()
        self.tel_interiorAcoustics_data = self.tel_interiorAcoustics.DataType()
        self.tel_sonicAnemometers_data = self.tel_sonicAnemometers.DataType()


    def do_start(self, id_data):
        super().do_start(id_data)
        #
        # start the telemetry loop as a task. It won't actually send telemetry
        # unless the CSC is in the DISABLED or ENABLED states

        print('starting telemetryLoop')
        asyncio.ensure_future(self.telemetryLoop())
        
        print('starting EventLoop')
        asyncio.ensure_future(self.eventLoop())

    ###############################################################
    # Loop sending Telemetry every conf.telemeteryInterval seconds.
    async def telemetryLoop(self):
        if self.telTask and not self.telTask.done():
            self.telTask.cancel()
        
        while self.summary_state in (salobj.base_csc.State.DISABLED, salobj.base_csc.State.ENABLED):
            self.telTask = await asyncio.sleep(self.conf.telemetryInterval)
            self.sendTelemetry()

    def sendTelemetry(self):
        print('sendTelemetry: ', '{:.4f}'.format(time.time()))

        self.tel_timestamp_data.timestamp = time.time()
        self.tel_timestamp.put(self.tel_timestamp_data)
        
        self.tel_loopTime_data.loopTime = 0.5
        self.tel_loopTime.put(self.tel_loopTime_data)
        
        for i in range(len(self.tel_airTemperature_data.airTemperature)):
            self.tel_airTemperature_data.airTemperature[i] = random.uniform(0.0, 85.0)  
        self.tel_airTemperature.put(self.tel_airTemperature_data)
        
        for i in range(len(self.tel_surfaceTemperature_data.surfaceTemperature)):
            self.tel_surfaceTemperature_data.surfaceTemperature[i] = random.uniform(0.0, 85.0) 
        self.tel_surfaceTemperature.put(self.tel_surfaceTemperature_data)
        
        for i in range(len(self.tel_humidity_data.humidity)):
            self.tel_humidity_data.humidity[i] = random.uniform(30.0, 33.0) 
        self.tel_humidity.put(self.tel_humidity_data)
        
        self.tel_barometricPressure_data.barometricPressure = random.uniform(30.0, 60.0)  
        self.tel_barometricPressure.put(self.tel_barometricPressure_data)
        
        for i in range(len(self.tel_airflowDynamics_data.airflowDynamics)):
            self.tel_airflowDynamics_data.airflowDynamics[i] = random.uniform(5.0, 15.0)
        self.tel_airflowDynamics.put(self.tel_airflowDynamics_data)
        
        for i in range(len(self.tel_bulkAirflow_data.bulkAirflow)):
            self.tel_bulkAirflow_data.bulkAirflow[i] = random.uniform(5.0, 15.0)
        self.tel_bulkAirflow.put(self.tel_bulkAirflow_data)
        
        self.tel_radioFrequencyInterference_data.radioFrequencyInterference = random.uniform(0.1, 20.0)  
        self.tel_radioFrequencyInterference.put(self.tel_radioFrequencyInterference_data)
        
        self.tel_currentLeakage_data.currentLeakage = random.uniform(0.0, 1.0)
        self.tel_currentLeakage.put(self.tel_currentLeakage_data)
        
        self.tel_weatherStation_data.weatherStation = random.uniform(0.0, 100.0)
        self.tel_weatherStation.put(self.tel_weatherStation_data)
        
        for i in range(len(self.tel_accelrometers_data.accelrometers)):
            self.tel_accelrometers_data.accelrometers[i] = random.uniform(0.0, 0.9)  
        self.tel_accelrometers.put(self.tel_accelrometers_data)
        
        self.tel_interiorAcoustics_data.interiorAcoustics = random.uniform(0.0, 10.0) 
        self.tel_interiorAcoustics.put(self.tel_interiorAcoustics_data)
        
        for i in range(len(self.tel_sonicAnemometers_data.sonicAnemometers)):
            self.tel_sonicAnemometers_data.sonicAnemometers[i] = random.uniform(0.0, 10.0)
        self.tel_sonicAnemometers.put(self.tel_sonicAnemometers_data)
        
    ###############################################################
    # Loop sending Events every conf.eventsInterval seconds.
    async def eventLoop(self):
        if self.evtTask and not self.evtTask.done():
            self.evtTask.cancel()
        
        while self.summary_state in (salobj.base_csc.State.DISABLED, salobj.base_csc.State.ENABLED):
            self.evtTask = await asyncio.sleep(self.conf.eventsInterval)
            self.sendEvents()


    def sendEvents(self):
        print('sendEvents: ', '{:.4f}'.format(time.time()))


