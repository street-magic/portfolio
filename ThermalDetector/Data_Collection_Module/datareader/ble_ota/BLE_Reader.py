from .patientzone_otaclient.patientzone_otaclient.patientzone_otaclient import *
from threading import Thread, Lock
from datetime import datetime


class OtaClient(OtaClient):
    def display_data(self, loop, duration=20, mac=None, listener=None):
        self._dummy_ret = None
        def simple_callback(device, advertisement_data):
            if MANUFACTUR_DATA in advertisement_data.manufacturer_data:
                if not mac or mac.lower() == device.address.lower():
                    manufactureBytes = advertisement_data.manufacturer_data[MANUFACTUR_DATA]
                    if not self._distanceAdv.isComplete():
                        self._distanceAdv.feedWithAdvBytes(manufactureBytes)
                    if self._distanceAdv.isComplete():
                        self._dummy_ret = self._distanceAdv
                        if listener:
                            with listener.lock:
                                listener._message_buffer_roi.append(self._dummy_ret)
                        self._distanceAdv = None
                        self._distanceAdv = DistanceAdv()

                    if not self._thermalAdv.isComplete():
                        self._thermalAdv.feedWithAdvBytes(manufactureBytes)
                    if self._thermalAdv.isComplete():
                        self._dummy_ret = self._thermalAdv
                        if listener:
                            with listener.lock:
                                listener._message_buffer_thermal.append(self._dummy_ret)
                        self._thermalAdv = None
                        self._thermalAdv = ThermalAdv()

        async def run():
            scanner = bleak.BleakScanner()
            scanner.register_detection_callback(simple_callback)
            await scanner.start()
            await asyncio.sleep(duration)
            await scanner.stop()

        loop.run_until_complete(run())

class BLE_Listener():
    '''
    Used to format data in the same format as MQTT
    Only for Dev purposes
    '''
    def __init__(self, mac_address, loop):
        raise NotImplementedError # Outdated class
        self.mac_address = mac_address
        self.client = OtaClient(logLvl=logging.WARN)
        self._message_buffer_roi = []
        self._message_buffer_thermal = []
        self.lock = Lock()

        def thread_worker(loop):
            while True:
                self.client.display_data(loop, mac=self.mac_address, listener=self)
        tt = Thread(target=thread_worker, args=(loop,), daemon=True)
        tt.start()

    def get_message(self):
        ret = None
        with self.lock:
            if len(self._message_buffer_roi):
                message = self._message_buffer_roi[0]
                _rois = [hex(int(x/2)).replace('0x','') for x in message.roisCm]
                _rois = [x if len(x)>1 else '0'+x for x in _rois]
                try:
                    ret = {
                    'sender' : self.mac_address,
                    'received_time': datetime.now(),
                    'payload': {
                        'crc': message.crc,
                        'roi_values' : ''.join(x for x in _rois),
                        'accelerometer_x': message.accelX/1000,
                        'accelerometer_y': message.accelY/1000,
                        'accelerometer_z': message.accelZ/1000,
                        'pir_event_counter': message.pirCount,
                        'battery' : message.batteryMv,
                        'pir': int(bool(message.pirCount))
                        }
                    }
                except:
                    print(message)
                    exit()
                self._message_buffer_roi.pop(0)
                return ret
            if len(self._message_buffer_thermal):
                message = self._message_buffer_thermal[0]
                _rois = [hex(int((x-10)/.25)).replace('0x','') for x in message.roisC]
                _rois = [x if len(x)>1 else '0'+x for x in _rois]
                ret = {
                'sender': self.mac_address,
                'received_time': datetime.now(),
                'payload': {
                    'crc': message.crc,
                    'roi': ''.join(x for x in _rois),
                    }
                }
                self._message_buffer_thermal.pop(0)
                return ret
