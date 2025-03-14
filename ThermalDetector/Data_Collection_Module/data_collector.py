# from hcb_visualizer import Visualizer
# import os
# import yaml
# import numpy as np
# import pandas as pd
# from datetime import datetime

import os
import yaml
from time import sleep
import traceback
from threading import Thread
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import sys


from hcb_visualizer import Visualizer
from datareader.mqtt_reader import MQTT_Subscriber
from datareader.ble_ota.BLE_Reader import BLE_Listener

from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))


class Datareader:
    def __init__(self, cfg):
        if cfg['READ_FROM'] == 'MQTT':
            topic = [
                (f"chip/{cfg['ZONE_0']['MAC_ADDRESS']}/idle_patientzone", 0),
                (f"chip/{cfg['ZONE_0']['MAC_ADDRESS']}/data/patient_zone/thermal/event", 0),
                # ----
                (f"chip/{cfg['ZONE_1']['MAC_ADDRESS']}/idle_patientzone", 0),
                (f"chip/{cfg['ZONE_2']['MAC_ADDRESS']}/idle_patientzone", 0),
                (f"chip/{cfg['ZONE_3']['MAC_ADDRESS']}/idle_patientzone", 0),
                # ---
                (f"chip/{cfg['ZONE_4']['MAC_ADDRESS']}/idle_patientzone", 0),
                (f"chip/{cfg['ZONE_4']['MAC_ADDRESS']}/data/patient_zone/thermal/event", 0),
            ]
            self.data_reader = MQTT_Subscriber(topic, cfg['MQTT'])
        elif cfg['READ_FROM'] == 'BLE':
            self.data_reader = BLE_Listener(
                cfg['ZONE_0']['MAC_ADDRESS'], asyncio.get_event_loop())
        else:
            raise NotImplementedError

    def get_message(self):
        return self.data_reader.get_message()


def _get_config(filename='config.yml'):

    cfg_file = os.path.join(current_dir, filename)
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f'{filename} not found')
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert cfg['READ_FROM'] in ['BLE', 'MQTT'], 'invalid READ_FROM value'

    return cfg


thermal_reading_arr = []
thermal_reading_rec_time = []

thermal_2_reading_arr = []
thermal_2_reading_rec_time = []

Tof_BIG_reading_arr = []
Tof_BIG_reading_rec_time = []

Tof_OLD_reading_arr = []
Tof_OLD_reading_rec_time = []

Tof_FEET_reading_arr = []
Tof_FEET_reading_rec_time = []

Tof_RIGHT_reading_arr = []
Tof_RIGHT_reading_rec_time = []

Tof_LEFT_reading_arr = []
Tof_LEFT_reading_rec_time = []

rotation = []
angles = []

dirname = sys.argv[1]

m0 = {'sender': 'E6:F9:C6:94:6E:E9', 'payload': {'luminance': 0, 'battery_voltage': 2498, 'roi_values': '', 'accelerometer_y': 0.0, 'accelerometer_x': -343.75, 'captured_by': [
    'datahub/e3efb8b4/scanresults'], 'accelerometer_z': -921.875, 'raw': '0200C2090000D4008A0F00', 'captured_at': '2022-03-17T16:03:16.650Z', 'luminance_in_mv': 15, 'pir_triggered': False, 'pir_error': True, 'crc': True, 'pir_event_counter': 0}, 'received_time': '2022-03-17 17:03:17.530654'}
m1 = {'sender': 'E6:F9:C6:94:6E:E9',
      'payload': {
          'crc': True,
          'captured_by': ['datahub/e3efb8b4/scanresults'],
          'captured_at': '2022-03-17T16:02:37.800Z',
          'type': 'ToF_ROI',
          'roi': '459FA0767676799CA1956D6F6F6F6E998F8D696A6A6A6A918A896566666665898484626464636285807F7F807D7C81817B7A7B7B7C7B7C7C7877777777777779',
          'version': 1
      },
      'received_time': '2022-03-17 17:02:39.588711'
      }
m2 = {'sender': 'E6:F9:C6:94:6E:E9',
      'payload': {
          'crc': True,
          'captured_by': ['datahub/ab39f170/scanresults'],
          'captured_at': '2022-03-17T15:59:56.983Z',
          'type': 'THERMAL_ROI',
          'roi': '413B36303333353442363334333032373C353434353331333A373637353431333B373436353532324038353434323335443B36353535353B47433D39373A3C3C',
          'version': 1
      },
      'received_time': '2022-03-17 16:59:59.523996'
      }

m3 = {'sender': 'DF:37:5B:6B:FA:A9', 'payload': {'luminance': 0, 'battery_voltage': 3032, 'roi_values': '69696A696768696969', 'accelerometer_y': 15.625, 'accelerometer_x': -15.625, 'captured_by': [
    'datahub/e357e07e/scanresults'], 'accelerometer_z': -984.375, 'raw': '010969696A696768696969D80B0100FE02822F00', 'captured_at': '2022-05-05T09:53:14.734Z', 'luminance_in_mv': 47, 'pir_triggered': True, 'pir_error': False, 'crc': True, 'pir_event_counter': 1}, 'received_time': '2022-05-05 11:53:13.254927'}

messages = [m3, m1, m2, m0]


def main():

    cfg = _get_config()
    listener = Datareader(cfg)
    viz = Visualizer(cfg)

    def _updater():
        #        for message in messages:
        while True:
            if viz.need_to_save:
                time_now = Path(
                    dirname+"/"+str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
                time_now.mkdir(parents=True, exist_ok=True)

                if len(thermal_reading_arr) >= 1:
                    df = pd.DataFrame(thermal_reading_arr).add_prefix('TZ_')
                    df['received_time'] = thermal_reading_rec_time
                    df.to_csv(time_now/"EYE_new.csv", index=False)

                if len(thermal_2_reading_arr) >= 1:
                    df = pd.DataFrame(thermal_2_reading_arr).add_prefix('TZ_')
                    df['received_time'] = thermal_2_reading_rec_time
                    df.to_csv(time_now/"EYE_old.csv", index=False)

                if len(Tof_BIG_reading_arr) >= 1:
                    df = pd.DataFrame(Tof_BIG_reading_arr).add_prefix('ToFZ_')
                    df['received_time'] = Tof_BIG_reading_rec_time
                    df.to_csv(time_now/"TOF_BIG.csv", index=False)

                if len(Tof_OLD_reading_arr) >= 1:   
                    df = pd.DataFrame(Tof_OLD_reading_arr).add_prefix('ToFZ_')
                    df['received_time'] = Tof_OLD_reading_rec_time
                    df.to_csv(time_now/"TOF_SMALL.csv", index=False)

                if len(Tof_FEET_reading_arr) >= 1:
                    df = pd.DataFrame(Tof_FEET_reading_arr).add_prefix('ToFZ_')
                    df['received_time'] = Tof_FEET_reading_rec_time
                    df.to_csv(time_now/"TOF_FEET.csv", index=False)

                if len(Tof_RIGHT_reading_arr) >= 1:
                    df = pd.DataFrame(Tof_RIGHT_reading_arr).add_prefix('ToFZ_')
                    df['received_time'] = Tof_RIGHT_reading_rec_time
                    df.to_csv(time_now/"TOF_RIGHT.csv", index=False)

                if len(Tof_LEFT_reading_arr) >= 1:
                    df = pd.DataFrame(Tof_LEFT_reading_arr).add_prefix('ToFZ_')
                    df['received_time'] = Tof_LEFT_reading_rec_time
                    df.to_csv(time_now/"TOF_LEFT.csv", index=False)

                if len(rotation) >= 1:
                    df = pd.DataFrame(rotation)
                    df.columns = ['sender', 'x', 'y', 'z']
                    df.to_csv(time_now/"Accel.csv", index=False)
                if len(angles) >= 1:
                    df = pd.DataFrame(angles)
                    df.columns = ['sender', 'angle']
                    df.to_csv(time_now/"Angles.csv", index=False)

                viz.need_to_save = False
                viz.is_recording = False
                viz._ax.set_title('')
                print(
                    '\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nDATA SAVED\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
                thermal_reading_arr.clear()
                thermal_reading_rec_time.clear()

                thermal_2_reading_arr.clear()
                thermal_2_reading_rec_time.clear()

                Tof_BIG_reading_arr.clear()
                Tof_BIG_reading_rec_time.clear()

                Tof_OLD_reading_arr.clear()
                Tof_OLD_reading_rec_time.clear()

                Tof_FEET_reading_arr.clear()
                Tof_FEET_reading_rec_time.clear()

                Tof_RIGHT_reading_arr.clear()
                Tof_RIGHT_reading_rec_time.clear()

                Tof_LEFT_reading_arr.clear()
                Tof_LEFT_reading_rec_time.clear()

                rotation.clear()
                angles.clear()
            message = listener.get_message()
            if message:
                try:
                    if 'crc' in message['payload'].keys():
                        if message['payload']['crc'] != True:
                            print(
                                '\nChecksum error - packet skipped\n', flush=True)
                            continue
                    print(message, flush=True)
                    viz.update_zone(message['sender'], message['payload'])
                    print("~~~~~~~~~~~~~~~~")
                    if viz.get_if_recording():
                        print('--- recorded data chunck ---')
                        if 'type' in message['payload']:
                            if message['payload']['type'] == 'THERMAL_ROI':
                                roi_values = [int(x+y, 16)*.25+10 for x,
                                              y in zip(message['payload']['roi'][::2], message['payload']['roi'][1::2])]
                                if message['sender'] == cfg['ZONE_0']['MAC_ADDRESS']:
                                    thermal_reading_arr.append(roi_values)
                                    thermal_reading_rec_time.append(
                                        message['received_time'])
                                elif message['sender'] == cfg['ZONE_4']['MAC_ADDRESS']:
                                    thermal_2_reading_arr.append(roi_values)
                                    thermal_2_reading_rec_time.append(
                                        message['received_time'])
                            elif message['payload']['type'] == 'ToF_ROI':
                                roi_values = [int(x+y, 16)*2 for x,
                                              y in zip(message['payload']['roi'][::2], message['payload']['roi'][1::2])]
                                if message['sender'] == cfg['ZONE_0']['MAC_ADDRESS']:
                                    Tof_BIG_reading_arr.append(roi_values)
                                    Tof_BIG_reading_rec_time.append(
                                        message['received_time'])
                                if message['sender'] == cfg['ZONE_4']['MAC_ADDRESS']:
                                    Tof_OLD_reading_arr.append(roi_values)
                                    Tof_OLD_reading_rec_time.append(
                                        message['received_time'])
                        elif 'roi_values' in message['payload'].keys():
                            if(len(message['payload']['roi_values'])):
                                roi_values = [int(x+y, 16)*2 for x,
                                              y in zip(message['payload']['roi_values'][::2], message['payload']['roi_values'][1::2])]
                                if message['sender'] == cfg['ZONE_0']['MAC_ADDRESS']:
                                    Tof_BIG_reading_arr.append(roi_values)
                                    Tof_BIG_reading_rec_time.append(
                                        message['received_time'])
                                elif message['sender'] == cfg['ZONE_1']['MAC_ADDRESS']:
                                    Tof_FEET_reading_arr.append(roi_values)
                                    Tof_FEET_reading_rec_time.append(
                                        message['received_time'])
                                elif message['sender'] == cfg['ZONE_2']['MAC_ADDRESS']:
                                    Tof_RIGHT_reading_arr.append(roi_values)
                                    Tof_RIGHT_reading_rec_time.append(
                                        message['received_time'])
                                elif message['sender'] == cfg['ZONE_3']['MAC_ADDRESS']:
                                    Tof_LEFT_reading_arr.append(roi_values)
                                    Tof_LEFT_reading_rec_time.append(
                                        message['received_time'])
                                elif message['sender'] == cfg['ZONE_4']['MAC_ADDRESS']:
                                    Tof_OLD_reading_arr.append(roi_values)
                                    Tof_OLD_reading_rec_time.append(
                                        message['received_time'])
                            # rotation vector here<<<<
                            accel_readings = [float(message['payload']['accelerometer_x'])/1000, float(
                                message['payload']['accelerometer_y'])/1000, float(message['payload']['accelerometer_z'])/1000]
                            rotation.append([message['sender']]+accel_readings)

                            # x, y, z - measured in G's
                            reference_down_vector = [0, 0, -1]
                            accel_vector = accel_readings / \
                                np.linalg.norm(accel_readings)
                            down_inclination = np.degrees(
                                np.arccos(np.clip(np.dot(reference_down_vector, accel_vector), -1.0, 1.0)))
                            if np.degrees(np.arccos(np.clip(np.dot([1, 0, 0], accel_vector), -1.0, 1.0))) - 90 < 0:
                                down_inclination = down_inclination

                            angles.append(
                                [message['sender'], down_inclination])
                        elif 'roi' in message['payload'].keys():
                            roi_values = [int(x+y, 16)*.25+10 for x,
                                          y in zip(message['payload']['roi'][::2], message['payload']['roi'][1::2])]

                            if message['sender'] == cfg['ZONE_0']['MAC_ADDRESS']:
                                thermal_reading_arr.append(roi_values)
                                thermal_reading_rec_time.append(
                                    message['received_time'])
                            elif message['sender'] == cfg['ZONE_4']['MAC_ADDRESS']:
                                thermal_2_reading_arr.append(roi_values)
                                thermal_2_reading_rec_time.append(
                                    message['received_time'])

                        viz._ax.set_title('REC [new:' + str(len(thermal_reading_arr))+'; old:'+str(len(thermal_2_reading_arr))
                                          + '; F:' +
                                          str(len(Tof_FEET_reading_arr))+'; R:' +
                                          str(len(Tof_RIGHT_reading_arr))+'; L:'
                                          + str(len(Tof_LEFT_reading_arr))+']')
                    sleep(0.75)  # BUG: fast TOF sets memory pointer null
                except KeyError:
                    continue
                except:
                    traceback.print_exc()
            else:
                sleep(0.1)
    t = Thread(target=_updater, daemon=True)
    t.start()
    viz.start()
    viz.show()

    # _updater()


if __name__ == '__main__':
    main()
