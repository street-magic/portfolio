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
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import joblib

from hcb_visualizer import Visualizer
from datareader.mqtt_reader import MQTT_Subscriber
from datareader.ble_ota.BLE_Reader import BLE_Listener

from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

AVG = [0.4412535,  0.20667809, -0.61849255, -0.45215652, -1.38443582,
       -0.99791856, -0.46623708, -0.24093715,  0.16735466, -0.06987688,
       -0.13519916, -0.40138724, -0.54655695, -1.03330881, -1.14087943,
       -0.49863535, -0.55127682, -0.03893927,  0.15762419,  0.08672506,
       -0.06094022, -0.145343, -0.20349913, -0.01384304, -1.53851646,
       -0.32619091,  0.15765641, -0.03478394,  0.06214772, -0.04530537,
       -0.14474446, -0.38655408, -0.8346859, -0.41594124, -0.09855538,
       0.09021421,  0.21530864, -0.12777397,  0.13131615,  0.34486676,
       -0.16162728, -0.28541327, -0.00158359,  0.40434041,  0.03817224,
       0.0610977, -0.19110897,  0.25794323,  0.91636137,  0.07913269,
       -0.01713727,  0.27365723,  0.09179505, -0.16565811,  0.34350621,
       0.69421214,  1.05304483,  1.4004424,  1.05112481,  0.95599601,
       0.96484039,  0.92457198,  1.02378982,  1.1802693]


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

 # feature extractor


def lim_check(x, y):
    if (x in range(8) and y in range(8)):
        return True
    return False


def ffill(i, j, A, visited, CC, label, T):
    if (visited[i, j] == 1):
        return
    visited[i, j] = 1
    CC[i, j] = label

    dirs = [[-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1]]
    for v in dirs:
        if lim_check(i+v[0], j+v[1]):
            if (abs(A[i, j] - A[i+v[0], j+v[1]]) <= T):
                ffill(i+v[0], j+v[1], A, visited, CC, label, T)


def find_components(A, visited, CC):
    label = 0
    for i in range(8):
        for j in range(8):
            if visited[i, j] == 0:
                label += 1
                ffill(i, j, A, visited, CC, label, 1.7)


def large_comp(frame):
    visited = np.zeros((8, 8))
    CC = np.zeros((8, 8))

    find_components(frame, visited, CC)

    unique, counts = np.unique(CC, return_counts=True)

    comp_list = sorted(list(zip(unique, counts)),
                       key=lambda x: x[1], reverse=True)

    if len(comp_list) <= 1:
        return [0, 0]
    elif len(comp_list) == 2:
        return [comp_list[1][1], 0]

    return [comp_list[1][1], comp_list[2][1]]


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

  #  model = tf.keras.models.load_model('CNN_model_400.h5')
    model = None

    if sys.argv[1] == "CNN":
        model = tf.keras.models.load_model(
            r"CNN_model.h5")
    elif sys.argv[1] == "MLP":
        model = tf.keras.models.load_model('model_MLP.h5')
    elif sys.argv[1] == "RF":
        model = joblib.load("model_RF")

    cfg = _get_config()
    listener = Datareader(cfg)
    viz = Visualizer(cfg)

    def _updater():
        #        for message in messages:
        while True:
            message = listener.get_message()
            if message:
                try:
                    if 'crc' in message['payload'].keys():
                        if message['payload']['crc'] != True:
                            print(
                                '\nChecksum error - packet skipped\n', flush=True)
                            continue
                    #print(message, flush=True)

                    viz.update_zone(message['sender'], message['payload'])

                    if message['sender'] == cfg['ZONE_4']['MAC_ADDRESS'] and message['payload']['type'] == 'THERMAL_ROI':
                        print(">>>")
                        roi_values = [int(x+y, 16)*.25+10 for x,
                                      y in zip(message['payload']['roi'][::2], message['payload']['roi'][1::2])]

                        vals = np.array(roi_values)

                        hist, bin_edges = np.histogram(vals, bins=np.linspace(
                            math.ceil(min(vals)), math.floor(max(vals)), 8))
                        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
                        AVG_bg = bin_centers[np.argmax(hist)]

                        diff = (vals - (AVG_bg + AVG)).reshape(8, 8)

                        diff[diff < 2.5] = 0

                        bg_removed = diff

                        if (np.count_nonzero(bg_removed) <= 1):
                            viz._ax.set_title("no people")
                        else:
                            if sys.argv[1] == "CNN":
                                input_data = bg_removed.reshape(1, 8, 8, 1)
                            elif sys.argv[1] == "RF":
                                num_of_comp = large_comp(
                                    bg_removed[:64].reshape(8, 8))
                                input_data = np.array([[np.std(bg_removed[:64]), np.mean(bg_removed[:64]), np.min(bg_removed[np.nonzero(bg_removed[:64])]), np.max(
                                    bg_removed[:64]), np.count_nonzero(bg_removed[:64])] + num_of_comp])
                            elif sys.argv[1] == "MLP":
                                num_of_comp = large_comp(
                                    bg_removed[:64].reshape(8, 8))
                                input_data = np.array([[np.std(bg_removed[:64]), np.mean(bg_removed[:64]), np.min(bg_removed[np.nonzero(bg_removed[:64])]), np.max(
                                    bg_removed[:64]), np.count_nonzero(bg_removed[:64])] + num_of_comp])

                            if sys.argv[1] == "RF":
                                pred_res = model.predict_proba(input_data)
                              #  viz._ax.set_title(
                               #     "1 Person: "+str(pred_res[0][0])+" \n 2 People:" + str(pred_res[0][1]))
                            else:
                                pred_res = model.predict(input_data)
                            viz._ax.set_title(
                                "1 Person: "+str(round(pred_res[0][0], 2))+"\n2 People:" + str(round(pred_res[0][1], 2)))
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
