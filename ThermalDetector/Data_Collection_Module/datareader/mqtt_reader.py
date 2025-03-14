import paho.mqtt.client as mqtt
from threading import Lock
import yaml
from random import randint
from typing import Union
import datetime


class Thermal_Event_Page_Combine:
    def __init__(self):
        self.__reset()
        self.roi = ''

    def __reset(self):
        self.offset = 0
        self.session_id = 0
        self.prev_roi = ''

    def combine(self, payload):
        if not payload['crc']:
            self.__reset()
            return

        if payload['session_id'] < self.session_id:
            return
        self.session_id = payload['session_id']

        # Do sessions need to be consecutive for consecutive pages?
        if payload['session_id'] % 16 != (self.session_id + 1) % 16:
            return

        if payload['offset'] < self.offset:
            # new packet
            self.prev_roi = payload['roi']
            self.offset = payload['offset']
            return
        self.offset = payload['offset']
        self.roi = self.prev_roi + payload['roi']

    def get_data(self):
        roi = self.roi
        self.roi = ''
        return roi


class MQTT_Subscriber:
    def __init__(self, topic: list, mqtt_cfg: dict) -> None:
        self._message_queue = []
        self._lock = Lock()
        self.mqtt_cfg = mqtt_cfg
        self._client = mqtt.Client(client_id=self.mqtt_cfg['USERNAME']+'_'+str(randint(1, 1000000)))
        self._client.username_pw_set(username=self.mqtt_cfg['USERNAME'], password=self.mqtt_cfg['PASSWORD'])
        self._client.connect(host=self.mqtt_cfg['ADDRESS'], port=self.mqtt_cfg['PORT'])
        print('MQTT subscription requested...', flush=True)
        self._client.subscribe(topic)
        self._client.on_message = self.__on_message
        self._client.on_subscribe = self.__on_sub

        self.thermal_page_combine = Thermal_Event_Page_Combine()

        self._client.loop_start() # runs in it's own daemon thread

    def __on_sub(self, client, userdata, mid, granted_qos) -> None:
        print('MQTT subscribed to all topics', flush=True)

    def __on_message(self, client, userdata, message) -> None:
        sender_mac = message.topic.split('/')[1] # chip/MAC/idle_patientzone
        decoded_payload = yaml.load(str(message.payload.decode("utf-8")), Loader=yaml.FullLoader)
        current_time = str(datetime.datetime.now())
        if 'thermal/sub_event' in message.topic:
            self.thermal_page_combine.combine(decoded_payload)
            decoded_payload = self.thermal_page_combine.get_data()
            if not len(decoded_payload):
                return
        with self._lock:
            self._message_queue.append({
                'sender': sender_mac,
                'payload': decoded_payload,
                'received_time': current_time
            })

    def get_message(self) -> Union[dict, None]:
        with self._lock:
            if len(self._message_queue):
                return self._message_queue.pop(0)
        return None
