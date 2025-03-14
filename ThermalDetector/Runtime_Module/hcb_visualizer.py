import matplotlib.pyplot as _plt
import matplotlib.animation as _animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _poligon
from matplotlib import cm as _pltcolormap
from matplotlib import colors as _pltcolors
from threading import Lock as _lock
import numpy as np
import matplotlib as mpl
import json


class Visualizer:

    '''
    Contextualized pixel locations on the floor
    Each sensor defines an area/context
    '''
    class Context:

        class ColorMap():
            def __init__(self, cfg_temp):
                self.colormap = _pltcolors.LinearSegmentedColormap.from_list(
                    "", ['#0008e3', '#6aed37', '#e3000f'])
                norm = _pltcolors.Normalize(
                    vmin=cfg_temp['MIN'], vmax=cfg_temp['MAX'], clip=True)
                self.colormapper = _pltcolormap.ScalarMappable(
                    norm=norm, cmap=self.colormap)
                self._colormapper_lock = _lock()

            def get_color(self, value):
                with self._colormapper_lock:
                    return self.colormapper.to_rgba(value)

        class FOV_Cone:
            class FOV_Line:
                def __init__(self, ax, start):
                    self.line_lock = _lock()
                    self.ax = ax
                    self.start = start
                    self.line = self.ax.plot(
                        [], [], [], color='#e3000f')  # list of Xs, Ys, Zs
                    self.line = self.line[0]  # plot returns a list of artist

                def update(self, end_point):
                    with self.line_lock:
                        self.line.set_data_3d([self.start[0], end_point[0]], [
                                              self.start[1], end_point[1]], [self.start[2], end_point[2]])

                def get_line(self):
                    with self.line_lock:
                        return self.line

            def __init__(self, ax, origin):
                self.cone_lock = _lock()
                self.ax = ax
                self.fov_lines = []
                for vertice in range(4):  # 4 corners
                    self.fov_lines.append(self.FOV_Line(self.ax, origin))

            def update(self, new_vertices):
                with self.cone_lock:
                    for fov_line, end_point in zip(self.fov_lines, new_vertices):
                        fov_line.update(end_point)

            def get_cone(self):
                with self.cone_lock:
                    return [x.get_line() for x in self.fov_lines]

        '''
        Each pixel/roi
        '''
        class ROI:
            def __init__(self, coordinates):
                self.alpha = 0.9
                self.coordinates = coordinates
                self.height = 0
                self.tof_facecolor = '#005087'
                self.thermal_color = None
                self.state_modified = False
                self.poligon_tof = None
                self._poligon_lock = _lock()
                self._self_lock = _lock()

            def set_height(self, value):
                if value != self.get_height():
                    self.state_modified = True
                else:
                    self.state_modified = False
                with self._self_lock:
                    self.height = value

            def was_modified(self):
                return self.state_modified

            def get_height(self):
                with self._self_lock:
                    return self.height

            def set_coordinates(self, coordinates):
                with self._poligon_lock:
                    self.coordinates = coordinates

            def get_coordinates(self):
                return self.coordinates

            def set_thermal_color(self, color):
                with self._self_lock:
                    self.thermal_color = color
                    self.state_modified = True

            def get_thermal_color(self):
                with self._self_lock:
                    return self.thermal_color

            def get_vertices(self):
                x1, y1, x2, y2 = * \
                    self.get_coordinates()[1], *self.get_coordinates()[0]
                height = self.get_height()

                dd = 3

                # anti-clockwise from origin. Bottom plane, then top plane
                _points = [
                    [x1, y1, height-dd],
                    [x2, y1, height-dd],
                    [x2, y2, height-dd],
                    [x1, y2, height-dd],
                    [x1, y1, height],
                    [x2, y1, height],
                    [x2, y2, height],
                    [x1, y2, height]
                ]

                # omit bottom plane face
                faces = [
                    [_points[0], _points[1], _points[5], _points[4]],  # front
                    [_points[1], _points[2], _points[6], _points[5]],  # right
                    [_points[2], _points[3], _points[7], _points[6]],  # back
                    [_points[3], _points[0], _points[4], _points[7]],  # left
                    [_points[4], _points[5], _points[6], _points[7]]   # top
                ]

                return faces

            def make_poligon(self):
                with self._poligon_lock:
                    self.poligon_tof = _poligon(
                        self.get_vertices(), linewidth=0)
                    self.poligon_tof.set_facecolor(self.tof_facecolor)

                    # idk what this is. There seems to be some bug in matplotlib
                    self.poligon_tof._facecolors2d = [
                        [], [], []]  # self.tof_facecolor
                    self.poligon_tof._edgecolors2d = [
                        [], [], []]  # self.tof_facecolor

                    self.poligon_tof.set_alpha(.3)

            def get_poligon(self):
                with self._poligon_lock:
                    return self.poligon_tof

            def update_poligon(self):
                with self._poligon_lock:
                    self.poligon_tof.set_verts(self.get_vertices())
                    self.poligon_tof.set_alpha(self.alpha)

            def update_thermal_color(self):
                with self._poligon_lock:
                    self.poligon_tof.set_facecolor(self.get_thermal_color())

        def __init__(self, zone, cfg, in_context):
            self._cfg = cfg
            self.tof_fov = None  # cfg['FOV']['TOF']
            self.thermal_fov = cfg['FOV']['THERMAL']
            self.thermal_res = cfg['RESOLUTION']['THERMAL']
            self.ceiling_height = cfg['CEILING_HEIGHT']
            self.zone = zone
            self.in_context = in_context
            self.lox, self.loy = cfg[zone]['LOC']
            self.colormapper = self.ColorMap(cfg['TEMP'])
            self.prev_accel_vector = [0, 0, -1]
            self._center_x = 0
            self._center_y = 0  # has no use
            self._init_tof_rois = False
            self._init_thermal_rois = False
            self._current_tof_res = None
            self._thermal_init = False
            self.find_inclination(None)
            self.rois_tof = []
            self.rois_thermal = []
            self.fov_cone = self.FOV_Cone(
                self.in_context._ax, [self.lox*100, self.loy*100, self.ceiling_height])
            self._is_recording = False

        def get_thermal_color(self, temperature_reading):
            return self.colormapper.get_color(temperature_reading)

        def __clear_artists(self, artists):
            # TODO BUG make this thread safe
            # Artitst can get cleared while the frame is beung updated
            # This causes some pixels to appear as artifacts
            for artist in artists:
                try:
                    self.in_context._ax.collections.remove(
                        artist.get_poligon())
                except:
                    pass  # TODO DEBUG:

        def __reset_rois_tof(self):
            coordinates_rois_tof = self.get_coverage_rois(
                self.tof_fov, self.tof_res, True)
            self.rois_tof = []
            for coordinate in coordinates_rois_tof:
                roi = self.ROI(coordinate)
                roi.make_poligon()
                self.rois_tof.append(roi)
                roi_poli = roi.get_poligon()
                roi_poli.set_alpha(0)
                self.in_context._ax.add_collection3d(roi_poli)

        def __reset_rois_thermal(self):
            self.rois_thermal = []
            if self.zone == 'ZONE_0':
                coordinates_rois_thermal = self.get_coverage_rois(
                    self.thermal_fov, self.thermal_res, True)
                for coordinate in coordinates_rois_thermal:
                    roi = self.ROI(coordinate)
                    roi.make_poligon()
                    self.rois_thermal.append(roi)
                    tpoli = roi.get_poligon()
                    tpoli.set_alpha(0)
                    self.in_context._ax.add_collection3d(tpoli)

        def get_rois_tof(self):
            return self.rois_tof

        def get_rois_thermal(self):
            if not self._init_thermal_rois:
                self.__reset_rois_thermal()
                self._init_thermal_rois = True
            return self.rois_thermal

        def find_inclination(self, accel_readings):
            if accel_readings is not None:  # TOF
                self.prev_accel_vector = accel_readings
            else:  # Thermal - use prev TOF accel data
                accel_readings = self.prev_accel_vector

            reference_down_vector = [0, 0, -1]  # x, y, z - measured in G's
            accel_vector = accel_readings/np.linalg.norm(accel_readings)
            self._down_inclination = np.degrees(
                np.arccos(np.clip(np.dot(reference_down_vector, accel_vector), -1.0, 1.0)))
            if np.degrees(np.arccos(np.clip(np.dot([1, 0, 0], accel_vector), -1.0, 1.0))) - 90 < 0:
                self._down_inclination = -self._down_inclination
            # can do the same for Y. And X in theory matches Z since we turn in this direction
            reference_ceiling_vector = [0, 1, 0]
            self._ceiling_inclination = np.degrees(np.arccos(np.clip(np.dot(
                reference_ceiling_vector, accel_vector), -1.0, 1.0))) - 90  # should be zero
            
            self.INVALID_INCLINATION = False
            #if (abs(self._down_inclination) > 60) or (abs(self._ceiling_inclination) > 20):
            #    self.INVALID_INCLINATION = True
            #else:
            #    self.INVALID_INCLINATION = False

        def set_tof_res_and_fov(self, roi_values):
            self.tof_res = int(len(roi_values)**.5)
            if self.tof_res == 8:
                self.tof_fov = self._cfg['FOV']['TOF_BIG']
            elif self.tof_res in [2, 3, 4]:
                self.tof_fov = self._cfg['FOV']['TOF_SMALL']
            else:
                raise NotImplementedError('Unknown TOF Resolution')

        def obtain_heights_tof(self, roi_values, roi_pixels):
            if not self._init_tof_rois:  # no artists yet
                self.__reset_rois_tof()
                self._init_tof_rois = True
            elif self._current_tof_res != self.tof_res:  # resolution changed
                self.__clear_artists(self.get_rois_tof())
                self.__reset_rois_tof()
            self._current_tof_res = self.tof_res
            roi_matrix = np.array(roi_values).reshape(
                (self.tof_res, self.tof_res))
            compensated_values = []
            for id_row, matrix_row in enumerate(roi_matrix):
                for id_col, x in enumerate(matrix_row):
                    adjustment = np.cos(np.radians(
                        self._down_inclination))*np.cos(np.radians(self._ceiling_inclination))
                    compensated_values.append(x*adjustment)
            # These are compensated heights as a single point -> inclination is not yet removed
            roi_values = np.array(
                [self.ceiling_height - x for x in compensated_values]).reshape((self.tof_res, self.tof_res))
            # rotate them around their own center as a plane
            # height matrix goes top left to bottom right
            # coordinates go bottom left to top right
            # rotate each column in roi_values one by one -> rotates overall as a plane
            pivot = np.array([self._center_x, 0])
            alpha = np.radians(self._down_inclination)
            rotation_matrix = np.array(
                [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            rotated_values = []
            # "The reference above"
            for i, roi_value_column in enumerate(np.rot90(roi_values, 1)):
                # i-th roi coordinates
                roi_coord_i = roi_pixels[i::self.tof_res]
                # reverse order to match height order
                # all have the same Y coordinate
                roi_coord_i = roi_coord_i[::-1]
                # Xs are Xs, and heights are Ys
                # rotate aroung center -> from get_coverage_rois
                # preserve XY, only set new height corrected for angle
                for j, roicords in enumerate(roi_coord_i):
                    x1, y1, x2, y2 = *roicords[1], *roicords[0]
                    xy = np.array([(x1+x2)/2, roi_value_column[j]])
                    rotated_cords = pivot + \
                        np.matmul(rotation_matrix, xy-pivot)
                    rotated_values.append(rotated_cords[1])  # Don't touch Xs

            roi_values = np.array(rotated_values).reshape(
                (self.tof_res, self.tof_res))
            # rotate back - See the reference above
            roi_values = np.rot90(roi_values, -1)

            # rotate each zone accordingly in reference to coordinate plane
            # Remember that coordinate matrix is not defined in the same way -> updown mirror
            if self.zone in ['ZONE_0', 'ZONE_1']:
                return np.rot90(roi_values, -1).flatten()
            if self.zone == 'ZONE_2':
                return roi_values.flatten()
            if self.zone == 'ZONE_3':
                return np.rot90(roi_values, -2).flatten()

        def get_coverage_rois(self, for_fov, for_resolution, show_coverage_lines=False):
            if self.INVALID_INCLINATION:
                print('INVALID INCLINATION - SENSOR IS NOT POINTING DOWN', flush=True)
                self.__clear_artists(self.get_rois_tof())
                self.__reset_rois_tof()
                self.__clear_artists(self.get_rois_thermal())
                self.__reset_rois_thermal()
                self._init_thermal_rois = False
                self._init_tof_rois = False
                # default height i 0
                return [[(0, 0), (0, 0)] for _ in range(for_resolution*for_resolution)]
            # get distance from sensor/center
            _hip = self.ceiling_height / \
                np.cos(np.radians(abs(self._ceiling_inclination)))
            dis_x = np.sqrt(_hip**2 - self.ceiling_height ** 2)
            dis_x = dis_x if self._ceiling_inclination < 0 else -dis_x
            self._center_x = dis_x
            # Y distance from center
            _hip = self.ceiling_height / \
                np.cos(np.radians(abs(self._down_inclination)))
            dis_y = np.sqrt(_hip**2 - self.ceiling_height ** 2)
            dis_y = -dis_y if self._down_inclination < 0 else dis_y
            self._center_y = dis_y  # Unused

            _hip = self.ceiling_height / np.cos(np.radians(for_fov/2))
            depth = np.sqrt(_hip**2 - self.ceiling_height**2)
            _hip = self.ceiling_height / np.cos(np.radians(for_fov/2))
            width = np.sqrt(_hip**2 - self.ceiling_height**2)

            depth = depth/np.cos(np.radians(self._ceiling_inclination))
            width = width/np.cos(np.radians(self._down_inclination))

            if show_coverage_lines:
                self.__update_coverage_lines(dis_x, dis_y, depth, width)

            coordinate_list = []  # list of tuples
            _start_point = [self.lox*100+dis_x-depth, self.loy*100+dis_y-width]
            for row in range(for_resolution):
                for i in range(for_resolution):
                    x, y = _start_point.copy()
                    coordinate_list.append(
                        [(x, y), (x+2*depth/for_resolution, y+2*width/for_resolution)])
                    _start_point[1] += 2*width/for_resolution
                _start_point[0] += 2*depth/for_resolution
                _start_point[1] -= 2*width
            return coordinate_list

        def __update_coverage_lines(self, x, y, depth, width):
            xyz_list = [
                [x-depth+self.lox*100, y-width+self.loy*100, 0],
                [x-depth+self.lox*100, y+width+self.loy*100, 0],
                [x+depth+self.lox*100, y-width+self.loy*100, 0],
                [x+depth+self.lox*100, y+width+self.loy*100, 0],
            ]
            self.fov_cone.update(xyz_list)

        def get_fov_cone(self):
            return self.fov_cone.get_cone()

    class Payload_decoder:
        """
        Handles HCB roi data
        """

        def __decode_thermal(self, roi_values: str) -> list:
            roi_values = [int(x+y, 16)*.25+10 for x,
                          y in zip(roi_values[::2], roi_values[1::2])]
            return roi_values

        def __decode_tof(self, roi_values: str) -> list:
            roi_values = [int(x+y, 16)*2 for x,
                          y in zip(roi_values[::2], roi_values[1::2])]
            return roi_values

        def __decode_accel(self, hcb_data) -> list:
            return [float(hcb_data['accelerometer_x'])/1000, float(hcb_data['accelerometer_y'])/1000, float(hcb_data['accelerometer_z'])/1000]

        def decode_payload(self, hcb_data: dict) -> dict:
            return_dict = {}
            if 'type' in hcb_data.keys():  # Thermal packet
                if hcb_data['type'] == 'THERMAL_ROI':
                    return_dict['thermal_readings'] = self.__decode_thermal(
                        hcb_data['roi'])
                elif hcb_data['type'] == 'ToF_ROI':
                    return_dict['tof_readings'] = self.__decode_tof(
                        hcb_data['roi'])
            elif 'roi_values' in hcb_data.keys():  # idle packet
                if len(hcb_data['roi_values']):  # 8x8 has empty string
                    return_dict['tof_readings'] = self.__decode_tof(
                        hcb_data['roi_values'])
                return_dict['accel_vector'] = self.__decode_accel(hcb_data)
            elif 'roi' in hcb_data.keys():  # old thermal packet
                return_dict['thermal_readings'] = self.__decode_thermal(
                    hcb_data['roi'])
            if len(return_dict.keys()):
                return return_dict
            raise ValueError(f'Invalid payload: {hcb_data}')

    def __create_canvas(self):
        self._figure = _plt.figure('Raw readings')
        self._ax = self._figure.add_subplot(111, projection='3d', proj_type = 'ortho')
        self._ax.set_aspect('auto')
        self._ax.set_xlim(-self.cfg['VIEW_LIMITS'][0]
                          * 100/2, self.cfg['VIEW_LIMITS'][0]*100/2)
        self._ax.set_xticklabels([])
        self._ax.set_xlabel(f"X-axis {self.cfg['VIEW_LIMITS'][0]} meters")
        # self._ax.set_xlabel("Fuß")
        self._ax.set_ylim(-self.cfg['VIEW_LIMITS'][1]
                          * 100/2, self.cfg['VIEW_LIMITS'][1]*100/2)
        self._ax.set_yticklabels([])
        self._ax.set_ylabel(f"Y-axis {self.cfg['VIEW_LIMITS'][1]} meters")
        # self._ax.set_ylabel("Links")
        self._ax.set_zlim(0, self.cfg['CEILING_HEIGHT'])
        #self._ax_title = self._ax.text(x=0, y=0, z=450, s="", ha="center")
        #self._ax.set_aspect('equal')

    def __init__(self, cfg):
        self.show_ToF = True
        self.is_recording = False
        self.need_to_save = False
        self.frames = None
        self.cfg = cfg
        self._previous_tof_readings = []
        self._previous_themal_readings = []
        self.__create_canvas()
        self.decoder = self.Payload_decoder()
        self.zones = {
            cfg['ZONE_0']['MAC_ADDRESS']: self.Context('ZONE_0', cfg, self),
            #cfg['ZONE_1']['MAC_ADDRESS']: self.Context('ZONE_1', cfg, self),
            #cfg['ZONE_2']['MAC_ADDRESS']: self.Context('ZONE_2', cfg, self),
            #cfg['ZONE_3']['MAC_ADDRESS']: self.Context('ZONE_3', cfg, self)
        }
        self.intrusion_zones = {
            cfg['ZONE_0']['MAC_ADDRESS']: 'ZONE_0',
            #cfg['ZONE_1']['MAC_ADDRESS']: 'ZONE_1',
            #cfg['ZONE_2']['MAC_ADDRESS']: 'ZONE_2',
            #cfg['ZONE_3']['MAC_ADDRESS']: 'ZONE_3'
        }
        self.zone_codes = {
            'ZONE_0': cfg['ZONE_0']['MAC_ADDRESS'],
            'ZONE_1': cfg['ZONE_1']['MAC_ADDRESS'],
            'ZONE_2': cfg['ZONE_2']['MAC_ADDRESS'],
            'ZONE_3': cfg['ZONE_3']['MAC_ADDRESS'],
            'ZONE_4': cfg['ZONE_4']['MAC_ADDRESS']
        }

        # cmap=self.zones[cfg['ZONE_0']['MAC_ADDRESS']].colormapper.colormap
        # cb = self._ax.scatter(range(MIN_TEMP, MAX_TEMP),range(MIN_TEMP, MAX_TEMP),range(MIN_TEMP, MAX_TEMP),c=range(MIN_TEMP, MAX_TEMP), s=0, cmap=cmap)
        # self._figure.colorbar(cb, ax=self._ax)

        x1, y1, x2, y2 = -5, -5, 5, 5
        # anti-clockwise from origin. Bottom plane, then top plane
        _points = [
            [x1+self.cfg['ZONE_0']['LOC'][0]*100, y1+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']],
            [x2+self.cfg['ZONE_0']['LOC'][0]*100, y1+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']],
            [x2+self.cfg['ZONE_0']['LOC'][0]*100, y2+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']],
            [x1+self.cfg['ZONE_0']['LOC'][0]*100, y2+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']],
            [x1+self.cfg['ZONE_0']['LOC'][0]*100, y1+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']-5],
            [x2+self.cfg['ZONE_0']['LOC'][0]*100, y1+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']-5],
            [x2+self.cfg['ZONE_0']['LOC'][0]*100, y2+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']-5],
            [x1+self.cfg['ZONE_0']['LOC'][0]*100, y2+self.cfg['ZONE_0']
                ['LOC'][1]*100, self.cfg['CEILING_HEIGHT']-5]
        ]

        # omit bottom plane face
        hcb_vertices = [
            [_points[0], _points[1], _points[5], _points[4]],  # front
            [_points[1], _points[2], _points[6], _points[5]],  # right
            [_points[2], _points[3], _points[7], _points[6]],  # back
            [_points[3], _points[0], _points[4], _points[7]],  # left
            [_points[4], _points[5], _points[6], _points[7]]   # top
        ]
        artist = _poligon(hcb_vertices, linewidth=0)
        artist.set_facecolor('#e3000f')
        artist.set_alpha(0.9)
        self._ax.add_collection3d(artist)
    def get_if_recording(self):
        return self.is_recording
    def switch_recording(self):
        self.is_recording ^= True
    def __update_accel_vector(self, context, payload) -> None:
        try:
            accel_vector = payload['accel_vector']
        except KeyError:  # not an idle packet
            accel_vector = None
        new_accel_vector = accel_vector != context.prev_accel_vector
        if new_accel_vector:
            context.find_inclination(accel_vector)

    def __update_tof_data(self, context, payload) -> None:
        try:
            tof_readings = payload['tof_readings']
        except KeyError:
            tof_readings = self._previous_tof_readings
        if not len(tof_readings):
            return
        self._previous_tof_readings = tof_readings
        context.set_tof_res_and_fov(tof_readings)
        roi_proyections = context.get_coverage_rois(
            context.tof_fov, context.tof_res)
        corrected_heights = context.obtain_heights_tof(
            tof_readings, roi_proyections)
        context_rois = context.get_rois_tof()
        for roi, new_proyection, new_height in zip(context_rois, roi_proyections, corrected_heights):
            roi.set_coordinates(new_proyection)
            roi.set_height(new_height)

    def __update_thermal_data(self, context, payload) -> None:
        try:
            thermal_readings = payload['thermal_readings']
        except KeyError:
            thermal_readings = self._previous_themal_readings
        if not len(thermal_readings):
            return
        self._previous_themal_readings = thermal_readings
        thermal_readings = np.rot90(np.array(thermal_readings).reshape(
            (context.thermal_res, context.thermal_res)), -1).flatten()
        context_rois = context.get_rois_thermal()
        roi_proyections = context.get_coverage_rois(
            context.thermal_fov, context.thermal_res, show_coverage_lines=False)
        for roi, new_proyection, new_temp in zip(context_rois, roi_proyections, thermal_readings):
            roi.set_coordinates(new_proyection)
            roi.set_thermal_color(context.get_thermal_color(new_temp))

    def update_zone(self, senser_mac, hcb_data) -> None:
        if senser_mac == self.zone_codes['ZONE_0']:
            sender_context = self.zones[senser_mac]
            payload = self.decoder.decode_payload(hcb_data)
            self.__update_accel_vector(sender_context, payload)
            self.__update_tof_data(sender_context, payload)
        elif senser_mac == self.zone_codes['ZONE_4']:
            sender_context = self.zones[self.zone_codes['ZONE_0']]
            payload = self.decoder.decode_payload(hcb_data)
            self.__update_accel_vector(sender_context, payload)
            self.__update_thermal_data(sender_context, payload)
        elif senser_mac in self.zones.keys():
            print(senser_mac+"================wewefwfwfe")
            sender_context = self.zones[senser_mac]
            payload = self.decoder.decode_payload(hcb_data)
            self.__update_accel_vector(sender_context, payload)
            self.__update_tof_data(sender_context, payload)

    def start(self, frames=None) -> None:
        """
        Runs the animation
        """
        self.frames = frames
        self._figure.canvas.mpl_connect('key_press_event', self.__on_press)
        self._ani = _animation.FuncAnimation(
            self._figure, self.__animate, interval=10, frames=frames)
        self._ani.running = True

    def __animate(self, i):
        artists = []
        if self.frames:
            self.update_zone(i[0], i[1])
            self._ax_title.set_text(i[2])
            artists.append(self._ax_title)
        for zone, zone_contex in self.zones.items():
            if self.show_ToF:
                for l in zone_contex.fov_cone.fov_lines:
                    l.line.set_alpha(0.0)
            else:
                for l in zone_contex.fov_cone.fov_lines:
                    l.line.set_alpha(0.0)
            for roi in zone_contex.get_rois_tof():
                if self.show_ToF:
                    roi.alpha = 0.9
                else:
                    roi.alpha = 0.0
                roi.update_poligon()  # update TOF height
                artists.append(roi.get_poligon())
            for roi in zone_contex.get_rois_thermal():
                if roi.was_modified():
                    roi.update_thermal_color()  # update Thermal color
                    roi.update_poligon()
                    artists.append(roi.get_poligon())
            #artists.append(zone_contex.get_fov_cone())
        return artists

    def __on_press(self, event):
        """
        spacebar to pause animation.
        """
        try:
            if event.key.isspace():
                if self._ani.running:
                    print('running -> paused', flush=True)
                    self._ax.set_title('(PAUSED)')
                    self._ani.event_source.stop()
                    self._figure.canvas.resize_event()
                else:
                    print('paused -> running', flush=True)
                    self._ax.set_title('')
                    self._ani.event_source.start()
                self._ani.running ^= True
            if event.key == 't':
                self.show_ToF ^= True
                if not self.show_ToF:
                    self._ax.view_init(azim=-90, elev=90)
                    self._ax.set_zticklabels(self._ax.get_zticklabels(), color='white')
                    self._ax.set_xlim(-self.cfg['VIEW_LIMITS'][0]
                          * 100/4, self.cfg['VIEW_LIMITS'][0]*100/4)
                    self._ax.set_ylim(-self.cfg['VIEW_LIMITS'][1]
                          * 100/4, self.cfg['VIEW_LIMITS'][1]*100/4)
                    self._ax.grid(False)
                else:
                    self._ax.view_init(azim=-60, elev=30)
                    self._ax.set_zticklabels(self._ax.get_zticklabels(), color='black')
                    self._ax.set_xlim(-self.cfg['VIEW_LIMITS'][0]
                          * 100/2, self.cfg['VIEW_LIMITS'][0]*100/2)
                    self._ax.set_ylim(-self.cfg['VIEW_LIMITS'][1]
                          * 100/2, self.cfg['VIEW_LIMITS'][1]*100/2)
                    self._ax.grid(True)

                
        except:  # key is not mapped / key is None -> null event trigger
            pass

    def show(self) -> None:
        """
        This calls matplotlib.pyplot.show()

        Warning: Matplotlib runs on the main thread, meaning further
        code execution is paused untill the window is closed.

        Warning: This will show all other instances of TOF_3D
        """
        _plt.show()

    def save(self):
        '''
        -> Set frames in start
        -> call save instead of show
        '''
        self._ani.save('Animation.gif', writer='imagemagick', fps=5)
