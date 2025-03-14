import numpy as np
import struct
from scapy.all import *
import time
import polarTransform
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt
from skimage import data
from skimage import filters
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

class SonarData:
    def __init__(self, filename, log = False):
        self.filename = filename
        self.f = open(self.filename, 'rb')
        self.data = self.f.read()
        self.pck_len = struct.unpack('I', self.data[198:198+4])[0]
        self.pck_num = len(self.data) // self.pck_len
        self.num_range = struct.unpack('H', self.data[170:170+2])[0]
        self.num_beams = struct.unpack('H', self.data[172:172+2])[0]
        self.img_offset = struct.unpack('H', self.data[190:190+2])[0]
        self.res = struct.unpack('d', self.data[162:170])[0]
        self.range = self.res * self.num_range
        angles = []
        for i in range(self.num_beams):
            angles += [ struct.unpack('h', self.data[202 + 2 * i: 204 + 2 * i])[0] / 100 ]
        self.angles = np.array(angles)
        self.last_time = struct.unpack('d', self.data[(self.pck_num - 1) * self.pck_len + 153: (self.pck_num - 1) * self.pck_len + 161])[0]
        self.time_mod = struct.unpack('f', self.data[len(self.data)-4: len(self.data)])[0] + \
            struct.unpack('I', self.data[len(self.data)-8: len(self.data)-4])[0]
        self.bootup_time = self.time_mod - self.last_time
        
        self.sonar_array = None
        total_time = 0.0
        sonar_times = []
        # Asssuming 4 bytes of gain for every range
        for scan_id in range(self.pck_num):
            start = time.time()
            sonar_times += [ self.bootup_time + struct.unpack('d', self.data[scan_id * self.pck_len + 153: scan_id * self.pck_len + 161])[0] ]
            a = self.pck_len * scan_id + self.img_offset
            img = np.fromstring(self.data[a + 4:a+ 4 + self.num_beams], dtype='uint8')
            for x in range(1, self.num_range):
                begin = a + (self.num_beams + 4) * x + 4
                img = np.vstack((img, np.fromstring(self.data[begin:begin + self.num_beams], dtype='uint8')))
            if self.sonar_array is None:
                self.sonar_array = np.copy(img).reshape(1,self.num_range,self.num_beams)
            else:
                self.sonar_array = np.append(self.sonar_array, img.reshape(1,self.num_range,self.num_beams), axis = 0)
            end = time.time()
            proc_time = round(end - start,2)
            if log:
                print("SonarData :: Parsed ", scan_id + 1, " / ", self.pck_num,' in ', proc_time,' s')
            total_time += proc_time
        self.sonar_times = np.array(sonar_times)
        print("Initialized ", self.filename, " with ", self.pck_num, " packages in ", int(total_time) // 60, "m ", int(total_time) % 60,"s")

class LocationData:
    def __init__(self, filename, nozeroes=True):
        self.scapy_cap = rdpcap(filename)
        self.loc_times = []
        self.x = []
        self.y = []
        self.z = []
        self.rot_x = []
        self.rot_y = []
        self.rot_z = []
        self.obj_x = []
        self.obj_y = []
        self.obj_z = []
        self.obj_rot_x = []
        self.obj_rot_y = []
        self.obj_rot_z = []

        zero_loc = 0
        for i in range(len(self.scapy_cap)):
            if Raw in self.scapy_cap[i] and UDP in self.scapy_cap[i] and self.scapy_cap[i][UDP].dport == 50010 and IP in self.scapy_cap[i] and self.scapy_cap[i][IP].src == '192.168.3.120':
                pk_data = self.scapy_cap[i][Raw].load
                if nozeroes and struct.unpack('f', pk_data[1*4: 1*4 + 4])[0] == 0.0:
                    zero_loc += 1
                    continue
                self.loc_times.append(self.scapy_cap[i].time)
                self.x.append( struct.unpack('f', pk_data[1*4: 1*4 + 4])[0] )
                self.y.append( struct.unpack('f', pk_data[2*4: 2*4 + 4])[0] )
                self.z.append( struct.unpack('f', pk_data[3*4: 3*4 + 4])[0] )
                self.rot_x.append(   struct.unpack('f', pk_data[4*4: 4*4 + 4])[0] )
                self.rot_y.append(   struct.unpack('f', pk_data[5*4: 5*4 + 4])[0] )
                self.rot_z.append(   struct.unpack('f', pk_data[6*4: 6*4 + 4])[0] )
                self.obj_x.append( struct.unpack('f', pk_data[8*4: 8*4 + 4])[0] )
                self.obj_y.append( struct.unpack('f', pk_data[9*4: 9*4 + 4])[0] )
                self.obj_z.append( struct.unpack('f', pk_data[10*4: 10*4 + 4])[0] )
                self.obj_rot_x.append(   struct.unpack('f', pk_data[11*4: 11*4 + 4])[0] )
                self.obj_rot_y.append(   struct.unpack('f', pk_data[12*4: 12*4 + 4])[0] )
                self.obj_rot_z.append(   struct.unpack('f', pk_data[13*4: 13*4 + 4])[0] )
        print("Parsed ",len(self.scapy_cap)-zero_loc,'/',len(self.scapy_cap),' location packets.')

def compute_binarization(sonar_data: SonarData, log = False):
    output = None
    args = (
        sonar_data.angles[-1] - sonar_data.angles[0],  # ArcAngle
        0,   # RotateAngle
        1020,
        1
    )
    footprint = disk(5)
    block_size = 13
    used_percentage = 0.7
    res = -1
    total_time = 0.0
    for i in range(sonar_data.pck_num):
        start = time.time()
        angle = np.deg2rad(sonar_data.angles[-1] - sonar_data.angles[0])
        start_angle = 0.5* (np.pi - angle)
        end_angle = start_angle + angle
        image, ptSettings = polarTransform.convertToCartesianImage(np.rot90(sonar_data.sonar_array[i]),
                                                            finalRadius=1020,
                                                            initialAngle = start_angle,
                                                            finalAngle= end_angle,
                                                            order=4)
        if res == -1:
            res = sonar_data.range / image.shape[0]
        image = image[0:round(image.shape[0] * used_percentage) ][:]
        
        image = rank.mean_bilateral(
            image, footprint=footprint, s0=500, s1=500
        )
        #plt.imshow(image)
        #plt.show()
        std = np.std(image)
        local_thresh = filters.threshold_local(image, block_size, offset=-1.2*std, method = 'median')
        binary_local = image > local_thresh
        
        if output is None:
            output = np.copy(binary_local).reshape(1,binary_local.shape[0], binary_local.shape[1])
        else:
            output = np.append(output, binary_local.reshape(1,binary_local.shape[0], binary_local.shape[1]), axis = 0)
        end = time.time()
        proc_time = round(end - start,2)
        total_time += proc_time
        if log:
            print(output.shape, "Binarized ",i,'/',sonar_data.pck_num,' packages in ',proc_time)
        #plt.imshow(binary_local)
        #plt.show()
    print("Binarized ",sonar_data.pck_num," packages in ", int(total_time) // 60, "m ", int(total_time) % 60,"s")
    return (output, res)

def make_pointcloud(sonar_times, binarization, location, resolution, log = False):
    count = 0
    output = None
    total_time = 0.0
    for i in range(len(binarization)):
        start = time.time()
        pck_time = sonar_times[i]
        min_diff = 10**10
        t_id = -1
        for q in range(len(location.loc_times)):
            rot_time = location.loc_times[q]
            if abs(rot_time - pck_time) < min_diff:
                t_id = q
                min_diff = abs(rot_time - pck_time)

        if min_diff >= 0.5:
            count += 1
            if log:
                print("[!] ", i," :: Can't find rotation data within 0.5 second of sonar recordings")
            continue
        rot_x = location.rot_x[t_id]
        rot_y = location.rot_y[t_id]
        rot_z = location.rot_z[t_id]
        pos_x = location.x[t_id]
        pos_y = location.y[t_id]
        pos_z = location.z[t_id]
        
        img = binarization[i]
        r = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        
        offset = img.shape[1] * resolution / 2.0
        offset_v = np.array([pos_x/1000,pos_y/1000,pos_z/1000])
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y]:
                    v = np.array([x * resolution, y * resolution - offset, 0.0])
                    v = r.apply(v)
                    v += offset_v
                    if output is None:
                        output = np.copy(v)[None,:]
                    else:
                        output = np.append(output, v[None,:], axis = 0)
        end = time.time()
        proc_time = round(end - start,2)
        total_time += proc_time
        if log:
            print(i+1, ' / ', len(binarization), ' [',proc_time,'s]')
        
    print("Created pointcloud with",output.shape[0]," in ", int(total_time) // 60, "m ", int(total_time) % 60,"s")
    return output

def save_pointcloud(output_points, out_filename):
    #os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    out_f = open(out_filename, "w")
    text_list = [
        "ply",
        "format ascii 1.0",
        "element vertex " + str(len(output_points)),
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    for line in text_list:
    # write line to output file
        out_f.write(line)
        out_f.write("\n")
    for v in output_points:
        out_f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
    out_f.close()