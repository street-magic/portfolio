import struct
import math
import os

# range: 5m, bit: 8, verbose: True
# ./fips run oc_client -- auto -r 5 -b 8 --output [OUTPUT_FILENAME] -v
# time_mod = os.path.getmtime("OUTPUT_FILENAME")

with open("230517_Alan-Vagner-HiWi-SB/SonarDataCollection/00-Data/14.07.23/01-Raw/SONAR-14.07.23-13.36.bin", "rb") as binary_file:
    value = 1689334700.5225632
    dec, integer = math.modf(value)
    action = "read"
    #ab - append, rb - read
    if action == "write":
        b_int = bytearray(struct.pack("I", math.ceil(integer)))
        b_dec = bytearray(struct.pack("f", round(dec,5)))
        binary_file.write(b_int)
        binary_file.write(b_dec)
    elif action == "read":
        binary_file.seek(-8,2)
        print(struct.unpack('I', binary_file.read(4))[0] )
        print(struct.unpack('f', binary_file.read(4))[0] )
