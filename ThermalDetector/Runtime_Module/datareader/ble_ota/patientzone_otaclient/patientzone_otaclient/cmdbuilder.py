from enum import Enum


def nosoexCrc(bytes_):
    # KEY
    k=165
    # set crc init value
    crc = k
    for t in bytes_:
        crc ^= t
    return crc


class CmdBuilder:
    class Dir(Enum):
        """self.value == bit in protocol. dont alter"""
        TO_CLIENT = 0x00
        TO_SERVER = 0x01


    class Cmd(Enum):
        """self.value == bit in protocol. dont alter"""
        ACK = 0x00
        ERR = 0x01

        # get/set generic data
        GET_DATA = 0x02
        SET_DATA = 0x03

        DATA_VALUE = 0x04
        START_STREAM = 0x05
        STOP_STREAM = 0x06
        STREAM_DATA = 0x07
        RETRY_STREAM_DATA = 0x08

        KEEP_ALIVE = 0x09

        CMD = 0x0A


    class DataName(Enum):
        APP_STR = "app_str"  # get
        COMPILE_DATE = "compile_date"  # get
        ELF_SHA_256 = "elf_sha_256"  # get
        IMAGE = "image"  # set
        LOG = "log"  # get
        CRASH_COUNTER = "crash_counter"  # get, set
        HIGHEST_LOG_LEVEL = "highest_log_level"


    class CmdName(Enum):
        FORMAT_INTERNAL_NVS = "format_internal_nvs"
        FORMAT_FAT = "format_fat"


    def __init__(self):
        self.reset()

    def reset(self):
        self._len = None
        self._dir = None
        self._cmd = None
        self._data = None
        self._crc = None

    @classmethod
    def fromBytes(cls, bytes_):
        assert(isinstance(bytes_, list))
        assert(len(bytes_) >= 4)  # len, dir, cmd, crc
        assert(all([0 <= x <= 255 for x in bytes_]))
        builder = cls()
        builder._len = bytes_[0]
        builder._dir = CmdBuilder.Dir(bytes_[1])
        builder._cmd = CmdBuilder.Cmd(bytes_[2])
        builder._data = bytes_[3:-1]
        builder._crc = bytes_[-1]
        return builder

    def _isComplete(self):
        return all([x != None for x in [self._len, self._dir, self._len, self._crc]])

    def _toBytes(self):
        ret = []
        if self._len != None:
            ret.append(self._len)
        if self._dir != None:
            ret.append(self._dir.value)
        if self._cmd != None:
            ret.append(self._cmd.value)
        if self._data != None:
            ret += self._data
        if self._crc != None:
            ret.append(self._crc)
        return ret

    def toBytes(self):
        assert(self._isComplete())
        return self._toBytes()

    def cmd(self, dir_, cmd, data=None):
        assert(isinstance(dir_, CmdBuilder.Dir))
        assert(isinstance(cmd, CmdBuilder.Cmd))
        if data != None:
            assert(isinstance(data, list))
            assert(all([0 <= x <= 255 for x in data]))
        self._len = 3
        if data != None:
            self._len += len(data)
        self._dir = dir_
        self._cmd = cmd
        self._data = data
        self._generateCrc()
        return self

    def getDir(self):
        return self._dir

    def getCmd(self):
        return self._cmd

    def getData(self):
        return self._data

    def _generateCrc(self):
        bytes_ = self._toBytes()
        if self._crc:
            bytes_ = bytes_[:-1]  # delete previous crc value
        self._crc = nosoexCrc(bytes_)
        return self

    def crcValid(self) -> bool:
        assert(self._crc != None)
        bytes_ = self._toBytes()[:-1]
        return self._crc == nosoexCrc(bytes_)

    def lenValid(self) -> bool:
        assert(self._isComplete())
        return self._len == len(self._toBytes()) - 1  # don't count len byte itself

    def __str__(self):
        ret = ""
        ret += str([hex(x) for x in self.toBytes()])
        ret += " len: " + str(self._len) 
        ret += " dir: " + str(self._dir)
        ret += " cmd: " + str(self._cmd)
        if self._data:
            ret += " data length: " + str(len(self._data))
        ret += " crc: " + str(self._crc)
        return ret