import unittest
import sys
import time
import logging
import io
import random
import os
import context
from ruamel.yaml import YAML
from patientzone_otaclient import *



# chdir to here
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# get env variables
yaml = YAML(typ="safe")
with open("test_variables.yaml", "r") as f:
    VARIABLES = yaml.load(f)["COMMON"]

class TestCmdBuilder(unittest.TestCase):
    def test_fromBytes(self):
        # build a valid bytearray
        len_ = 6
        dir_ = CmdBuilder.Dir.TO_CLIENT.value
        cmd = CmdBuilder.Cmd.ACK.value
        data = [7, 7, 7]
        bytes_ = [len_, dir_, cmd] + data
        crc = nosoexCrc(bytes_)
        bytes_.append(crc)
        print(bytes_)

        # test internals
        builder = CmdBuilder.fromBytes(bytes_)
        self.assertEqual(len_, builder._len)
        self.assertEqual(dir_, builder._dir.value)
        self.assertEqual(cmd, builder._cmd.value)
        self.assertEqual(data, builder._data)
        self.assertEqual(crc, builder._crc)

    def test_cmd_crc_len_toBytes(self):
        builder = CmdBuilder()
        builder.cmd(dir_=CmdBuilder.Dir.TO_SERVER, cmd=CmdBuilder.Cmd.ERR, data=[7, 7, 7])
        self.assertTrue(builder.crcValid())
        self.assertTrue(builder.lenValid())

        # inject invalid data
        builder._crc = 123
        builder._len = 2
        self.assertFalse(builder.crcValid())
        self.assertFalse(builder.lenValid())
        bytes_ = builder.toBytes()

        self.assertListEqual([2, 1, 1, 7, 7, 7, 123], bytes_)

    def test_fromBytes_toBytes_same(self):
        bytes_ = [2, 1, 1, 7, 7, 7, 123]
        builder = CmdBuilder.fromBytes(bytes_)
        self.assertListEqual(bytes_, builder.toBytes())


class TestOtaClient(unittest.TestCase):
    def test_get_meta_info(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        cmdAppStr = CmdGetData(CmdBuilder.DataName.APP_STR.value)
        cmdCompileDate = CmdGetData(CmdBuilder.DataName.COMPILE_DATE.value)
        cmdSha256 = CmdGetData(CmdBuilder.DataName.ELF_SHA_256.value)
        client.addCmd(cmdAppStr)
        client.addCmd(cmdCompileDate)
        client.addCmd(cmdSha256)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        print(bytes(cmdAppStr.getData()).decode("utf-8"))
        print(bytes(cmdCompileDate.getData()).decode("utf-8"))
        print(cmdSha256.getData())

    def test_get_highest_log_level(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        cmd = CmdGetData(CmdBuilder.DataName.HIGHEST_LOG_LEVEL.value)
        client.addCmd(cmd)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        val = int.from_bytes(bytes(cmd.getData()), byteorder="little", signed=True)
        print("highest log level: " + str(val))

    def test_get_crash_counter(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        cmdCc = CmdGetData(CmdBuilder.DataName.CRASH_COUNTER.value)
        client.addCmd(cmdCc)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        cc = int.from_bytes(bytes(cmdCc.getData()), byteorder="little", signed=True)
        print("crash counter: " + str(cc))

    def test_set_crash_counter_to_10(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        cc = 10
        ccBytes = cc.to_bytes(INT_LEN, byteorder="little", signed=True)
        cmdCc = CmdSetData(CmdBuilder.DataName.CRASH_COUNTER.value, ccBytes)
        client.addCmd(cmdCc)

        client.connectAndExecuteCmds(VARIABLES["MAC"])

    def test_set_crash_counter_to_0(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        cc = 0
        ccBytes = cc.to_bytes(INT_LEN, byteorder="little", signed=True)
        cmdCc = CmdSetData(CmdBuilder.DataName.CRASH_COUNTER.value, ccBytes)
        client.addCmd(cmdCc)

        client.connectAndExecuteCmds(VARIABLES["MAC"])

    def test_get_wrong_data(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        wrongCmd = CmdGetData("nonexistingdata")
        client.addCmd(wrongCmd)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        print(bytes(wrongCmd.getData()).decode("utf-8", errors='replace'))

    def test_get_log(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)

        stream = io.StringIO()
        cmdLogStream = CmdGetStreamData(CmdBuilder.DataName.LOG.value, stream)
        client.addCmd(cmdLogStream)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        print(stream.getvalue())

    def test_small_fake_ota(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)
        stream = io.BytesIO(200 * b"a" + 200 * b"b" + 200 * b"c")
        cmdSetImage = CmdSetStreamData(CmdBuilder.DataName.IMAGE.value, stream)
        client.addCmd(cmdSetImage)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        print("kByte/s: " + str(cmdSetImage.getKbytePerS()))
        print("chunks success rate: " + str(cmdSetImage.getSuccessRate()))
        self.assertNotEqual(cmdSetImage.getSuccessRate(), 0)

    def test_benchmark_ota(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.INFO)
        bytesLen = 50_000
        stream = io.BytesIO(os.urandom(bytesLen))
        cmdSetImage = CmdSetStreamData(CmdBuilder.DataName.IMAGE.value, stream)
        client.addCmd(cmdSetImage)

        client.connectAndExecuteCmds(VARIABLES["MAC"])
        print("kByte/s: " + str(cmdSetImage.getKbytePerS()))
        print("chunks success rate: " + str(cmdSetImage.getSuccessRate()))
        self.assertNotEqual(cmdSetImage.getSuccessRate(), 0)

    def test_ota(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        self.assertNotEqual(VARIABLES["OTA_FILE"], '', 'need to specify OTA_FILE in test_variables.yaml')
        self.assertTrue(os.path.isfile(VARIABLES["OTA_FILE"]), 'OTA_FILE ' + VARIABLES["OTA_FILE"] + ' does not exist')

        with open (VARIABLES["OTA_FILE"], "rb") as stream:
            client = OtaClient(logLvl=logging.INFO)

            cmdSetImage = CmdSetStreamData(CmdBuilder.DataName.IMAGE.value, stream)
            client.addCmd(cmdSetImage)
            client.connectAndExecuteCmds(VARIABLES["MAC"])

            print("kByte/s: " + str(cmdSetImage.getKbytePerS()))
            print("chunks success rate: " + str(cmdSetImage.getSuccessRate()))
            self.assertNotEqual(cmdSetImage.getSuccessRate(), 0)

    def test_format_internal_nvs(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)
        cmd= CmdCmd(CmdBuilder.CmdName.FORMAT_INTERNAL_NVS.value)
        client.addCmd(cmd)

        client.connectAndExecuteCmds(VARIABLES["MAC"])

    def test_format_fat(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.DEBUG)
        cmd= CmdCmd(CmdBuilder.CmdName.FORMAT_FAT.value)
        client.addCmd(cmd)

        client.connectAndExecuteCmds(VARIABLES["MAC"])

    def test_scanOtaAdv(self):
        client = OtaClient(logLvl=logging.DEBUG)
        devices = client.scanOtaAdv(timeout=15)
        print(str(devices))

    def test_displayDistanceAdv(self):
        client = OtaClient(logLvl=logging.ERROR)
        client.displayDistanceAdv(duration=300)

    def test_displaySingleDistanceAdv(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.ERROR)
        client.displayDistanceAdv(duration=300, mac=VARIABLES["MAC"])

    def test_displayThermalAdv(self):
        client = OtaClient(logLvl=logging.ERROR)
        client.displayThermalAdv(duration=300)

    def test_displaySingleThermalAdv(self):
        self.assertNotEqual(VARIABLES["MAC"], '', 'need to specify MAC in test_variables.yaml')
        client = OtaClient(logLvl=logging.ERROR)
        client.displayThermalAdv(duration=300, mac=VARIABLES["MAC"])


if __name__ == '__main__':
    unittest.main()
