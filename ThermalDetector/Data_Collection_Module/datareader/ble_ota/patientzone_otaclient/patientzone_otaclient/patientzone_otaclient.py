import asyncio
import bleak
import logging
import sys
import io
import time
import abc
from .cmdbuilder import CmdBuilder
from .distanceadv import DistanceAdv
from .thermaladv import ThermalAdv


SERVICE_UUID = "00001234-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "00005678-0000-1000-8000-00805f9b34fb"
MANUFACTUR_DATA = 1836  # ble gwa id
MAX_CHUNK_LEN = 256 - 4 # len, dir, cmd, crc
INT_LEN = 4  # client and server side need to have same when sending int values via bytes


LOG = logging.getLogger("otaClient")
LOG_SERVER = LOG.getChild("server")
LOG.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)


notificationEvent = asyncio.Event()
notificationEvent.clear()


class IOtaClientCmd(abc.ABC):
    @abc.abstractmethod
    async def execute(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()

    def setClient(self, client):
        assert(isinstance(client, OtaClient)), type(client)
        self._client = client

    
class OtaClient:
    def __init__(self, logLvl=logging.INFO):
        LOG.setLevel(logLvl)
        # cmds
        self._cmds = []

        #
        self._client = None
        self._lastNotifyBuilder = None

        # stats stream
        self._lastChunksSuccessRate = None
        self._lastKbytePerS = None

        # scan 
        self._distanceAdv = DistanceAdv()
        self._thermalAdv = ThermalAdv()

    # user ################################
    def addCmd(self, cmd : IOtaClientCmd):
        assert(isinstance(cmd, IOtaClientCmd)), type(cmd)
        cmd.setClient(self)
        self._cmds.append(cmd)

    def connectAndExecuteCmds(self, mac, retry=True):
        assert(isinstance(mac, str)), type(mac)
        assert(isinstance(retry, bool))

        async def run():
            while(True):
                try:
                    LOG.info("connecting to: " + mac)
                    async with bleak.BleakClient(mac) as client:
                        LOG.info("connected")
                        self._client = client
                        await self._client.start_notify(CHAR_UUID, self._onNotify)
                        LOG.debug("subscribed for notifications")
                        for cmd in self._cmds:
                            LOG.info("executing command: " + str(cmd))
                            await cmd.execute()
                        break
                except Exception as e:
                    if retry:
                        LOG.info("retry ...")
                    else:
                        raise e

        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())

    def scanOtaAdv(self, timeout):
        """returns dictionary {mac : rssi}"""
        ret = {}
        async def run():
            devices = await bleak.discover(timeout=timeout)
            for d in devices:
                if self._isOtaAdv(d):
                    ret.update({d.address : d.rssi})

        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())

        return ret

    def displayDistanceAdv(self, duration=20, mac=None):
        def simple_callback(device, advertisement_data):
            if MANUFACTUR_DATA in advertisement_data.manufacturer_data:
                if not mac or mac.lower() == device.address.lower():
                    print(device.address, "RSSI:", device.rssi)
                    manufactureBytes = advertisement_data.manufacturer_data[MANUFACTUR_DATA]
                    if not self._distanceAdv.isComplete():
                        self._distanceAdv.feedWithAdvBytes(manufactureBytes)

                    if self._distanceAdv.isComplete():
                        print(self._distanceAdv)
                        self._distanceAdv = None
                        self._distanceAdv = DistanceAdv()

        async def run():
            scanner = bleak.BleakScanner()
            scanner.register_detection_callback(simple_callback)

            await scanner.start()
            await asyncio.sleep(duration)
            await scanner.stop()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())

    def displayThermalAdv(self, duration=20, mac=None):
        def simple_callback(device, advertisement_data):
            if MANUFACTUR_DATA in advertisement_data.manufacturer_data:
                if not mac or mac.lower() == device.address.lower():
                    print(device.address, "RSSI:", device.rssi)
                    manufactureBytes = advertisement_data.manufacturer_data[MANUFACTUR_DATA]
                    if not self._thermalAdv.isComplete():
                        self._thermalAdv.feedWithAdvBytes(manufactureBytes)

                    if self._thermalAdv.isComplete():
                        print(self._thermalAdv)
                        self._thermalAdv = None
                        self._thermalAdv = ThermalAdv()

        async def run():
            scanner = bleak.BleakScanner()
            scanner.register_detection_callback(simple_callback)

            await scanner.start()
            await asyncio.sleep(duration)
            await scanner.stop()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())

    # internals ################################
    def _isOtaAdv(self, d) -> bool:
        assert(isinstance(d, bleak.backends.device.BLEDevice))
        if not list(d.metadata["manufacturer_data"].keys()):
            return False
        if list(d.metadata["manufacturer_data"].keys())[0] == MANUFACTUR_DATA \
        and SERVICE_UUID in d.metadata["uuids"]:
            return True
        else:
            return False

    def _isDistanceAdv(self, d) -> bool:
        pass

    def _isThermalAdv(self, t) -> bool:
        pass

    # stats ota
    def getLastChunksSuccessRate(self):
        return self._lastChunksSuccessRate

    def getLastKbytePerS(self):
        return self._lastKbytePerS

    def _onNotify(self, sender, data):
        self._lastNotifyBuilder = CmdBuilder.fromBytes(list(data))
        notificationEvent.set()

    async def _waitForNotifiedCmd(self, cmdList):
        if(isinstance(cmdList, list)):
            assert(all(isinstance(i, CmdBuilder.Cmd) for i in cmdList))
        else:
            assert(isinstance(cmdList, CmdBuilder.Cmd))
            cmdList = [cmdList]

        notificationEvent.clear()
        self._lastNotifyBuilder = None
        while(True):
            LOG.debug("waiting for notification: " + str(cmdList))
            await notificationEvent.wait()
            notificationEvent.clear()
            LOG.debug("recieved: " + str(self._lastNotifyBuilder))
            if not self._lastNotifyBuilder.lenValid:
                LOG.warning("length byte invalid")
                return False
            if not self._lastNotifyBuilder.crcValid:
                LOG.warning("crc invalid")
                return False
            if not self._lastNotifyBuilder.getCmd() in cmdList:
                LOG.warning("commands expected: " + str(cmdList) + "actual: " + str(self._lastNotifyBuilder.getCmd()))
                return False
            if not self._lastNotifyBuilder.getDir() == CmdBuilder.Dir.TO_CLIENT:
                LOG.warning("wrong direction")
                return False
            return self._lastNotifyBuilder.getCmd()

    async def _sendCmd(self, cmdBuilder : CmdBuilder):
        assert(isinstance(cmdBuilder, CmdBuilder))
        assert(cmdBuilder.lenValid() and cmdBuilder.crcValid())
        LOG.debug("sending: " + str(cmdBuilder))
        payload = cmdBuilder.toBytes()
        await self._client.write_gatt_char(CHAR_UUID, bytearray(payload))
        LOG.debug("send done")

    async def _cmd(self, cmdName):
        builder = CmdBuilder()
        CMD = CmdBuilder.Cmd

        builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.CMD, data=list(cmdName.encode()))
        await self._sendCmd(builder)

        cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR])
        if cmd == CMD.ACK:
            return
        elif cmd == CMD.ERR:
            data = self._lastNotifyBuilder.getData()
            LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))

    async def _getData(self, dataName, out=None):
        if out != None:
            assert(isinstance(out, (io.BufferedIOBase, io.TextIOBase)))

        builder = CmdBuilder()
        CMD = CmdBuilder.Cmd

        # get data
        builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.GET_DATA, data=list(dataName.encode()))
        await self._sendCmd(builder)

        # small value or stream?
        cmd = await self._waitForNotifiedCmd([CMD.DATA_VALUE, CMD.START_STREAM, CMD.ERR])

        # err
        if cmd == CMD.ERR:
            data = self._lastNotifyBuilder.getData()
            LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))
            return data
        # small value
        elif cmd == CMD.DATA_VALUE:
            data = self._lastNotifyBuilder.getData()
            # ack recieved value
            builder.reset()
            builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.ACK)
            await self._sendCmd(builder)
            if out == None:
                return data
            else:
                if isinstance(out, io.BufferedIOBase):
                    out.write(bytes(data))
                elif isinstance(out, io.TextIOBase):
                    out.write(bytes(data).decode("utf-8"))
                else:
                    raise NotImplementedError
                return

        # stream
        elif(cmd == CMD.START_STREAM):
            builder.reset()
            builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.ACK)
            await self._sendCmd(builder)

            data = []
            streamStop = False
            while not streamStop:
                cmd = await self._waitForNotifiedCmd([CMD.STREAM_DATA, CMD.STOP_STREAM, CMD.ERR])
                if cmd == CMD.ERR:
                    data = self._lastNotifyBuilder.getData()
                    LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))
                    streamStop = True
                    break
                elif cmd == CMD.STREAM_DATA:
                    builder.reset()
                    builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.ACK)
                    await self._sendCmd(builder)
                    if out == None:
                        data += self._lastNotifyBuilder.getData()
                    else:
                        if isinstance(out, io.BufferedIOBase):
                            out.write(bytes(self._lastNotifyBuilder.getData()))
                        elif isinstance(out, io.TextIOBase):
                            out.write(bytes(self._lastNotifyBuilder.getData()).decode("utf-8"))
                        else:
                            raise NotImplementedError
                elif cmd == CMD.STOP_STREAM:
                    builder.reset()
                    builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.ACK)
                    await self._sendCmd(builder)
                    streamStop = True

            if out == None:
                return data
            else:
                return

    async def _setData(self, dataName, data, isStream=False):
        CMD = CmdBuilder.Cmd

        if isStream:
            assert(isinstance(data, io.BufferedIOBase))
            acks = 0
            errs = 0
            self._lastChunksSuccessRate = 0
            self._lastKbytePerS = 0
            builder = CmdBuilder()

            builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.SET_DATA, data=list(dataName.encode()))
            await self._sendCmd(builder)
            cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR])
            if cmd == CMD.ACK:
                pass
            elif cmd == CMD.ERR:
                data = self._lastNotifyBuilder.getData()
                LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))
                return

            # send start stream
            builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.START_STREAM)
            await self._sendCmd(builder)
            cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR])
            if cmd == CMD.ACK:
                pass
            elif cmd == CMD.ERR:
                data = self._lastNotifyBuilder.getData()
                LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))
                return

            # data chunks
            chunk = None
            retryChunk = False
            while(True):
                # send
                start = time.time()
                builder.reset()
                if not retryChunk:
                    # build new chunk
                    chunk = data.read(MAX_CHUNK_LEN)
                    # EOF -> stop sending data. handshake yadayada
                    if len(chunk) == 0:
                        builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.STOP_STREAM)
                        await self._sendCmd(builder)
                        cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR])
                        if cmd == CMD.ACK:
                            LOG.info(" #### SUCCESS #### ")
                        if cmd == CMD.ERR:
                            data = self._lastNotifyBuilder.getData()
                            LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))
                            LOG.info(" #### ERROR #### ")

                        self._lastChunksSuccessRate = acks / (acks + errs)
                        return
                        
                # send chunk
                builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.STREAM_DATA, data=list(chunk))
                await self._sendCmd(builder)
                cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR, CMD.RETRY_STREAM_DATA])

                currentKbs = (len(chunk) / 1000) / (time.time() - start)
                self._lastKbytePerS = (currentKbs + self._lastKbytePerS) / 2  # moving average window length 2

                # response to that chunk
                if cmd == CMD.ACK:
                    acks += 1
                    if acks % 4 == 0:
                        LOG.info(f"progress: {((acks * MAX_CHUNK_LEN) / 1000):.2f} kbyte")
                    retryChunk = False
                    continue
                elif cmd == CMD.RETRY_STREAM_DATA:
                    errs += 1
                    retryChunk = True
                    continue
                elif cmd == CMD.ERR:
                    data = self._lastNotifyBuilder.getData()
                    LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))
                    break

        elif not isStream:
            assert isinstance(data, (bytes)), type(data)

            builder = CmdBuilder()

            builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.SET_DATA, data=list(dataName.encode()))
            await self._sendCmd(builder)
            cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR])
            if cmd == CMD.ACK:
                pass
            elif cmd == CMD.ERR:
                data = self._lastNotifyBuilder.getData()
                LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))

            builder.reset()
            builder.cmd(CmdBuilder.Dir.TO_SERVER, CMD.DATA_VALUE, data=list(data))
            await self._sendCmd(builder)

            cmd = await self._waitForNotifiedCmd([CMD.ACK, CMD.ERR])
            if cmd == CMD.ACK:
                return
            elif cmd == CMD.ERR:
                data = self._lastNotifyBuilder.getData()
                LOG_SERVER.warning(bytes(data).decode("utf-8", errors='replace'))


# commands ##########################
class CmdCmd(IOtaClientCmd):
    def __init__(self, cmdName : str):
        assert(isinstance(cmdName, str)), type(cmdName)
        self._cmdName = cmdName

    async def execute(self):
        await self._client._cmd(self._cmdName)

    def __str__(self):
        return "cmd: " + self._cmdName


class CmdGetData(IOtaClientCmd):
    def __init__(self, dataName : str):
        assert(isinstance(dataName, str)), type(dataName)
        self._dataName = dataName
        self._data = None

    async def execute(self):
        self._data = await self._client._getData(self._dataName)

    def getData(self):
        return self._data

    def __str__(self):
        return "get data: " + self._dataName

class CmdSetData(IOtaClientCmd):
    def __init__(self, dataName : str, data : bytes):
        assert(isinstance(dataName, str)), type(dataName)
        self._dataName = dataName
        self._data = data

    async def execute(self):
        await self._client._setData(self._dataName, self._data, isStream=False)

    def __str__(self):
        return "set data: " + self._dataName


class CmdGetStreamData(IOtaClientCmd):
    def __init__(self, dataName : str, stream):
        assert(isinstance(dataName, str)), type(dataName)
        assert(isinstance(stream, (io.BufferedIOBase, io.TextIOBase)))
        self._dataName = dataName
        self._stream = stream

    async def execute(self):
        await self._client._getData(self._dataName, out=self._stream)

    def __str__(self):
        return "get stream data: " + self._dataName


class CmdSetStreamData(IOtaClientCmd):
    def __init__(self, dataName : str, stream):
        assert(isinstance(dataName, str)), type(dataName)
        assert(isinstance(stream, (io.BufferedIOBase, io.TextIOBase)))
        self._dataName = dataName
        self._stream = stream

        # stats
        self._successRate = None
        self._kbytePerS = None

    async def execute(self):
        await self._client._setData(self._dataName, data=self._stream, isStream=True)
        self._successRate = self._client.getLastChunksSuccessRate()
        self._kbytePerS = self._client.getLastKbytePerS()

    def getSuccessRate(self):
        return self._successRate

    def getKbytePerS(self):
        return self._kbytePerS

    def __str__(self):
        return "set stream data: " + self._dataName
