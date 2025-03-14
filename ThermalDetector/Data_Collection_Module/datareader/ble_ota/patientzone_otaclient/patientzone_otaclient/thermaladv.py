import logging
from dataclasses import dataclass, field
from typing import List

ROI_LSB_TO_C = 0.25
ROI_LSB_OFFSET = 10
COMMON_BYTES_LEN = 3

LOG = logging.getLogger("otaClient")
LOG_THERM_ADV = LOG.getChild("thermalAdv")

def nosoexDecrypt(bytes_):
  key = 161
  offset = 47
  decrypted = b""

  for byte in bytes_:
    b = ((byte - offset) ^ key) & 0xff
    decrypted += b.to_bytes(1, "little")
    offset = byte

  return decrypted

def nosoexCrc(bytes_):
  key=165
  crc = key

  for byte in bytes_:
    crc ^= byte

  return crc

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

@dataclass
class ThermalAdv:
  roisC: List[int] = field(default_factory=list)
  crc: int = 0

  def __post_init__(self):  # __init__ should not be touched cause its generated by dataclass
    self._page0Bytes = None
    self._page1Bytes = None
    self._page2Bytes = None
    self._pkt_complete = list()
    self._complete = False
    self._sessionId = -1

  def __str__(self):
    return f"""
{self.roisC[ 0]:2.2f} {self.roisC[ 1]:2.2f} {self.roisC[ 2]:2.2f} {self.roisC[ 3]:2.2f} {self.roisC[ 4]:2.2f} {self.roisC[ 5]:2.2f} {self.roisC[ 6]:2.2f} {self.roisC[ 7]:2.2f}
{self.roisC[ 8]:2.2f} {self.roisC[ 9]:2.2f} {self.roisC[10]:2.2f} {self.roisC[11]:2.2f} {self.roisC[12]:2.2f} {self.roisC[13]:2.2f} {self.roisC[14]:2.2f} {self.roisC[15]:2.2f}
{self.roisC[16]:2.2f} {self.roisC[17]:2.2f} {self.roisC[18]:2.2f} {self.roisC[19]:2.2f} {self.roisC[20]:2.2f} {self.roisC[21]:2.2f} {self.roisC[22]:2.2f} {self.roisC[23]:2.2f}
{self.roisC[24]:2.2f} {self.roisC[25]:2.2f} {self.roisC[26]:2.2f} {self.roisC[27]:2.2f} {self.roisC[28]:2.2f} {self.roisC[29]:2.2f} {self.roisC[30]:2.2f} {self.roisC[31]:2.2f}
{self.roisC[32]:2.2f} {self.roisC[33]:2.2f} {self.roisC[34]:2.2f} {self.roisC[35]:2.2f} {self.roisC[36]:2.2f} {self.roisC[37]:2.2f} {self.roisC[38]:2.2f} {self.roisC[39]:2.2f}
{self.roisC[40]:2.2f} {self.roisC[41]:2.2f} {self.roisC[42]:2.2f} {self.roisC[43]:2.2f} {self.roisC[44]:2.2f} {self.roisC[45]:2.2f} {self.roisC[46]:2.2f} {self.roisC[47]:2.2f}
{self.roisC[48]:2.2f} {self.roisC[49]:2.2f} {self.roisC[50]:2.2f} {self.roisC[51]:2.2f} {self.roisC[52]:2.2f} {self.roisC[53]:2.2f} {self.roisC[54]:2.2f} {self.roisC[55]:2.2f}
{self.roisC[56]:2.2f} {self.roisC[57]:2.2f} {self.roisC[58]:2.2f} {self.roisC[59]:2.2f} {self.roisC[60]:2.2f} {self.roisC[61]:2.2f} {self.roisC[62]:2.2f} {self.roisC[63]:2.2f}
"""

  def isComplete(self):
    return self._complete

  def getSignedNumber(self, number, bitLength):
    mask = (2 ** bitLength) - 1
    if number & (1 << (bitLength - 1)):
      return number | ~mask
    else:
      return number & mask

  def feedWithAdvBytes(self, bytes_ : bytes):
    """takes manufacture bytes from ble patientzone adv in right order and then builds an object out of it"""

    if self.isComplete() or not bytes_:
      return;

    # packetId == patientzone?
    packetId = int(bytes_[0])
    if packetId != 0x25:
      LOG_THERM_ADV.debug("packetId wrong. expected: 0x25. actual: " + hex(packetId))
      return None

    # decrypt
    decrypted = bytes_[:2] + nosoexDecrypt(bytes_[2:])  # 2 first bytes are not encrypted
    LOG_THERM_ADV.debug("decrypted: " + str([hex(x) for x in decrypted]))

    # get page number
    pageNumber = (decrypted[2] & 0b0110_0000) >> 5
    if pageNumber in [0, 1, 2]:
      LOG_THERM_ADV.info("found page number: " + str(pageNumber))
    else:
      LOG_THERM_ADV.warning("page number wrong. expected: [0, 1, 2], actual: " + str(pageNumber))

    if pageNumber == 0 and self._page0Bytes == None:
        self._page0Bytes = decrypted
        LOG_THERM_ADV.info("adding page number: " + str(pageNumber))
    elif pageNumber == 1 and self._page1Bytes == None and self._page0Bytes != None:
        self._page1Bytes = decrypted
        LOG_THERM_ADV.info("adding page number: " + str(pageNumber))
    elif pageNumber == 2 and self._page2Bytes == None and self._page0Bytes != None and self._page1Bytes != None:
        self._page2Bytes = decrypted
        LOG_THERM_ADV.info("adding page number: " + str(pageNumber))

    if self._page0Bytes != None and self._page1Bytes != None and self._page2Bytes != None:
      LOG_THERM_ADV.info("found all pages. initialize thermal data")

      # crc ok? not? -> discard everything
      crc = nosoexCrc(self._page0Bytes[COMMON_BYTES_LEN:] + self._page1Bytes[COMMON_BYTES_LEN:] + self._page2Bytes[COMMON_BYTES_LEN:])
      if self.crc != crc:
        LOG_THERM_ADV.warning("crc invalid. expected: " + hex(crc) + ". actual: " + hex(self.crc) + ". discarding all pages")
        self._page0Bytes = None
        self._page1Bytes = None
        self._page2Bytes = None
        return
      
      flags = self._page0Bytes[COMMON_BYTES_LEN]
      pktId = flags & 0x3
      sessionID = (flags & 0xf0) >> 4
      
      LOG_THERM_ADV.info("pktId: " + str(pktId) + " sessionId: " + str(sessionID))
      
      if (pktId == 1 and 0 not in self._pkt_complete):
        LOG_THERM_ADV.info("received pkt 1, but not pkt 0")
        return
      
      if (self._sessionId != sessionID):
        if (self._sessionId == -1) or (pktId == 0):
          LOG_THERM_ADV.info("session mismatch (set new session): " + str(self._sessionId) + " <-> " + str(sessionID))
          # start a new try to collect a sessionID
          self._sessionId = sessionID
          self._pkt_complete.clear()
        else:
          LOG_THERM_ADV.info("session mismatch: " + str(self._sessionId) + " <-> " + str(sessionID))
          return
      
      self._pkt_complete.append(pktId)
      
      # get stuff from page 0
      for i in range(14):
        self.roisC.append(ROI_LSB_OFFSET + ROI_LSB_TO_C * self._page0Bytes[COMMON_BYTES_LEN + 2 + i])

      # get stuff from page 1
      for i in range(16):
        self.roisC.append(ROI_LSB_OFFSET + ROI_LSB_TO_C * self._page1Bytes[COMMON_BYTES_LEN + i])

      # get stuff from page 2
      for i in range(2):
        self.roisC.append(ROI_LSB_OFFSET + ROI_LSB_TO_C * self._page2Bytes[COMMON_BYTES_LEN + i])
      
      # reset pages to receive next pkt
      self._page0Bytes = None
      self._page1Bytes = None
      self._page2Bytes = None

      if (0 in self._pkt_complete and 1 in self._pkt_complete):
        self._complete = True
