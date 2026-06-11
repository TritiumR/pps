import os
from typing import Optional
from smbus2.smbus2 import SMBus, i2c_msg
from serial import Serial
import threading
from enum import IntEnum
import binascii

from . import log

HEX = binascii.hexlify


class IOBase:
    def reset(self):
        pass

    def read(self, len: int) -> Optional[bytes]:
        pass

    def write(self, data: bytes) -> int:
        return 0

    def get_async(self):
        return False

    is_async = property(get_async)


class I2CIO(IOBase):
    def __init__(self, bus_number=1, addr=0x50):
        self.bus = SMBus(bus_number, force=True)
        self.addr = addr
        self.lock = threading.Lock()

    def read(self, len: int):
        with self.lock:
            retry = 10
            while retry:
                try:
                    msg = i2c_msg.read(self.addr, len)
                    self.bus.i2c_rdwr(msg)
                    return bytes(list(msg))
                except OSError as e:
                    log.error('read error')
                    retry -= 1
                    if retry == 0:
                        raise e


    def write(self, data: bytes):
        with self.lock:
            log.debug(f'I2CTX: {HEX(data)}')
            retry = 10
            while retry:
                try:
                    msg = i2c_msg.write(self.addr, data)
                    self.bus.i2c_rdwr(msg)
                    return len(data)
                except OSError as e:
                    log.error("write error")
                    retry -= 1
                    if retry == 0:
                        raise e

    def __str__(self):
        return f'I2C[{self.bus}]@0x{self.addr:x}'

class SerialIO(IOBase):
    def __init__(self, port="/dev/ttyUSB0", baud=230400):
        self.bus = Serial(port, baud)

    def reset(self):
        self.bus.reset_input_buffer()

    def read(self, len: int) -> Optional[bytes]:
        return self.bus.read(len)

    def write(self, data: bytes) -> int:
        data_len = self.bus.write(data)
        self.bus.flush()
        return data_len


    def __str__(self):
        return f'{self.bus}'

    def get_async(self):
        return True


class MockIO(IOBase):
    def __init__(self):
        self._buffer = bytearray()

    def reset(self):
        pass

    def read(self, len: int) -> Optional[bytes]:
        ret, self._buffer = self._buffer[:len], self._buffer[len:]
        return ret

    def write(self, data: bytes) -> int:
        self._buffer.extend(data)

    def get_async(self):
        return False

    is_async = property(get_async)
