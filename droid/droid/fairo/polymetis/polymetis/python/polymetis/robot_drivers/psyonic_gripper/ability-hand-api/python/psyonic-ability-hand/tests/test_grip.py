from unittest import expectedFailure
import pytest
import binascii
import array

from typing import Optional

from psyonic_ability_hand import log
from psyonic_ability_hand.io import IOBase
from psyonic_ability_hand.hand import Hand, Grip, checksum


"""
W: AD 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 53
R: CD B3 1E 41 32 56 1C 41 F5 48 1D 41 64 49 1E 41 7F EE 1E 41 C3 95 15 C1 44 D0 0B A2 40 05 04 C1 14 40 10 0B 76 30 09 42 82 08 7D 91 34 15 D3 03 95 E0 03 AD 20 04 43 20 04 75 20 04 00 00 00 00 00 00 00 00 00 3F 31
           index           | middle          | ring            | pinky           | thumb flexor    | thumb rotator   | Calls/Errors
Position |          9.9189 |          9.7710 |          9.8303 |          9.8929 |          9.9332 |         -9.3491 | 8 / 4
Current  |          0.0000 |          0.0000 |          0.0000 |          0.0000 |          0.0000 |          0.0000 |
sensor 0 |            0068 |            0064 |            0381 |            0173 |            0000 |            0000 |
sensor 1 |            0189 |            0177 |            0841 |            0066 |            0000 |            0000 |
sensor 2 |            0162 |            0118 |            0789 |            0067 |            0000 |            0000 |
sensor 3 |            0084 |            0147 |            0061 |            0066 |            0000 |            0000 |
sensor 4 |            0260 |            0578 |            0149 |            0117 |            0000 |            0000 |
sensor 5 |            0332 |            0136 |            0062 |            0066 |            0000 |            0000 |

"""

class MockComm(IOBase):
    def open(self, *args, **kw):
        log.info("open")
        pass

    def read(self, *args, **kw) -> Optional[bytes]:
        log.info("read")
        pass

    def write(self, data, *args, **kw) -> int:
        log.info(f"write {binascii.hexlify(data)}")
        return len(data)


@pytest.fixture
def hand():
    hand = Hand(MockComm())
    return hand


def test_set_grip_speed(hand):
        hand.set_grip(Grip.Open, -1)
        hand.set_grip(Grip.Open, 101)


def test_grip_open(hand):
    expected = b"\x1d\x00\x0A\xD9"

    def expected_write(data):
        assert expected == data
        return len(data)

    hand._comm.write = expected_write
    hand.set_grip(Grip.Open)


def test_checksum(hand):
    data = [0xAD]
    chksum = checksum(data)

    data = [173, 46, 173, 30, 65, 194, 180, 28, 65, 215, 5, 29, 65, 151, 211, 29, 65, 157, 115, 30, 65, 69, 27, 17, 193, 0]
    chksum = checksum(data)

    log.info(f"checksum: 0x{chksum:x}")


# def test_unpack(hand):
#     indata = [0x98, 0x76, 0x54, 0x32, 0x10, 0xFE, 0xDC, 0xBA, 0x98 ]
#     expected = [0x987, 0x654, 0x321, 0x0FE, 0xDCB, 0xA98]
#     outdata = list( PressureData.unpack(indata) )

#     assert outdata == expected

# def test_control_v1(hand):
#     packet_str = "CD B3 1E 41 32 56 1C 41 F5 48 1D 41 64 49 1E 41 7F EE 1E 41 C3 95 15 C1 44 D0 0B A2 40 05 04 C1 14 40 10 0B 76 30 09 42 82 08 7D 91 34 15 D3 03 95 E0 03 AD 20 04 43 20 04 75 20 04 00 00 00 00 00 00 00 00 00 3F 31"
#     packet = [int(c, 16) for c in packet_str.split() ]
#     log.info(f"packet[{len(packet)}]: {packet}")
#     hand._comm.read = lambda *args, **kw: array.array('B', packet )

#     ret = hand.control_v1(Mode.PosControl, JointData() )

#     log.info(f"ret: {ret}")
