import asyncio
import array
import dataclasses
import struct
import time
import threading
from enum import Enum, IntEnum
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from uuid import RESERVED_FUTURE
import numpy as np
import binascii
import queue
from .io import IOBase

from psyonic_ability_hand import log

# packet transmission rate for v1/i2c
HAND_V1_TX_SEC = 0.030
# packet transmission rate for v2/serial
HAND_V2_TX_SEC = 0.020

# limits according to psyonic documentation
POSITION_LIMIT_MIN = 0.0
POSITION_LIMIT_MAX = 150.0
VELOCITY_LIMIT_MIN = 0
VELOCITY_LIMIT_MAX = 100.0
TORQUE_LIMIT_MIN = -3.2
TORQUE_LIMIT_MAX = 3.2
PWM_LIMIT_MIN = 0
PWM_LIMIT_MAX = 100.0

# V1/I2C control modes
V1_READ_ONLY = 0x31
V1_POS_UPDATE = 0xAD
V1_VELOCITY_UPDATE = 0xAC
V1_TORQUE_UPDATE = 0xAB

GEAR_RATIO = 649
GEAR_RATIO_TR = 162.45

HEX = binascii.hexlify


class ProtocolError(Exception):
    pass


# V1/I2C interface mode
class Mode(IntEnum):
    Query = 0
    GripControl = 0x1D
    PosControl = 0xAD


# V1/I2C Preprogrammed grips
class Grip(IntEnum):
    Open = 0
    PowerGrasp = 1
    KeyGrasp = 2
    PinchGrasp = 3
    ChuckGrasp = 4
    Point = 9
    Handshake = 17


@dataclass
# Finger specific control data
class JointData:
    Index: float = 0
    Middle: float = 0
    Ring: float = 0
    Pinky: float = 0
    ThumbFlexor: float = 0
    ThumbRotator: float = 0

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_list(self):
        return dataclasses.astuple(self)


@dataclass
# Finger specific motor overheated status
class MotorHotStatus:
    Index: bool = False
    Middle: bool = False
    Ring: bool = False
    Pinky: bool = False
    ThumbFlexor: bool = False
    ThumbRotator: bool = False

    @staticmethod
    def unpack(status: int):
        return MotorHotStatus(
            bool(status & 0x1),
            bool(status & 0x2),
            bool(status & 0x4),
            bool(status & 0x8),
            bool(status & 0x10),
            bool(status & 0x20),
        )


NUM_TOUCH_SITES = 6
NUM_TOUCH_FINGERS = 5
EMPTY_TOUCH = (0.0,) * NUM_TOUCH_SITES


@dataclass
# Finger specific touch sensor data. Some hands may have more touch sensors than others,
# empty sensors will be 0 values. The ThumbRotator does not have sensor data.
class TouchSensors:
    Index: Tuple[float] = EMPTY_TOUCH
    Middle: Tuple[float] = EMPTY_TOUCH
    Ring: Tuple[float] = EMPTY_TOUCH
    Pinky: Tuple[float] = EMPTY_TOUCH
    ThumbFlexor: Tuple[float] = EMPTY_TOUCH

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_list(self):
        return dataclasses.astuple(self)

    @staticmethod
    def unpack(data):
        if len(data) != 45:
            raise ProtocolError(f"unexpected data length: {len(data)}")

        def decode_triplets(data):
            for i in range(0, len(data), 3):
                (a, m, b) = data[i : i + 3]
                yield float((a << 4) | (m >> 4))
                yield float(((m & 0xF) << 8) | b)

        # convert 45-byte packed nibbles to list of 30 (6 sensors, 5 fingers with sensors) flaots
        data = [d for d in decode_triplets(data)]

        # chunk per finger
        return TouchSensors(
            *(
                tuple(data[i : i + NUM_TOUCH_SITES])
                for i in range(0, len(data), NUM_TOUCH_SITES)
            )
        )


# The reply type requested. The defaults is PositionCurrentTouch
class ReplyType(IntEnum):
    PositionCurrentTouch = 0
    PositionVelocityTouch = 1
    PositionCurrentVelocity = 2


def checksum(data, dlen=None):
    cksum = 0
    dlen = dlen or len(data)
    for i in range(0, dlen):
        cksum = cksum + data[i]
    return (-cksum) & 0xFF


class MockComm(IOBase):
    POS_DATA = b"\x01\x02\x03" * 6
    TOUCH_DATA = b"\x00\x10\x02\x00\x30\x04\x00\x50\x06" * 5
    STATUS = 0x3F

    def __init__(self):
        self._cond = threading.Condition()
        self._reply = b""
        self.pos_data = b"\0" * 12

    def read(self, n: int = 1) -> Optional[bytes]:
        with self._cond:
            while len(self._reply) < n:
                self._cond.wait(timeout=1.0)

            resp = self._reply[:n]
            self._reply = self._reply[n:]
            return resp

    def write(self, data: bytes) -> int:
        _, command = data[:2]

        if command in (0x1D,):
            pass
        elif command in (0x10, 0xA0):
            # position + current + touch
            if command == 0x10:
                self.pos_data = data[2 : 2 + 12]

            pos = self.pos_data
            cur = b"\0" * 12

            resp_data = b""

            for i in range(0, 24, 2):
                resp_data += pos[i : i + 2] + cur[i : i + 2]

            resp = (
                command.to_bytes(1, "little")
                + resp_data
                + self.TOUCH_DATA
                + self.STATUS.to_bytes(1, "little")
            )

            chk = checksum(resp)
            resp += chk.to_bytes(1, "little")

            time.sleep(0.002)

            with self._cond:
                self._reply += resp
                self._cond.notify()

        return len(data)

    def __str__(self):
        return f"MockBus"


# Current command mode. The mode switches when a command like set_position or set_velocity is used
class ControlMode(IntEnum):
    Query = 0
    Position = 1
    Velocity = 2
    Torque = 3
    Pwm = 4
    Grasp = 5


@dataclass
class HandStats:
    tx_bytes: int = 0
    rx_bytes: int = 0
    tx_packets: int = 0
    rx_packets: int = 0
    tx_bytes_per_sec: int = 0
    rx_bytes_per_sec: int = 0
    tx_packets_per_sec: float = 0
    rx_packets_per_sec: float = 0
    dt: float = 0.0
    rx_baud: int = 0
    tx_baud: int = 0


class HandThread(threading.Thread):
    def __init__(self, hand: "Hand"):
        super().__init__()
        self.exception = None
        self.hand = hand


class HandV1Thread(HandThread):
    def run(self):
        hand = self.hand

        hand.set_v1_mode(V1_POS_UPDATE)
        while hand._run:
            try:
                hand._command(
                    V1_POS_UPDATE, struct.pack("<6f", *hand._pos_input.to_list())
                )
                hand.read_v1()
                time.sleep(HAND_V1_TX_SEC)
            except Exception as e:
                hand._on_error(f"v1 error: {e}")
                self.exception = e
                hand._run = False


class HandTxThread(HandThread):
    def run(self):
        hand = self.hand
        log.debug("tx thread starting")
        replyType = ReplyType.PositionCurrentTouch

        while hand._run:
            try:
                if hand._control_mode == ControlMode.Position:
                    hand.position_command(replyType, hand._pos_input)
                elif hand._control_mode == ControlMode.Velocity:
                    hand.velocity_command(replyType, hand._velocity_input)
                elif hand._control_mode == ControlMode.Query:
                    hand.query_command(replyType)
                elif hand._control_mode == ControlMode.Grasp:
                    # send velocity commands until actual position is aprox requested
                    log.debug(
                        f"grasp width {hand._grasp_width} speed {hand._grasp_speed}"
                    )
                    # convert 0.0-1.0 'width' into joint positions:
                    pmin = np.array(hand._position_min.to_list())
                    pmax = np.array(hand._position_max.to_list())
                    w = np.full(len(pmin), hand._grasp_width)

                    target = ((pmax - pmin) * w) + pmin
                    # convert speed to joint velocities. if target position is less than current, a negative velocity is used
                    pos = np.array(hand._position.to_list())
                    dpos = target - pos

                    log.debug(f"pos:    {pos}")
                    log.debug(f"target: {target}")
                    log.debug(f"delta:  {dpos}")

                    factor = [
                        VELOCITY_LIMIT_MAX if g else -VELOCITY_LIMIT_MAX
                        for g in np.greater(target, pos)
                    ]
                    s = np.full(len(pmin), hand._grasp_speed)

                    vel = s * factor
                    log.debug(f"target velocities {vel}")

                    close = np.isclose(pos, target, atol=4)
                    log.debug(f"close: {close}")

                    for i, c in enumerate(close):
                        if c:
                            vel[i] = 0

                    if all(close):
                        log.debug("switching to query mode")
                        hand._control_mode = ControlMode.Query
                        continue
                    velocity = JointData(*vel)
                    log.debug(f"velocity update: {velocity}")
                    hand.velocity_command(replyType, velocity)

            except Exception as e:
                hand._on_error(f"TX error: {e}")

            time.sleep(HAND_V2_TX_SEC)

        log.debug("tx thread ending")


class HandRxThread(HandThread):
    def run(self):
        hand = self.hand
        log.debug("rx thread starting")
        while hand._run:
            try:
                hand.read_v2()
            except ProtocolError as e:
                hand._on_error(f"RX error: {e}")
                hand._comm.reset()
            except Exception as e:
                hand._on_error(f"RX error: {e}")
                log.exception("RX error")
                hand._run = False

        log.debug("rx thread ending")


class Hand:
    def __init__(
        self,
        comm: IOBase,
        slave_address=0x50,
        protocol_version=2,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self._comm = comm
        self._slave_address = slave_address
        self._on_error = on_error or (lambda msg: log.exception(msg))
        self._run = True
        self._v1_thread = HandV1Thread(self)
        self._tx_thread = HandTxThread(self)
        self._tx_packets = 0
        self._tx_bytes = 0
        self._tx_time_prev = None
        self._control_mode = ControlMode.Query
        self._command_prev = 0  # for v1/i2c, track the last command sent to know what size packet to read
        self._rx_thread = HandRxThread(self)
        self._rx_packets = 0
        self._rx_bytes = 0
        self._rx_time_prev = None
        self._pos_input = JointData()
        self._torque_input = JointData()
        self._velocity_input = JointData()
        self._pwm_input = JointData()
        self._protocol_version = protocol_version

        self.width = 0
        self.max_width = 100.0
        self.is_grasped = False
        self.is_moving = False
        self._start_time = None
        self._stop_time = None
        self._position = JointData()
        self._current = JointData()
        self._velocity = JointData()
        self._touch = TouchSensors()
        self._motor_status = MotorHotStatus()
        self._position_min = JointData(9, 9, 9, 9, 9, -9)
        self._position_max = JointData(90, 90, 90, 90, 50, -40)
        # self._grip_positions = {
        #     "power", JointData(0, 90, 90, 0, 0, 20)
        # }

        log.debug(
            f"initializing hand over {comm} protocol version: { self._protocol_version } "
        )

    def is_v1(self):
        return self._protocol_version == 1

    def is_v2(self):
        return self._protocol_version == 2

    def get_position(self):
        return self._position

    def set_position(self, pos: JointData) -> None:
        self._pos_input = pos
        log.debug(f"position update: {pos}")
        self._control_mode = ControlMode.Position

    def get_current(self):
        return self._current

    def set_torque(self, torque: JointData) -> None:
        self._torque_input = torque
        log.debug(f"torque update: {torque}")
        self._control_mode = ControlMode.Torque

    def get_velocity(self):
        return self._velocity

    def set_velocity(self, velocity: JointData):
        self._velocity_input = velocity
        log.debug(f"velocity update: {velocity}")
        self._control_mode = ControlMode.Velocity

    def set_pwm(self, pwm: JointData) -> None:
        self._pwm_input = pwm
        log.debug(f"pwm update: {pwm}")
        self._control_mode = ControlMode.Pwm

    def get_touch(self):
        return self._touch

    def get_motor_status(self):
        return self._motor_status

    position = property(get_position)
    current = property(get_current)
    velocity = property(get_velocity)
    touch = property(get_touch)
    motor_status = property(get_motor_status)

    def start(self):
        self._control_mode = ControlMode.Query
        self._tx_packets = 0
        self._rx_packets = 0
        self._comm.reset()

        if self.is_v1():
            self._v1_thread.start()
        else:
            self._tx_thread.start()
            self._rx_thread.start()

        self._start_time = time.time()
        self._stop_time = None

        # give some time for at least one position update
        time.sleep(0.5)

    def stop(self):
        self._run = False

        if self.is_v1():
            self._v1_thread.join()
        else:
            self._rx_packets += 1
            self._tx_thread.join()
            self._rx_thread.join()

        self._stop_time = time.time()

    def stats(self):
        stats = HandStats(
            rx_bytes=self._rx_bytes,
            tx_bytes=self._tx_bytes,
            rx_packets=self._rx_packets,
            tx_packets=self._tx_packets or 0,
        )

        if not self._run or not self._start_time:
            return stats

        stop_time = self._stop_time or time.time()

        stats.dt = stop_time - self._start_time

        stats.rx_bytes_per_sec = int(self._rx_bytes / stats.dt)
        stats.tx_bytes_per_sec = int(self._tx_bytes / stats.dt)
        stats.rx_baud = stats.rx_bytes_per_sec * 10
        stats.tx_baud = stats.tx_bytes_per_sec * 10
        stats.tx_packets_per_sec = (self._tx_packets or 0) / stats.dt
        stats.rx_packets_per_sec = self._rx_packets / stats.dt

        return stats

    def _command(self, command: int, data: bytes = b""):
        if self.is_v2() and self._slave_address is not None:
            buf = struct.pack("BB", self._slave_address, command)
        else:
            buf = command.to_bytes(1, "little")

        if data:
            buf += data

        self._command_prev = command

        buf += struct.pack("B", checksum(buf))
        log.debug(f"TX[{self._tx_packets}]: {HEX(buf)}")

        self._comm.write(buf)

        self._tx_time_prev = time.time()
        self._tx_bytes += len(buf)
        self._tx_packets = (self._tx_packets or 0) + 1

    def upsample_thumb_rotator(self, enable: bool = True):
        self._command(0xC2 if enable else 0xC3)

    def exit_api_mode(self):
        self._command(0x7C)

    def set_grip_v1(self, grip: Grip, speed: float = 0.50):
        """
        Set grip at speed 0-100% (0.0 - 1.0)
        """

        def scale(speed):
            speed = 0 if speed < 0 else 1.0 if speed > 1.0 else speed
            # speed is actually period, higher values go slower
            speed = 1.0 - speed
            return min(1, int(speed * 254))

        log.debug(f"set grip: {grip} speed: {speed}")

        data = struct.pack("BB", grip, scale(speed))
        self._command(0x1D, data)

    def set_v1_mode(self, mode):
        d = struct.pack("25B", mode, *([0] * 24))
        d += checksum(d).to_bytes(1, "little")
        self._comm.write(d)

    def query_command(self, replyType: ReplyType):
        self._command(0xA0 | replyType)

    def position_command(self, replyType: ReplyType, pos: JointData) -> None:
        """
        Joint position in degrees
        """

        def scale(p):
            LIMIT = 150
            p = -LIMIT if p < -LIMIT else LIMIT if p > LIMIT else p
            return int(p * 32767 / LIMIT)

        positions = pos.to_list()
        positions = [scale(p) for p in positions]
        data = struct.pack("<6h", *positions)
        self._command(0x10 | replyType, data)

    def velocity_command(self, replyType: ReplyType, vel: JointData) -> None:
        """
        Joint velocities in degrees/sec
        """

        def scale(v):
            LIMIT = 3000
            v = -LIMIT if v < -LIMIT else LIMIT if v > LIMIT else v
            return int(v * 32767 / LIMIT)

        data = struct.pack("<6h", *[scale(v) for v in vel.to_list()])
        self._command(0x20 | replyType, data)

    def torque_command(self, replyType: ReplyType, torque: JointData) -> None:
        """
        Joint torques in mNM
        """

        def scale(t):
            LIMIT = 3.6
            t = -LIMIT if t < -LIMIT else LIMIT if t > LIMIT else t
            kt = 1.49  # nNM per Amp
            return int(t / kt * 7000 / 0.540)

        data = struct.pack("<6h", *[scale(t) for t in torque.to_list()])
        self._command(0x30 | replyType, data)

    def pwm_command(self, replyType: ReplyType, pwm: JointData) -> None:
        """
        JointData should specify a PWM range or -100% to 100% for each joint
        """

        def scale(t):
            LIMIT = 3546
            t = -LIMIT if t < -LIMIT else LIMIT if t > LIMIT else t
            return int(t / 100 * LIMIT)

        data = struct.pack("<6h", *[scale(t) for t in pwm.to_list()])
        self._command(0x40 | replyType, data)

    def read_v1(self):
        p_len = 71
        p = self._comm.read(p_len)
        if p is None or len(p) != p_len:
            raise ProtocolError("invalid length")

        self._rx_bytes += 1
        self._rx_packets += 1

        log.debug(f"RX (v1): {HEX(p)}")
        data = struct.unpack("<6f47B", p)
        pos = data[:6]
        touch = data[6:-2]
        stat = data[-2]
        cksum = data[-1]
        cksum_calc = checksum(p[:-1])

        if cksum != cksum_calc:
            log.warning(
                f"packet checksum error: expected 0x{cksum_calc} received 0x{cksum}"
            )
            return

        # convert float list to JointData
        self._position = JointData(*pos)
        self._touch = TouchSensors.unpack(touch)

        log.debug(f"RX (pos): {self._position}")

    def read_v2(
        self,
    ) -> None:
        p_type: Optional[bytes] = None
        p_len = 0
        p_sum = 0

        p = self._comm.read(1)

        if p is None:
            log.warning("timed out waiting for RX")
            return

        rx_time = time.time()
        # trigger TX thread as soon as first bytes detected
        self._rx_bytes += 1
        self._rx_packets += 1

        if p is None or len(p) != 1:
            raise ProtocolError("invalid/empty packet type")

        format = p[0]
        variant = format & 0xF

        if variant == 2:
            p_len = 38
        else:
            p_len = 71

        ret = self._comm.read(p_len)
        if ret is None or len(ret) != p_len:
            raise ProtocolError("incomplete packet received")

        p += ret

        dRX = (rx_time - (self._rx_time_prev or rx_time)) * 1000
        dTX = (rx_time - (self._tx_time_prev or rx_time)) * 1000

        log.debug(f"RX[{self._rx_packets-1}] dRX:{dRX:.2f}ms dTX:{dTX:.2f}ms: {HEX(p)}")

        if p is None or len(p) != (p_len + 1):
            raise ProtocolError("invalid length")

        self._rx_bytes += len(p)

        p_sum = p[-1]
        p_sum_calc = checksum(p, len(p) - 1)
        if p_sum != p_sum_calc:
            raise ProtocolError(
                f"checksum failed: received 0x{p_sum:02X} expected: 0x{p_sum_calc:02X} : {HEX(p)}"
            )

        # variant 0: [ (position, motor current), ... ]  | [ touch sensor, ... ] | hot/cold status
        # variant 1: [ (position, rotor velocity), ... ] | [ touch sensor, ... ] | hot/cold status
        # variant 2: [ (position, motor current), ... ] | [ rotor velocity, ... ] | hot/cold status

        self._motor_status = MotorHotStatus.unpack(p[-2])

        # each finger: [position,current,..] or [position,velocity,...]
        d = struct.unpack("<12h", p[1 : 1 + 24])

        decode_position = lambda pos: [p * 150 / 32767 for p in pos]
        decode_current = lambda qv: [d * 0.540 / 7000 for d in qv]
        decode_velocity = lambda qv: [d * 3000 / 32767 for d in qv]

        self._position = JointData(*decode_position(d[::2]))
        qv = d[1::2]

        if variant == 0:
            self._current = JointData(*decode_current(qv))
            self._touch = TouchSensors.unpack(p[1 + 24 : 1 + 24 + 45])
        elif variant == 1:
            self._velocity = JointData(*decode_velocity(qv))
            self._touch = TouchSensors.unpack(p[1 + 24 : 1 + 24 + 45])
        elif variant == 2:
            # each finger: u16 rotor velocity
            self._current = JointData(*decode_current(qv))
            self._velocity = JointData(
                *decode_velocity(struct.unpack("<6h", p[1 + 24 : -2]))
            )
        else:
            raise ProtocolError(f"Unsupported reply variant; {variant}")

        self._rx_time_prev = rx_time

    def grasp(self, *, width: float, speed: float):
        width = 0 if width < 0 else 1 if width > 1 else width
        speed = 0 if speed < 0 else 1 if speed > 1 else speed

        self._grasp_width = width
        self._grasp_speed = speed
        self._control_mode = ControlMode.Grasp
