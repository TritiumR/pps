import time
import dataclasses
import sys, termios, tty, select
import threading, queue
from enum import IntEnum
from typing import Optional, Tuple

from serial import Serial

from rich.logging import RichHandler
from rich.style import Style
from rich.console import Console, Group
from rich.columns import Columns
from rich.live import Live
from rich.bar import Bar
from rich.text import Text
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.layout import Layout
from decimal import *

from psyonic_ability_hand import log
from psyonic_ability_hand.hand import (
    Hand,
    JointData,
    Grip,
    MockComm,
)

from psyonic_ability_hand.io import I2CIO, SerialIO

console = Console()

LOG_FILE_NAME = "hand-demo.log"
LOG_LINES = 30

class AppMode(IntEnum):
    Position = 0
    Velocity = 1
    Torque = 2
    Pwm = 3


class App:
    key_input = queue.Queue()
    run: bool = True
    position_input = JointData()
    position_input_init = False
    velocity_input = JointData(0, 0, 0, 0, 0, 0)
    torque_input = JointData()
    pwm_input = JointData()
    initialized: bool = False
    mode: int = 0
    hand: Optional[Hand] = None


PINKY_KEYS = ("A", "a", "z", "Z")
RING_KEYS = ("S", "s", "x", "X")
MIDDLE_KEYS = ("D", "d", "c", "C")
INDEX_KEYS = ("F", "f", "v", "V")
THUMB_FLEXOR_KEYS = ("G", "g", "b", "B")
THUMB_ROTATOR_KEYS = ("H", "h", "n", "N")
KEY_VALUE_MAP = (-10, -1, 1, 10)


def handle_key(app: App, key: str):
    if key == "q":
        raise KeyboardInterrupt

    def update_joint_data(joint, key):
        if key in PINKY_KEYS:
            joint.Pinky += KEY_VALUE_MAP[PINKY_KEYS.index(key)]
            return True
        if key in RING_KEYS:
            joint.Ring += KEY_VALUE_MAP[RING_KEYS.index(key)]
            return True
        if key in MIDDLE_KEYS:
            joint.Middle += KEY_VALUE_MAP[MIDDLE_KEYS.index(key)]
            return True
        if key in INDEX_KEYS:
            joint.Index += KEY_VALUE_MAP[INDEX_KEYS.index(key)]
            return True
        if key in THUMB_FLEXOR_KEYS:
            joint.ThumbFlexor += KEY_VALUE_MAP[THUMB_FLEXOR_KEYS.index(key)]
            return True
        if key in THUMB_ROTATOR_KEYS:
            joint.ThumbRotator += KEY_VALUE_MAP[THUMB_ROTATOR_KEYS.index(key)]
            return True

    if key == "m":
        app.mode = (app.mode + 1) % 4
    elif app.hand is not None:
        try:
            if app.mode == AppMode.Position:
                if update_joint_data(app.position_input, key):
                    app.hand.set_position(app.position_input)
            elif app.mode == AppMode.Velocity:
                if update_joint_data(app.velocity_input, key):
                    app.hand.set_velocity(app.velocity_input)
            elif app.mode == AppMode.Torque:
                if update_joint_data(app.torque_input, key):
                    app.hand.set_torque(app.torque_input)
            elif app.mode == AppMode.Pwm:
                if update_joint_data(app.pwm_input, key):
                    app.hand.set_torque(app.pwm_input)
        except Exception as e:
            console.print(e)
            raise e



def get_input(app:App):
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        fds = ([sys.stdin], [], [])
        ch = None
        while app.run:
            if select.select(*fds, 0.010) == fds:
                app.key_input.put_nowait( sys.stdin.read(1) )
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        app.run = False


def main(comm_type):
    app = App()
    msgs = []

    layout = Layout()

    info="""Keys:
    Cycle Input Mode: 'm'

    Joint Control:
    --------------------------------------------------------------
    |     | Pinky | Ring | Middle | Index | Thumb Fl | Thumb Rot |
    |-----|-------|------|--------|-------|----------|-----------|
    | +10 |   A   |  S   |   D    |   F   |    G     |     H     |
    | +1  |   a   |  s   |   d    |   f   |    g     |     h     |
    | -1  |   z   |  x   |   c    |   v   |    b     |     n     |
    | -10 |   Z   |  X   |   C    |   V   |    B     |     N     |
    --------------------------------------------------------------
"""

    layout.split(
        Layout(info, name="info", size=13),
        Layout(" ", name="stats", size=1),
        Layout(" ", name="table", size=10),
        Layout(" ", name="log", size=LOG_LINES)
    )

    def update_log(msg):
        nonlocal msgs
        msgs = msgs[-LOG_LINES:] + [msg]
        layout['log'].update(Text(''.join( msgs ) ))

    log.configure(handlers=[])
    log.add(LOG_FILE_NAME)

    if comm_type=='mock':
        io = MockComm()
    elif comm_type == 'serial':
        io = SerialIO()
    elif comm_type == 'i2c':
        io = I2CIO()

    app.hand = Hand(io, on_error=lambda msg: update_log(msg + '\n'))
    app.hand.set_grip(Grip.Open)

    time.sleep(2.0)

    input_thread = threading.Thread(target=get_input, args=(app,))
    input_thread.start()

    active_column = Style(color="cyan")

    try:
        app.hand.start()

        with Live(layout, console=console, auto_refresh=False, screen=False) as live:
            log.add( update_log, colorize=True )

            while app.run:
                while (not app.key_input.empty()) and (key:=app.key_input.get_nowait()) is not None:
                    handle_key(app, key)

                table = Table(padding=(0, 1))
                table.add_column("Item")
                table.add_column(
                    "Position",
                    header_style=(active_column if app.mode == 0 else None),
                    justify="center",
                )
                table.add_column(
                    "Velocity", header_style=(active_column if app.mode == 1 else None)
                )
                table.add_column(
                    "Torque", header_style=(active_column if app.mode == 2 else None)
                )
                table.add_column(
                    "PWM", header_style=(active_column if app.mode == 3 else None)
                )
                table.add_column("Touch Sensors")

                if not app.position_input_init:
                    app.position_input = app.hand.position
                    app.position_input_init = True

                requested = app.position_input.to_dict()
                position = app.hand.position.to_dict()
                touch = app.hand.touch.to_dict()
                velocity_input = app.velocity_input.to_dict()
                torque_input = app.torque_input.to_dict()
                pwm_input = app.pwm_input.to_dict()

                for finger in position:
                    touch_sensors = touch.get(finger)

                    position_request = f'{requested[finger]:5.2f}'
                    position_actual = f'{position[finger]:5.2f}'
                    position_col = (
                        f"{position_request: ^8}|{position_actual: ^10}"
                    )
                    velocity_col = f"{velocity_input[finger]: ^10}"
                    torque_col = f"{torque_input[finger]: ^10}"
                    pwm_col = f"{pwm_input[finger]: ^10}"
                    touch_col = ""
                    if touch_sensors:
                        touch_col = Columns(
                            [f"{p:> 12.3f}" for p in touch_sensors], equal=True
                        )

                    table.add_row(
                        finger,
                        position_col,
                        velocity_col,
                        torque_col,
                        pwm_col,
                        touch_col,
                    )

                layout["table"].update(table)
                st = app.hand.stats()
                layout["stats"].update(
                    Text(
                        f"Baud: RX[{st.rx_baud:> 10d}]  TX[{st.tx_baud:> 10d}]  Packets: TX[{st.tx_packets}] RX[{st.rx_packets}] Dropped[{st.tx_packets - st.rx_packets}]"
                    )
                )
                live.refresh()
#                live.update(layout, refresh=True)
                time.sleep(0.060)

            app.run = False

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.exception("error")
    finally:
        print("Shutting down")
        app.hand.stop()

    app.run = False
    input_thread.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--comm', help="'i2c' or 'serial', 'mock', default: serial", default="serial")

    args = parser.parse_args()

    main(args.comm)
