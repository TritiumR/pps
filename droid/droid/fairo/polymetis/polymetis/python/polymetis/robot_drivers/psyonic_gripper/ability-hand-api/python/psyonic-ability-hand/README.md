# Psyonic Hand Interface

# Overview

The hand communicates using a bidirectional, asynchronous RS-485/RS-422 connection. A thread is created to send packets to the hand, and a separate thread reads response packets from the hand. 
In the various control modes, the rate at which packets are sent largely determines the overall quality of motion (smoothness). When a packet is sent, the hand moves a small increment towards the desired position (in position control mode). Therefore, the packets are sent at a fixed, high rate of speed (configurable as `HAND_V2_TX_SEC` in `hand.py`). When the caller commands a new position, the position data is cached and the transmit thread sends the new position repeatedly. 
The hand, however, does not send unsolicited responses; it must be polled for status. When a position, velocity or torque is set, one of the provided parameters is the 'reply type'. The hand can respond to any of the control commands in one of three formats. Typically, only the first format is used, which provided a response containing position, current and touch sensor data. As packets are received, internal states variables are updated with the current position, current, velocity, torque, etc., depending on the requested 'reply type'.  
Due to a variety of reasons, packets may be lost either in transmission or reception. While the hand does support a constant request/response cycle, lost packets coupled with the fact that the response packets do not contain any sort of counter or indicator of the associated transmit packet, the transmit and receive operations are largely decoupled from one another. Packets are sent to the hand as fast as is reasonable, and the hand responds as fast as it can.  
Transmit packet sizes are 25 bytes. This provides a theoretical maximum rate of around 921 packets per second at 230400 baud. Response packet sizes, on the other hand, for the primary response format, are 72 bytes. This provides a theoretical maximum rate at 230400 baud of around 320 packets/second. 

# Development

This is a [python-poetry](https://python-poetry.org/) based project. To install locally,
after installing poetry via the link, run
```
poetry install
```

## Configuration

Then hand comes default configured for I2C communications. However then hand needs to be
configured for UART mode communication (which may require a Psyonic firmware upgrade).

Also, most of the current code defaults to 230400 UART speed, whereas the hand defaults
to 460800.

## Running the demos

After installing with `poetry install`, run

```
poetry shell
```
to create a local python environment.

### hand-demo.py

```
python src/psyonic_ability_hand/hand-demo.py
```

This provides a basic interface for controlling each finger in position or velocity mode,
as well as viewing the touch data.

### hand-grasp-test.py

```
python src/psyonic_ability_hand/hand-grasp-test.py
```

A simpler tool for stepping through opening and closing the gripper


## Hand Interface API

`hand.py` contains `Hand` object, a typical usage might be:

```py
from psyonic_ability_hand.hand import Hand
from psyonic_ability_hand.io import SerialIO

hand = Hand(SerialIO())

# packet transmission to and from the hand is asyncronous and must be started
# prior to sending commands
hand.start()

hand.set_position( JointData(0, 0, 0, 0, 0, -10) )

hand.stop()
```

Most of the data passed to the interface is in the form of the JointData object:

```py
class JointData:
    Index: float = 0
    Middle: float = 0
    Ring: float = 0
    Pinky: float = 0
    ThumbFlexor: float = 0
    #NOTE: the ThumbRotator is inverted, -10 is the lower limit, around -90 the upper
    ThumbRotator: float = 0
```

The basic commands are:

```py
get_position()
set_position( JointData() )
```
With positions, JointData represents angles in degress of each finger, from `0` to `150`
(with a practical limit for most fingers around `90`).

*NOTE: in tests, the hand, due to hardware/firmware limitations, may stop approaching a
position within a few degrees of the target, even if it's physically possible for it
to reach the exact target.*

The general 'smoothness' of position movements is also highly depending on the packet
transmission interval, currently 20ms. The hand will move in small incremenents towards
the target position with each packet sent.

```py
set_velocity( JointData() )
```

In velocity set mode, the `JointData` provides degrees per second for each finger.


Additionally, the status of the Touch Sensors can be retrieved via:

```py
get_touch()
```
