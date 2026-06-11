import time
from psyonic_ability_hand import log
from psyonic_ability_hand.hand import Hand, MockComm
from psyonic_ability_hand.io import SerialIO


def main():
    log.info("starting grasp test")

#    hand = Hand(MockComm())
    hand = Hand(SerialIO())

    try:
        hand.start()

        hand.grasp(width=0, speed=.1)
        time.sleep(10)

        hand.grasp(width=1, speed=.1)
        time.sleep(10)

        hand.grasp(width=0, speed=.4)
        time.sleep(6)

        hand.grasp(width=.5, speed=.4)
        time.sleep(6)

        hand.grasp(width=0, speed=.3)
        time.sleep(6)

        hand.grasp(width=1, speed=.3)
        time.sleep(6)

        hand.grasp(width=0, speed=6)
        time.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        hand.stop()

    print("Hand stopped")
    print(hand.stats())


if __name__ == '__main__':
    main()