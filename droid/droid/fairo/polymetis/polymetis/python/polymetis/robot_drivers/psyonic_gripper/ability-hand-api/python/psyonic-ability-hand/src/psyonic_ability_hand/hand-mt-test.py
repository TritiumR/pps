import time

from psyonic_ability_hand import log
from psyonic_ability_hand.hand import Hand, MockComm
from psyonic_ability_hand.io import SerialIO


def main():
    log.info("starting mt test")

    hand = Hand(SerialIO())

    try:
        hand.start()

        t = 0
        while True:
            t += 1
            time.sleep(1)
            print(hand.stats())
    except KeyboardInterrupt:
        pass
    finally:
        hand.stop()

    print("Hand stopped")
    print(hand.stats())


if __name__ == '__main__':
    main()