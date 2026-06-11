import time
import serial

#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)

port = "/dev/ttyUSB0"
baud = 230400

print(f"opening {port} at {baud}")
#ser=serial.Serial("/dev/ttyS0", baud)
ser=serial.Serial(port, baud, timeout=0.100)

tx=b"abcdefghijgklmnopqrtuvwxyz" * 100

print(f"test message: {len(tx)} bytes")

n = len(tx)
bps_expected = int( baud / 10 )
errors = 0
start = time.time()
while True:
    ser.write(tx)
    d = ser.read(n)

    dt = 0
    bps= 0

    if len(d) != 0:
        end = time.time()
        dt = end-start
        start = end
        bps = int(n/dt)

        if len(d) != len(tx):
            errors += 1
        elif d != tx:
            errors += 1
    else:
        errors += 1

    print(f"{dt:0.6f}  B/s:{bps}/{bps_expected} Errors:{errors}",end='\r')
