import sys
sys.path.append("/home/astrobotic/Documents/SDK/MotorControllers/roboclaw/Libraries/Python/roboclaw_python")
import time
import os
from pyPS4Controller.controller import Controller
from roboclaw import Roboclaw

# Connect to RoboClaw via UART
rc = Roboclaw("/dev/serial0", 38400)
rc.Open()
ADDR = 128  # Default motor controller address

class RoverController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_R2_press(self, value):
        speed = int((value / 255.0) * 127)
        print(f"Forward: {speed}")
        rc.ForwardM1(ADDR, speed)
        rc.ForwardM2(ADDR, speed)

    def on_L2_press(self, value):
        speed = int((value / 255.0) * 127)
        print(f"Backward: {speed}")
        rc.BackwardM1(ADDR, speed)
        rc.BackwardM2(ADDR, speed)

    def on_L1_press(self):
        print("Turning left")
        rc.BackwardM1(ADDR, 60)
        rc.ForwardM2(ADDR, 60)

    def on_R1_press(self):
        print("Turning right")
        rc.ForwardM1(ADDR, 60)
        rc.BackwardM2(ADDR, 60)

    def on_x_press(self):
        print("Stopping")
        rc.ForwardM1(ADDR, 0)
        rc.ForwardM2(ADDR, 0)

# Wait until controller appears
print("Waiting for /dev/input/js0 (PS4 controller)...")
while not os.path.exists("/dev/input/js0"):
    time.sleep(1)

controller = RoverController(interface="/dev/input/js0", connecting_using_ds4drv=False)
controller.listen()
