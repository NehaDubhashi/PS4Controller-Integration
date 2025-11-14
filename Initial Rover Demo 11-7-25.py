import time
from datetime import datetime
import pandas as pd 
import os
import sys
import threading
import numpy as np
from scipy import signal
from rtlsdr import RtlSdr

sys.path.append("/home/astrobotic/Documents/SDK/MotorControllers/roboclaw/Libraries/Python/roboclaw_python")
from introspection import Settings, process_file, process_data

from scipy.io import savemat
from functools import reduce
import scipy.io as spio

from pyPS4Controller.controller import Controller
from roboclaw_3 import Roboclaw

rc = Roboclaw("/dev/ttyS0", 38400)
rc.Open()
address1 = 0x80
address2 = 0x81

# === Define Functions ===
def accumulate(sq, sq_90, samples, num_it, samp):
    a_val = np.empty((num_it, 1))
    l = 0
    a_i, a_q = 0, 0
    for i in range(0, num_it * samp, samp):
        a_i = np.sum(sq[i:i+samp] * np.abs(samples[i:i+samp]))
        a_q = np.sum(sq_90[i:i+samp] * np.abs(samples[i:i+samp]))
        a_val[l] = np.sqrt(np.abs(np.power(a_i,2) + np.power(a_q,2)))
        l += 1
    a_val = a_val[~np.isnan(a_val)] 
    return a_val

def getBinary(a_val, th):
    s_d = np.empty((len(a_val), 1))
    for i in range(len(a_val)):
        if a_val[i] >= th:
            s_d[i] = 1
        else:
            s_d[i] = 0 
            
    return s_d

def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

class RadioRecorder:
    def __init__(self, center_freq=915e6, sample_rate=1e6, gain=2):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq
        self.sdr.gain = gain
        self.recording = False
        self.stop_flag = False
        self.thread = None
        self.samples = None

    def _record_loop(self):
        sample_blocks = []
        block_size = 995328 # 256 * 1024
        print("RadioRecorder: Recording started.")
        
        # equivalent to: samples = sdr.read_samples(N*duration)
        # only 90 seconds of recording
        start_time = time.time()
        while not self.stop_flag and (time.time() - start_time) < 90:
            block = self.sdr.read_samples(block_size)
            sample_blocks.append(block)
            
        if sample_blocks:
            self.samples = np.concatenate(sample_blocks)
        else:
            self.samples = np.array([], dtype=np.complex64)
            
        # equivalent to N = len(samples)
        if self.samples.size > 0:
            fs = self.sdr.sample_rate
            fb = 100
            dr = 1
            
            N = len(self.samples)
            t = np.arange(0, N / fs, 1/fs)
            
            # Generate Square Waves 
            sq = signal.square(2*np.pi*fb*t, duty = 0.5)
            sq_90 = signal.square(2*np.pi*fb*t + np.pi/2, duty = 0.5) # not clear why we need to do both in-phase and out-of-phase
            
            max_cycles = fb / dr
            samps_per_bit = 2
            cycles = int(np.ceil(max_cycles) / samps_per_bit)
            
            # Accumulator Parameters
            Tmax = (1.0 / fb) * cycles
            samp = round(Tmax / (1.0/fs))
            num_it = int(N / samp) # number of iterations
            
            a_val = accumulate(sq, sq_90, self.samples, num_it, samp)
            s_d = getBinary(a_val, np.mean(a_val))
            bit_sequence = [int(bit) for bit in s_d.flatten()]
            print(f"RadioRecorder: Captured binary sequence ({len(bit_sequence)} bits): {bit_sequence}")

            # save samples to file human decoding
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/home/astrobotic/nasa-dcgr-astrobotic-rover/data/sdr_record_{ts}.npz"
            try:
                np.savez_compressed(output_path,
                                    samples=self.samples,
                                    a_val=a_val,
                                    s_d=s_d,
                                    bit_sequence=np.array(bit_sequence))
                print(f"Saved SDR data snapshot to {output_path}")
            except Exception as e:
                print(f"Failed to save .npz data: {e}")
                
            barker7 = [1, 0, 1, 1, 0, 1, 0]
            packet_length = 20
            temperatures_f = []
            timestamps = []

            for i in range(len(bit_sequence) - packet_length):
                if bit_sequence[i:i+7] == barker7:
                    temp_bits = bit_sequence[i+7:i+20]
                    temp_raw = int("".join(str(b) for b in temp_bits), 2)
                    temp_f = (temp_raw * 0.0625 * 9/5) + 32
                    temperatures_f.append(temp_f)
                    timestamps.append(datetime.now().isoformat())

            # Save results if we captured anything
            if temperatures_f:
                temp_path = f"/home/astrobotic/nasa-dcgr-astrobotic-rover/data/temps_{ts}.npz"
                try:
                    np.savez_compressed(temp_path, temperatures_f=temperatures_f, timestamps=timestamps)
                    print(f"Saved decoded temperatures to {temp_path}")
                except Exception as e:
                    print(f"Failed to save temperature data: {e}")
            else:
                print("No valid temperature packets detected.")
        else:
            print("RadioRecorder: No samples captured (no data recorded or recording was too short).")
        print("RadioRecorder: Recording stopped.")
        self.recording = False

    def start(self):
        if self.recording:
            print("RadioRecorder: Already recording; start command ignored.")
            return
        self.stop_flag = False
        self.recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.recording:
            print("RadioRecorder: Not recording; stop command ignored.")
            return
        self.stop_flag = True
        if self.thread is not None:
            self.thread.join()

    def close(self):
        try:
            self.sdr.close()
        except Exception as e:
            print(f"Warning: SDR device close encountered an error: {e}")

radio_recorder = RadioRecorder()

class BaseController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_square_press(self):
        print("Square button pressed -> Start SDR recording")
        radio_recorder.start()

    def on_circle_press(self):
        print("Circle button pressed -> Stop SDR recording")
        radio_recorder.stop()

    def on_x_press(self):
        print("X button pressed -> Emergency stop: stopping motors and exiting")
        rc.SpeedM1(address1, 0)
        rc.SpeedM2(address1, 0)
        rc.SpeedM1(address2, 0)
        rc.SpeedM2(address2, 0)
        if radio_recorder.recording:
            radio_recorder.stop()
        radio_recorder.close()
        os._exit(0)

class JoystickController(BaseController):
    def __init__(self, maxSpeed, deadZone, **kwargs):
        super().__init__(**kwargs)
        self.maxSpeed = maxSpeed
        self.deadZone = deadZone

    def mapSpeed(self, value):
        if abs(value) < self.deadZone:
            return 0
        return int((abs(value) / 32767.0) * self.maxSpeed)

    def on_L3_up(self, value):
        speed = self.mapSpeed(value)
        rc.SpeedM1(address1, -speed)
        rc.SpeedM2(address2, -speed)

    def on_L3_down(self, value):
        speed = self.mapSpeed(value)
        rc.SpeedM1(address1, speed)
        rc.SpeedM2(address2, speed)

    def on_R3_up(self, value):
        speed = self.mapSpeed(value)
        rc.SpeedM2(address1, speed)
        rc.SpeedM1(address2, speed)

    def on_R3_down(self, value):
        speed = self.mapSpeed(value)
        rc.SpeedM2(address1, -speed)
        rc.SpeedM1(address2, -speed)

def main():
    try:
        while not os.path.exists("/dev/input/js0"):
            time.sleep(1)

        controller = JoystickController(maxSpeed=12000, deadZone=2000,
                                        interface="/dev/input/js0", connecting_using_ds4drv=False)
        print("Controller ready. Listening for input...")
        controller.listen()

    finally:
        print("Shutting down: stopping all motors and closing SDR.")
        rc.SpeedM1(address1, 0)
        rc.SpeedM2(address1, 0)
        rc.SpeedM1(address2, 0)
        rc.SpeedM2(address2, 0)
        radio_recorder.stop()
        radio_recorder.close()

if __name__ == '__main__':
    main()