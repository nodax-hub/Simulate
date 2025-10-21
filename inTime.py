
# /// script
# dependencies = ["pigpio",
#                 "simple_pid",
#                 "matplotlib",
#                 "scipy",
#                 "colorlog",
#                 "gpiozero",
#                 "lgpio",
#                 "RPi.GPIO",
#                 "simple-pid"]
# requires-python = ">=3.10"
# ///





import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Literal

import colorlog
import numpy as np
import pigpio
from simple_pid import PID




def main():

    pi = pigpio.pi()
    if not pi.connected:
        return

    gpio_in = 23

    impulses = 0
    def impulse_cb():
        nonlocal impulses
        impulses += 1


    pi.set_mode(gpio_in, pigpio.INPUT)
    pi.set_pull_up_down(gpio_in, pigpio.PUD_OFF)
    _cb = pi.callback(gpio_in, pigpio.RISING_EDGE, impulse_cb)

    print()

    pwm = 1500
    freq_hz = 50
    gpio_pwm = 18

    pi.set_mode(gpio_pwm, pigpio.OUTPUT)
    pi.set_PWM_frequency(gpio_pwm, freq_hz)


    wait_time = 60  # sec

    try:
        pi.set_servo_pulsewidth(gpio_pwm, pwm)
        time.sleep(wait_time)
    finally:
        pi.set_servo_pulsewidth(gpio_pwm, 0)

    print(f"Impulses : {impulses}")



if __name__ == "__main__":
    main()
