import numpy as np

min_pwm, max_pwm = 1200, 2000
num = 161


pwms = np.linspace(min_pwm, max_pwm, num)

total_time_for_test = 5 # min
time_per_pwm = total_time_for_test * 60 / num # in secs


assert time_per_pwm > 3 # длительность каждого теста слишком мала

#0.0057142857 л
