#%%
import math
import numpy as np
PATH_LOSS_EXPONENT = -4

class Shannon():
    def __init__(self,power=50,noise=-50) -> None:
        self.power = power# Watt
        self.noise = 10**(noise/10) # dBm -> mW
        # print(self.noise)
        pass
    def update_power(self,power):
        self.power = power
    def download(self,bandwidth,distance):
        transmission_rate = bandwidth * (np.log2(1+(self.power*(distance**PATH_LOSS_EXPONENT))/self.noise))
        # print(np.log2(1+(self.power*(distance**(-4)))/self.noise))
        return transmission_rate
    def upload(self,bandwidth,distance, worker_power):
        transmission_rate = bandwidth * (np.log2(1+(worker_power*(distance**PATH_LOSS_EXPONENT))/self.noise))
        # print(np.log2(1+(worker_power*(distance**(-4)))/self.noise))
        return transmission_rate
# %%
