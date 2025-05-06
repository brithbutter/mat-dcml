#%%
from math import ceil
import random
import numpy as np
from Shannon import Shannon 
R_MIN = 2**10
R_MAX = 2**20
C_MIN = 2**5
C_MAX = 2**10

TRANSMISSION_POWER_UPPER_BOUND = 60 # Watt
TRANSMISSION_POWER_LOWER_BOUND = 50 # Watt
DISTANCE_UPPER_BOUND = 100 # Meter
DISTANCE_LOWER_BOUND = 10
PR_MAX = 0.95
PR_MIN = 0
#%%
class Master():
    def __init__(self,I,test=False,preset=False) -> None:
        self.K = 2
        self.N = 2
        self.I = I
        self.preset = preset
        if test:
            random.seed(1)
        self.R = random.randint(R_MIN,R_MAX)
        self.C = random.randint(C_MIN,C_MAX)
        self.Pr = random.uniform(PR_MIN,PR_MAX)
        self.B_total = 100*10**9 # 300Ghz
        self.B = self.B_total / self.I
        self.work_load = ceil((self.R * self.C) / self.K)
        self.shannon = Shannon()
    def update_work_load(self):
        self.work_load = ceil((self.R * self.C) / self.K)
    def update_strategy(self,K,N):
        self.K = K
        self.N = N
        self.update_work_load()
    def get_workload(self):
        return ceil(self.R/self.K), self.C
    def get_transmission_rate(self,worker_power):
        self.shannon.update_power(random.uniform(TRANSMISSION_POWER_LOWER_BOUND,TRANSMISSION_POWER_UPPER_BOUND))
        distance = np.random.uniform(DISTANCE_LOWER_BOUND,DISTANCE_UPPER_BOUND,self.I)
        worker_power = worker_power
        return self.shannon.upload(self.B,distance,worker_power=worker_power),self.shannon.download(self.B,distance)
    def reset(self,shannon_enable = False,normalize = True):
        self.R = random.randint(R_MIN,round(R_MAX*(1+0.1)))
        self.C = random.randint(C_MIN,round(C_MAX*(1+0.1)))
        if shannon_enable:
            self.Pr = 0
        else:
            self.Pr = random.uniform(PR_MIN,PR_MAX)
        if normalize:
            return (self.R-R_MIN)/(R_MAX-R_MIN),(self.C-C_MIN)/(C_MAX-C_MIN),self.Pr
        else:
            return self.R,self.C,self.Pr
    def fake_reset(self,R,C,Pr,shannon_enable = False,normalize = True):
        self.R = R
        self.C = C
        if shannon_enable:
            self.Pr = 0
        else:
            self.Pr = Pr
        if normalize:
            return (self.R-R_MIN)/(R_MAX-R_MIN),(self.C-C_MIN)/(C_MAX-C_MIN),self.Pr
        else:
            return self.R,self.C,self.Pr
# %%
