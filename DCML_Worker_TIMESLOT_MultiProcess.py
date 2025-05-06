#%%
from math import ceil,floor
import random
import numpy as np
TASK_ARRIVAL_RATE = 5
TASK_SET = [[0.1* 10**9,10],
            [0.5* 10**9,15],
            [  1* 10**9,20]]
PERIOD = 20
LAMBDA_OF_POISSON = 3
SECOND_TO_CENTSEC = 1
BIT_TO_BYTE = 4

class Worker():
    # TODO : Extract common part of worker to remove duplicate code and gurantee the synchronization of worker
    def __init__(self,workload = None,network_heterogeneous = False,frequency = 2e9,standard_price = 1) -> None:
        self.frequency = frequency
        self.upload = 0
        self.download = 0
        self.local_workload = []
        self.local_workload_delay = 0
        self.delay = 0
        self.expected_delay = 0
        if standard_price is None:
            self.standard_price = 1
        else:
            self.standard_price = standard_price
        self.price = self.standard_price
        self.cost = 0
        self.costs = []
        self.n_retry = 0
        self.Pr = 0.95
        self.network_heterogeneous = network_heterogeneous
        self.no_avg_aperiodic = sum(np.random.poisson(lam=LAMBDA_OF_POISSON,size=PERIOD))/PERIOD
        self.standard_price = 1
        self.available = True
        if workload is not None:
            self.workload = workload
            self.all_workload = np.clip(self.workload * (np.random.uniform(0.8,1.2,len(self.workload))),0,1)
            # This is the timepoint, which defines the workloads at the timeslot 0
            # Dynamic:
            # self.timepoint = random.randint(0,20)
            # Constant
            self.timepoint = 0
        pass
    def process(self,workload,Pr,arrive_time):
        r,c = workload
        # print(r,"\t",c,"\t")
        compute_workload = (9*r-3)*c
        cost = SECOND_TO_CENTSEC * ceil(compute_workload)/self.frequency
        # print(cost)
        # print(compute_workload)
        n_retry = 1
        if self.network_heterogeneous:
            while random.uniform(0,1)<self.Pr:
                n_retry += 1
        else:
            while random.uniform(0,1)<Pr:
                n_retry += 1
        transmit_delay = SECOND_TO_CENTSEC * (((ceil(((r+1)*c)))*1*BIT_TO_BYTE)/self.download+0.001)*n_retry #+0.01
        # print(transmit_delay)
        # print(cost, transmit_delay)
        # Check whether ML arrives at worker's queue in current time slot
        # Remake needed
        price = floor(transmit_delay)*0.1
        arrive_timeslot = int(floor(transmit_delay+arrive_time))
        current_timepoint = arrive_timeslot + self.timepoint
        while current_timepoint > PERIOD-1:
            current_timepoint -= PERIOD
        
        
        finish_timeslot = arrive_timeslot
        
        availability = 0
        
        
        # Check whether the ML arrives after worker finish current timeslot's local workload
        # If yes, the cost of this timeslot is actually more than expected
        # if transmit_delay%1 > self.all_workload[current_timepoint]:
        #     cost += transmit_delay%1 - self.all_workload[current_timepoint]
        # while availability < cost:
        #     available_computation = 1 - self.all_workload[current_timepoint]
        # print(self.available,self.local_workload)
        prices = []
        if transmit_delay%1 > self.local_workload[current_timepoint]:
            cost += transmit_delay%1 - self.local_workload[current_timepoint]
        while availability < cost:
            available_computation = 1 - self.local_workload[current_timepoint]
            price += available_computation * self.standard_price
            prices.append(price)
            availability += available_computation
            current_timepoint += 1
            finish_timeslot += 1
            if current_timepoint > PERIOD-1:
                current_timepoint -= PERIOD
        # print(transmit_delay,"=====",cost)
        # self.expected_delay = self.local_workload_delay + ceil(compute_workload)/self.frequency

        # Upload
            if self.network_heterogeneous:
                while random.uniform(0,1)<self.Pr:
                    n_retry += 1
            else:
                while random.uniform(0,1)<Pr:
                    n_retry += 1
            upload_delay =  SECOND_TO_CENTSEC * (((ceil(r))*1*BIT_TO_BYTE)/self.download+0.001)*n_retry +0.02
        
        self.delay = finish_timeslot - arrive_time - (availability - cost) + upload_delay
        self.cost = price
        self.n_retry = n_retry
        self.all_workload = np.clip(self.workload * (np.random.uniform(0.8,1.2,len(self.workload))),0,1)
        return n_retry, self.delay, prices
    # bid: report the workload of following timeslots; also privde price for every unit of computation.
    def bid(self,size,Pr=0):
        # TODO: Extract the bid procedure as a common function
        # if self.timepoint <=200:
        #     workload = self.all_workload[self.timepoint:self.timepoint+100]
        #     self.timepoint += 100
        # else:
        #     workload = self.all_workload[self.timepoint:]
        #     workload = np.append(workload,self.all_workload[:self.timepoint-200])
        #     self.timepoint -= 200
        self.price = 1 * self.no_avg_aperiodic/LAMBDA_OF_POISSON
        self.no_avg_aperiodic = sum(np.random.poisson(lam=LAMBDA_OF_POISSON,size=PERIOD))/PERIOD
        if self.network_heterogeneous:
            self.Pr = Pr
        else:
            self.Pr = Pr
        workload = self.all_workload[self.timepoint:]
        workload = np.append(workload,self.all_workload[:self.timepoint-PERIOD])
        workload = np.append(workload,workload)
        # print(len(workload))
        self.local_workload = workload
        
        return workload
    def update_local_delay(self, local_delays):
        self.local_workload = local_delays
        self.local_workload_delay = sum(self.local_workload)
    def update_local_workload_delay(self):
        self.local_workload_delay = random.random()
    def update_transmission_rate(self,upload,download):
        self.upload = upload
        self.download = download
    def get_availability(self):
        if self.available:
            return [1,1]
        else:
            return [1,0]
# %%
