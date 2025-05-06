from .DCML_utils.DCML_Config import MIN_WORKER_POWER,MAX_WORKER_POWER,WORKER_NUMBER_MAX,NON_SHANNON_DATA_RATE,EXTRA_AGENT
from .DCML_utils.DCML_ENV_Functions import calculate_reward
import numpy as np


class DCML_Basic_Env():
    def __init__(self,central_execution = True,fixed= False,preset=False,multi_agent = True,master_class = None,worker_class = None):
        self.multi_agent = multi_agent
    def reorganize_step_output(self,done,final_delay,payment):
        if self.multi_agent:
            # print("Multi agent")
            dones = np.array([done]*(WORKER_NUMBER_MAX+EXTRA_AGENT))
            return np.array([calculate_reward(final_delay,payment)]*(WORKER_NUMBER_MAX+EXTRA_AGENT)).reshape(-1,1),dones,[{"delay":final_delay,"payment":payment}]
        else: 
            # print("Single agent")
            dones = np.array([done])
            return np.array([calculate_reward(final_delay,payment)]),dones,[{"delay":final_delay,"payment":payment}]
    def update_workers_transmission(self,shannon_enable = False):
        if shannon_enable:
            worker_power = np.random.uniform(MIN_WORKER_POWER,MAX_WORKER_POWER,WORKER_NUMBER_MAX)
            upload_trans,download_trans = self.master.get_transmission_rate(worker_power)
            self.upload_trans = upload_trans
            self.download_trans = download_trans
            # print(upload_trans,"\n",download_trans)
            for i in range(WORKER_NUMBER_MAX):
                self.workers[i].update_transmission_rate(upload_trans[i],download_trans[i])
                self.workers[i].update_local_workload_delay()
        else:
            self.upload_trans = np.ones(self.master.I)
            self.download_trans = np.ones(self.master.I)
            for i in range(WORKER_NUMBER_MAX):
                self.workers[i].update_transmission_rate(NON_SHANNON_DATA_RATE,NON_SHANNON_DATA_RATE)
                self.workers[i].update_local_workload_delay()