#%%
from DCML_Master import Master
from DCML_Worker_TIMESLOT_MultiProcess import Worker
from DCML_ENVs.DCML_utils.DCML_Config import *
from DCML_ENVs.DCML_utils.DCML_ENV_Functions import calculate_reward
from DCML_ENVs.DCML_utils.DCML_ActionSpace import Action_Space
from DCML_ENVs.DCML_Basic_Env import DCML_Basic_Env
import math
import numpy as np
import random



#%%
class Env(DCML_Basic_Env):
    def __init__(self,central_execution = True,fixed= False,preset=False,multi_agent = True,master_class=Master,worker_class=Worker) -> None:
        super(Env, self).__init__(central_execution=central_execution,fixed=fixed,
                                preset=preset,multi_agent=multi_agent,
                                master_class=master_class,worker_class=worker_class)
        self.multi_agent = multi_agent
        self.master = master_class(WORKER_NUMBER_MAX,preset = preset)
        self.workers = []
        self.fixed = fixed
        self.preset = preset
        if preset:
            with open('data/dcml_benchmark/Sample_1master_states.npy',"rb") as reader:
                self.master_status = np.load(reader,allow_pickle=True)
            with open('data/dcml_benchmark/Sample_1worker_states.npy',"rb") as reader:
                self.worker_Prs = np.load(reader,allow_pickle=False)
                self.disable_rates = np.load(reader,allow_pickle=False)
                
            self.eval_episode_i = 0
        with open('data/workloads.txt','rb') as reader:
            for i in range(WORKER_NUMBER_MAX):
                workload = np.load(reader,allow_pickle=True)
                # print(workload)
                self.workers.append(Worker(workload=workload,network_heterogeneous=HETEROGENEOUS))
        self.n_agents = WORKER_NUMBER_MAX + EXTRA_AGENT 
        # self.ps = [Process(target=worker_process,args=(self.master.get_workload(),self.master.Pr))]
        if self.multi_agent:
            self.observation_space = [[LOCAL_OBS_DIM]]*(self.n_agents)
        else:
            self.observation_space = [[LOCAL_OBS_DIM * (self.n_agents)]]
        self.share_observation_space = [[SOB_DIM]]
        self.disable_rate = 50
        if not self.multi_agent:
            self.action_space = [Action_Space(ACTION_DIM,semi_index=-EXTRA_AGENT,extra=True,mixed=True,low=0,high=WORKER_NUMBER_MAX)]
        elif central_execution:
            self.action_space = [Action_Space(ACTION_DIM,semi_index=-EXTRA_AGENT,extra=True)]
        else:
            self.action_space = [Action_Space(ACTION_DIM,mixed=False,high=1,low=0,continuous=False)]*(WORKER_NUMBER_MAX)
            self.action_space.extend([Action_Space(1,mixed=False,high=1,low=0,extra=True,continuous=True)]*EXTRA_AGENT)
        self.upload_trans = []
        self.download_trans = []
        self.update_workers_states()
    
    def step(self,action,shannon_enable = False,standalone=False):
        if self.fixed:
            
            strategy_2 = np.array([1 if worker.available else 0 for worker in self.workers])
            N = sum(strategy_2)
            K = int(np.floor(N * 0.7))
            
        else:
            action = action.reshape(-1)
            # ratio = 0
            # for i in range(EXTRA_AGENT):
            #     ratio +=  action[-i-1] * (2**i) 
            # ratio = ratio / (2** EXTRA_AGENT)
            # print(action[-1])
            ratio = action[-1]
            N = int(sum(action[:-EXTRA_AGENT]))
            K = int(math.ceil(N*ratio))
            strategy_2 = action[:-EXTRA_AGENT].reshape(-1)
            # strategy_2 = strategy_2 * np.array([1 if worker.available else 0 for worker in self.workers])
            # print(len(strategy_2))
        # print(self.workers[0].local_workload[self.arrive_time:self.arrive_time+3])
        # arrive_time = random.randint(0,LOCAL_WORKLOAD_PERIOD)
        arrive_time = self.arrive_time
        n_retries = []
        if standalone or N == 0:
            self.master.update_strategy(K=1,N=1)
            n_retry, delay, cost = self.workers[0].process(self.master.get_workload(),self.master.Pr,arrive_time)
            ob,s_ob,ava = self.reset()
            if random.random() < CONTINUE_PROBABILITY:
                dones = np.array([True]*(WORKER_NUMBER_MAX+EXTRA_AGENT))
            else:
                dones = np.array([False]*(WORKER_NUMBER_MAX+EXTRA_AGENT))
            if self.multi_agent:
                return ob,s_ob,np.array([1.5*calculate_reward(delay,cost[-1])]*(WORKER_NUMBER_MAX+EXTRA_AGENT)).reshape(-1,1),dones,[{"delay":delay,"payment":cost[-1]}],ava
            else:
                return ob,s_ob,np.array([1.5*calculate_reward(delay,cost[-1])]).reshape(-1,1),dones,[{"delay":delay,"payment":cost[-1]}],ava
        delays = []
        expected_delays = []
        costs = []
        if N> WORKER_NUMBER_MAX:
            N = WORKER_NUMBER_MAX
        elif N<1:
            N = 1
        if K> N:
            K = N
        elif K < 1:
            K = 1
        # print("======",K,"+++++",N,"-----",sum(np.array([1 if worker.available else 0 for worker in self.workers])))
        self.master.update_strategy(K=K,N=N)
        for i in range(WORKER_NUMBER_MAX):
            
            n_retry, delay, cost = self.workers[i].process(self.master.get_workload(),self.master.Pr,arrive_time)
            n_retries.append(n_retry)
            delays.append(delay)
            costs.append(cost)
            
        
        # for i in range(WORKER_NUMBER_MAX):
        #     worker = self.workers[i]
        #     threads.append(Process(target=worker_process,args=(worker,self.master.get_workload(),self.master.Pr,arrive_time)))
        #     threads[-1].start()
        #     # delay, cost, cost_s,n_retry = worker.process(self.master.get_workload(),self.master.Pr,arrive_time)
        
        # for thread in threads:
        #     thread.join()
        
        # for i in range(WORKER_NUMBER_MAX):
        #     n_retries.append(self.workers[i].n_retry)
        #     delays.append(self.workers[i].delay)
        #     costs.append(self.workers[i].cost)

        delays = strategy_2*np.array(delays) 
        final_delay = np.trim_zeros(sorted(delays))[K-1]
        end_timeslot = math.ceil(final_delay)
        final_costs = []
        for cost in costs:
            if len(cost)>= end_timeslot:
                final_costs.append(cost[end_timeslot-1])
            else:
                final_costs.append(cost[-1])
        payment = np.sum(strategy_2*np.array(final_costs))

        ob,s_ob,ava = self.reset()
        
        # Using done to indicate whether next step is just following the current step
        done = (random.random() < CONTINUE_PROBABILITY)
        rewards,dones,info = self.reorganize_step_output(done,final_delay,payment)
        return ob,s_ob,rewards,dones,info,ava

    def update_workers_states(self):
        # According to the poisson distribution, get the number of tasks
        # that are running on the workers
        if POISSON:
            local_work_delays = np.random.poisson(lam=2, size = WORKER_NUMBER_MAX)
            for i in range (WORKER_NUMBER_MAX):
                # Given the number of tasks, computation delay for every task
                # are generated with uniform random.
                self.workers[i].update_local_delay(np.random.uniform(MIN_LOCAL_TASK_DELAY,MAX_LOCAL_TASK_DELAY,local_work_delays[i]))
        
    
    def reset(self,arrive_time=None,shannon_enable = False,binary = False):
        self.disable_rate = random.randint(1,80)
        if arrive_time == None:
            arrive_time = random.randint(0,LOCAL_WORKLOAD_PERIOD-1)
        self.arrive_time = arrive_time
        shared_obs = []
        # Row number and column number 64
        if binary:
            R,C,Pr = self.master.reset(shannon_enable,normalize=False)
            R_b = "{:032b}".format(R)
            R_bi = list(map(int, list(R_b)))
            C_b = "{:032b}".format(C)
            C_bi = list(map(int, list(C_b)))
            
            shared_obs.extend(R_bi)
            shared_obs.extend(C_bi)
        else:
            if self.preset:
                R = self.master_status[self.eval_episode_i,0]
                C = self.master_status[self.eval_episode_i,1]
                Pr = self.master_status[self.eval_episode_i,2]
                R,C,Pr = self.master.fake_reset(R,C,Pr,shannon_enable,normalize=True)
            else:
                R,C,Pr = self.master.reset(shannon_enable,normalize=True)
            shared_obs.append(R*STATE_RATIO)
            shared_obs.append(C*STATE_RATIO)
        # Transmission Rate 3000*2(shannon) & 1(Pr)
        self.update_workers_transmission(shannon_enable)
        self.update_workers_states()
        
        obs = []
        workloads = []
        ava_worker_Prs = []
        ava_workloads = []
        if self.preset:
            worker_Prs = self.worker_Prs[self.eval_episode_i]
            self.disable_rate = self.disable_rates[self.eval_episode_i]
            self.eval_episode_i += 1
            # print(self.eval_episode_i)
        else:
            worker_Prs = np.random.uniform(PR_MIN,PR_MAX,WORKER_NUMBER_MAX)
            
        ava_worker_indices = np.random.choice(WORKER_NUMBER_MAX,self.disable_rate,replace=False)
        # Workload 3000*3
        disabled = 0
        if OBSERVER_WORKLOAD:
            for i in range(len(self.workers)):
                worker = self.workers[i]
                obs.extend(shared_obs[:2])
                if i in ava_worker_indices:
                    worker.available = False
                    workload = worker.bid(TIME_SLOT,worker_Prs[i])
                    obs.extend([1]*4)
                    if len(obs)>7:
                        obs.append(obs[-7])
                    else:
                        obs.append(0)
                    if DYNAMIC_PRICE:
                        obs.append(UNAVAILABLE_PRICE)
                    disabled += 1
                else:
                    worker.available = True
                    ava_worker_Prs.append(worker_Prs[i])
                    workload = worker.bid(TIME_SLOT,worker_Prs[i])

                    workloads.extend(workload[arrive_time:arrive_time+3])
                    ava_workloads.extend(workload[arrive_time:arrive_time+3])
                    if HETEROGENEOUS:
                        workloads.append(worker.Pr)
                    workloads.append((i-disabled)/(WORKER_NUMBER_MAX-self.disable_rate))
                    obs.extend(workloads[-5:])
                    if DYNAMIC_PRICE:
                        obs.append(worker.price)
                # if arrive_time != observe_time:
                #     workloads.extend([workload[arrive_time],np.mean(workload[arrive_time:observe_time]),np.median(workload[arrive_time:observe_time])])
                # else:
                #     workloads.extend([workload[arrive_time],workload[arrive_time],workload[arrive_time]])
            # arr_workload = np.array(workloads).reshape(-1,5)
            arr_workload = np.array(ava_workloads).reshape(-1,3)
            mu = np.mean(arr_workload,axis=0)
            workloads.extend([mu[0],mu[1],mu[2],np.mean(ava_worker_Prs),1.1]*EXTRA_AGENT)
            obs.extend(shared_obs[:2])
            obs.extend(workloads[-5:])
            if DYNAMIC_PRICE:
                obs.append(MASTER_PRICE)
            obs=[obs]
        else:
            arrive_time_one_hot = np.zeros(20)
            arrive_time_one_hot[arrive_time] = 1
            obs.extend(arrive_time_one_hot)
        # Append data trans parameter
        if shannon_enable:
            shared_obs = np.append(np.array(shared_obs),np.array(self.upload_trans)/(10**7))
            shared_obs = np.append(np.array(shared_obs),np.array(self.download_trans)/(10**7))
        else:
            if HETEROGENEOUS:
                shared_obs.extend(worker_Prs)
            else:
                shared_obs.append(Pr)
        
        # print(obs[0][:7])
        sobs = [shared_obs]
        
        # state.append(arrive_time)
        # state_1 = np.array(shared_obs)
        
        # worker_ava = [1]*2
        # ava = [worker_ava]*(WORKER_NUMBER_MAX+EXTRA_AGENT)
        
        ava = [worker.get_availability() for worker in self.workers]
        ava.append([1,1])
        # ava.append([1]*ACTION_DIM) 
        if self.multi_agent:
            shared_obs = sobs*(WORKER_NUMBER_MAX+EXTRA_AGENT)
            return np.array(obs).reshape(-1,LOCAL_OBS_DIM),np.array(shared_obs).reshape(-1,SOB_DIM),np.array(ava)
        else:
            shared_obs = sobs
            return np.array(obs).reshape(-1),np.array(shared_obs).reshape(-1),np.array(ava)
    def fake_reset(self,R,C,Pr,arrive_time,shannon_enable = False,binary = True):
        
        state = []
        if binary:
            R,C,Pr = self.master.fake_reset(R,C,Pr,shannon_enable,normalize=False)
            R_b = "{:032b}".format(R)
            R_bi = list(map(int, list(R_b)))
            C_b = "{:032b}".format(C)
            C_bi = list(map(int, list(C_b)))
            state.extend(R_bi)
            state.extend(C_bi)
        else:
            R,C,Pr = self.master.fake_reset(R,C,Pr,shannon_enable,normalize=True)
            state.append(R*STATE_RATIO)
            state.append(C*STATE_RATIO)
        self.update_workers_transmission(shannon_enable)
        self.update_workers_states()
        if shannon_enable:
            state = np.append(np.array(state),np.array(self.upload_trans)/(10**7))
            state = np.append(np.array(state),np.array(self.download_trans)/(10**7))
            return state
        else:
            state.append(Pr)
        workloads = []
        if OBSERVER_WORKLOAD:
            for worker in self.workers:
                workload = worker.bid(TIME_SLOT)
                # observe_time = arrive_time+10 if arrive_time <= 89 else 99
                workloads.append(workload[arrive_time])
                # if arrive_time != observe_time:
                #     workloads.extend([workload[arrive_time],np.mean(workload[arrive_time:observe_time]),np.median(workload[arrive_time:observe_time])])
                # else:
                #     workloads.extend([workload[arrive_time],workload[arrive_time],workload[arrive_time]])
            state.extend(workloads)
        else:
            arrive_time_one_hot = np.zeros(20)
            arrive_time_one_hot[arrive_time] = 1
            state.extend(arrive_time_one_hot)
        # state.append(arrive_time)
        state_1 = np.array(state)
        return state_1
    def generate_preset_data(self,n_episodes,shannon_enable=False,Row=None,Col=None,Probability=None,disable_rate = None,dir_name="./"):
        # master: Row number, Column number, Prs
        master_states = []
        disable_rates = []
        for i in range(n_episodes):
            
            
            R,C,Pr = self.master.reset(shannon_enable,normalize=False)
            if Row is not None:
                R = Row
            if Col is not None:
                C = Col
            if Probability is not None:
                Pr = Probability
            if disable_rate is None:
                d_r = random.randint(1,80)
                disable_rates.append(d_r)
            master_states.append([R,C,Pr])
        master_states = np.array(master_states)
        with open(dir_name+'master_states.npy', 'wb') as f:
            np.save(f,master_states)
        
        # worker: probabilities
        worker_Prs = np.random.uniform(PR_MIN,PR_MAX,(n_episodes,WORKER_NUMBER_MAX))
        
        with open(dir_name+'worker_states.npy', 'wb') as f:
            np.save(f,worker_Prs)
            np.save(f,np.array(disable_rates))
    def modify_preset(self,R=None,C=None,Pr=None,disable_rate=None):
        if R is not None:
            self.master_status[:,0]=R
        if C is not None:
            self.master_status[:,1]=C
        if disable_rate is not None:
            # self.disable_rates[:] = np.clip(np.random.poisson(lam=disable_rate,size=WORKER_NUMBER_MAX),a_min=0,a_max=80)
            self.disable_rates[:] = disable_rate 
        if Pr is not None:
            self.worker_Prs[:] = Pr
    def close(self):
        pass
# %%
