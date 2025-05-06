# %%
import sys
import time
import random
import torch
import numpy as np
from collections import deque
sys.path.append("mat_src")
from mat.config import get_config
from DCML_BID_FIRST_MA_ENV_SingleProcess import Env
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import matplotlib.pyplot as plt
# from models.td3_agent_test import Agent
from dcml_runner import DCMLRunner as Runner

# %%
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():

            env = Env()
            # env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            
            env = Env(preset=True)
            # env.modify_preset(R=round(1*(2**20)/10),C=(2**9))
            env.modify_preset(R=(2**19),C=(9*((2**10))/10))
            return env

        return init_env

    if all_args.eval_episodes == 1:
        print("=====")
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='DCML_Auction')
    parser.add_argument('--n_agent', type=int, default=3)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    all_args = parser.parse_known_args(args)[0]

    return all_args

# %%
ACTION_INTERVAL = 1
ACTION_DIM = 2
BINARY = False

# %%
env = Env()
obs,s_obs,ava = env.reset(arrive_time=None,binary = BINARY)
state = obs.flatten()
state_size = len(state)
# agent = Agent(state_size=state_size, action_size=202, random_seed=10,action_interval = ACTION_INTERVAL,act_dim=ACTION_DIM)

# agent.actor_local.load_state_dict(torch.load('checkpoint_td3_dcml_actor.pth'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_td3_dcml_critic.pth'))


# %
argv = ["--use_eval","--n_eval_rollout_threads","2","--algorithm_name","mat","--model_dir","./results/DCML/AS/mat/check/run1/models/transformer_1900.pt"]

parser = get_config()
all_args = parse_args(argv, parser)
if all_args.cuda and torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda:0")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
else:
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)
envs = make_train_env(all_args)
eval_envs = make_eval_env(all_args)
num_agents = envs.n_agents
config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": None
    }
runner = Runner(config)

# %%
rewards = []
cts = []
payments = []
# R,C FOR 10 interations
# AW FOR 11 interations
N_ITER = 11
for i in range(N_ITER):
    rewards.append([])
    cts.append([])
    payments.append([])
    env = Env(preset=True)
    # env.modify_preset(R=round((i+1)*(2**20)/10),C=(2**9))
    # env.modify_preset(R=(2**19),C=((i+1)*((2**10))/10))
    env.modify_preset(disable_rate = (i)*8)
    # env.modify_preset(R=(2**19),C=(2**9),Pr = (i)*0.1)
    obs,s_obs,ava = env.reset(arrive_time=None,binary = BINARY)
    for i_episode in range(0, 1000):
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = runner.policy.get_actions(obs=obs,cent_obs=s_obs,rnn_states_actor=None, rnn_states_critic=None, masks=None,available_actions=ava,deterministic=True,stride=10) 
        # print("\r {}" .format(i_episode),end="")
        
        
        obs, s_obs,reward,dones,train_info,ava = env.step(np.array(actions.detach()))
        reward = reward[0]
        ct = train_info[0]['delay']
        payment = train_info[0]['payment']
        next_state = obs.flatten()
        state = next_state
        rewards[-1].append(reward)
        cts[-1].append(ct) 
        payments[-1].append(payment)

# %%
w_cts = []
w_payments = []
for i in range(N_ITER):
    w_cts.append([np.mean(cts[i])])
    w_payments.append([np.mean(payments[i])])
    print('reward:',np.mean(rewards[i]),'ct:',np.mean(cts[i]),'payment:',np.mean(payments[i]))

# %%
with open('dcml_BMAT_RUN1_1900_AW.npy',"wb") as recorder:
    # np.save(recorder,np.array(rewards))
    np.save(recorder,np.array(w_cts))
    np.save(recorder,np.array(w_payments))

# %%



