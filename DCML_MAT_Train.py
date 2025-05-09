#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

sys.path.append("mat_src")
from mat.config import get_config
from DCML_BID_FIRST_MA_ENV_SingleProcess import Env
# from mat.envs.football.football_env import FootballEnv
from dcml_runner import DCMLRunner as Runner
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


# In[2]:


"""Train script for DCML."""

central_execution = True
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():

            env = Env(central_execution = central_execution)
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
            
            env = Env(central_execution = central_execution)
            return env

        return init_env

    if all_args.eval_episodes == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.eval_episodes)])


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='DCML_MAT')
    parser.add_argument('--n_agent', type=int, default=101)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


# In[3]:


#%%
def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("mumu config: ", all_args)

    if all_args.algorithm_name == "rmappo" :
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec"or all_args.algorithm_name == "happo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath("~/dcml_mat/")))[
                       0] + "/results") / all_args.env_name / all_args.scenario / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        with open(run_dir / 'args.txt', 'w') as f:
            f.write(str(args))
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    config['all_args'].use_centralized_V = True
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


# In[4]:


if __name__ == "__main__":
#     argv = ["--num_env_steps","400000","--episode_length","100","--algorithm_name","mat","--env_name","DCML","--scenario","Multi_Agent_Workload_Allocation","--lr","3e-4","--critic_lr","5e-4","--ppo_epoch","15","--num_mini_batch","4","--use_popart","--use_valuenorm"]
#     argv = ["--num_env_steps","600000","--save_interval","50","--episode_length","100","--algorithm_name","mat","--env_name","DCML","--scenario","Multi_Agent_Workload_Allocation","--lr","1e-4","--critic_lr","5e-4","--ppo_epoch","12","--num_mini_batch","8","--use_popart","--use_valuenorm"]
#     argv = ["--num_env_steps","600000","--save_interval","50","--episode_length","200","--algorithm_name","mat","--env_name","DCML","--scenario","Multi_Agent_Workload_Allocation","--lr","5e-4","--critic_lr","3e-4","--ppo_epoch","15","--num_mini_batch","8","--use_popart","--use_valuenorm"]
    argv = ["--n_rollout_threads","8","--num_env_steps","1000000","--save_interval","50","--episode_length","50","--algorithm_name","mat","--env_name","DCML","--scenario","AS","--lr","5e-5","--critic_lr","5e-5","--ppo_epoch","15","--num_mini_batch","4","--gamma","0.99","--use_valuenorm","--use_popart","value_loss_coef","1.5","--entropy_coef","0.01"]
    main(argv)
