# %%
import sys
import time
import wandb
import numpy as np
from functools import reduce
import torch
# from mat_src import mat
sys.path.append("./mat_src")
from mat_src.mat.runner.shared.base_runner import Runner
# from mat.runner.shared.base_runner import Runner
MULTI_AGENT_LOG = False
# %%
def _t2n(x):
    return x.detach().cpu().numpy()

class DCMLRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(DCMLRunner, self).__init__(config)
        self.use_single_network = False
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        if self.n_objective >1:
            train_episode_rewards = [[0,0] for _ in range(self.n_rollout_threads)]
        else:
            train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        train_episode_delays = [0.0 for _ in range(self.n_rollout_threads)]
        done_episodes_delays = []
        train_episode_payments = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_payments = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                
                # Obser reward and next obs

                
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                if self.all_args.algorithm_name == "momat":
                    # multi objective
                    reward_env = np.mean(rewards, axis=1)
                    train_episode_rewards += reward_env
                else:
                    # single objective
                    reward_env = np.mean(rewards, axis=1).flatten()
                    train_episode_rewards += reward_env

                delay_env = [t_info[0]["delay"] for t_info in infos]
                payment_env = [t_info[0]["payment"] for t_info in infos]
                train_episode_delays += np.array(delay_env)
                train_episode_payments += np.array(payment_env)
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_delays.append(train_episode_delays[t])
                        train_episode_delays[t] = 0
                        done_episodes_payments.append(train_episode_payments[t])
                        train_episode_payments[t] = 0

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                # print(actions[:,-1,0])
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    done_episodes_rewards = []

                    aver_episode_delays = np.mean(done_episodes_delays)
                    self.writter.add_scalars("train_episode_scores", {"aver_scores": aver_episode_delays}, total_num_steps)
                    done_episodes_delays = []
                    aver_episode_payments = np.mean(done_episodes_payments)
                    self.writter.add_scalars("train_episode_scores", {"aver_scores": aver_episode_payments}, total_num_steps)
                    done_episodes_payments = []
                    print("some episodes done, average rewards: {}, delays: {},payments: {}"
                          .format(aver_episode_rewards, aver_episode_delays,aver_episode_payments))

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, ava = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        if self.all_args.algorithm_name == "happo" :
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].share_obs[0] = share_obs[:,agent_id].copy()
                self.buffer[agent_id].obs[0] = obs[:,agent_id].copy()
                if self.buffer[agent_id].available_actions is not None:
                    self.buffer[agent_id].available_actions[0] = ava[:,agent_id].copy()
        else:
            self.buffer.share_obs[0] = share_obs.copy()
            self.buffer.obs[0] = obs.copy()
            self.buffer.available_actions[0] = ava.copy()

    @torch.no_grad()
    def collect(self, step):
        if self.all_args.algorithm_name == "happo":
            value_collector=[]
            action_collector=[]
            action_log_prob_collector=[]
            rnn_state_collector=[]
            rnn_state_critic_collector=[]
            for agent_id in range(self.num_agents):
                if self.buffer[agent_id].available_actions is not None:
                    ava = self.buffer[agent_id].available_actions[step]
                else:
                    ava = None
                self.trainer[agent_id].prep_rollout()
                value, action, action_log_prob, rnn_state, rnn_state_critic \
                    = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                    self.buffer[agent_id].obs[step],
                                                    self.buffer[agent_id].rnn_states[step],
                                                    self.buffer[agent_id].rnn_states_critic[step],
                                                    self.buffer[agent_id].masks[step],
                                                    ava)
                value_collector.append(_t2n(value))
                action_collector.append(_t2n(action))
                action_log_prob_collector.append(_t2n(action_log_prob))
                rnn_state_collector.append(_t2n(rnn_state))
                rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            # [self.envs, agents, dim]
            values = np.array(value_collector).transpose(1, 0, 2)
            actions = np.array(action_collector).transpose(1, 0, 2)
            action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
            rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
            rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        elif self.all_args.algorithm_name == "random":
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                np.concatenate(self.buffer.obs[step]),
                                                np.concatenate(self.buffer.rnn_states[step]),
                                                np.concatenate(self.buffer.rnn_states_critic[step]),
                                                np.concatenate(self.buffer.masks[step]),
                                                np.concatenate(self.buffer.available_actions[step]))
            # [self.envs, agents, dim]
            values = np.array(np.split(value, self.n_rollout_threads))
            actions = np.array(np.split((action), self.n_rollout_threads))
            action_log_probs = np.array(np.split((action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split((rnn_state), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split((rnn_state_critic), self.n_rollout_threads))
        elif self.all_args.algorithm_name == "ppo":
            self.trainer.prep_rollout()
            # start_time = time.time()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions((self.buffer.share_obs[step]),
                                                (self.buffer.obs[step]),
                                                (self.buffer.rnn_states[step]),
                                                (self.buffer.rnn_states_critic[step]),
                                                (self.buffer.masks[step]),
                                                (self.buffer.available_actions[step]))
            # [self.envs, agents, dim]
            # end_time = time.time()
            # print("Inference Time: ",end_time - start_time)
            values = np.array((_t2n(value)))
            actions = np.array((_t2n(action)))
            action_log_probs = np.array((_t2n(action_log_prob)))
            rnn_states = np.array((_t2n(rnn_state)))
            # print(rnn_states.shape)
            rnn_states_critic = np.array((_t2n(rnn_state_critic)))
        elif self.all_args.algorithm_name == "mat":
            self.trainer.prep_rollout()
            # start_time = time.time()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                np.concatenate(self.buffer.obs[step]),
                                                np.concatenate(self.buffer.rnn_states[step]),
                                                np.concatenate(self.buffer.rnn_states_critic[step]),
                                                np.concatenate(self.buffer.masks[step]),
                                                np.concatenate(self.buffer.available_actions[step]))
            # end_time = time.time()
            # print("Inference Time: ",end_time - start_time)
            # [self.envs, agents, dim]
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        elif self.all_args.algorithm_name == "momat":
            self.trainer.prep_rollout()
            # start_time = time.time()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                np.concatenate(self.buffer.obs[step]),
                                                np.concatenate(self.buffer.rnn_states[step]),
                                                np.concatenate(self.buffer.rnn_states_critic[step]),
                                                np.concatenate(self.buffer.masks[step]),
                                                np.concatenate(self.buffer.available_actions[step]),
                                                n_objective=self.n_objective)
            # end_time = time.time()
            # print("Inference Time: ",end_time - start_time)
            # [self.envs, agents, dim]
            values = (np.array(np.split(_t2n(value), self.n_rollout_threads)))
            
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        if self.all_args.algorithm_name == "happo": 
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)
        else:
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        if self.all_args.algorithm_name == "ppo":
            masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), 1), dtype=np.float32)
            active_masks = np.ones((self.n_rollout_threads,  1), dtype=np.float32)
            active_masks[dones.reshape(-1) == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), 1), dtype=np.float32)
        else:
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs
        if self.all_args.algorithm_name == "happo":
            bad_masks = np.array([[[0.0] for agent_id in range(self.num_agents)] for info in infos])
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].insert(share_obs[:,agent_id], obs[:,agent_id], rnn_states[:,agent_id],
                        rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                        values[:,agent_id], rewards[:,agent_id], masks[:,agent_id], bad_masks[:,agent_id], 
                        active_masks[:,agent_id], available_actions[:,agent_id])
        else:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                            actions, action_log_probs, values, rewards, masks, None, active_masks,
                            available_actions)

    def log_train(self, train_infos, total_num_steps):
        
        
        if self.all_args.algorithm_name == "happo":
            if MULTI_AGENT_LOG:
                for agent_id in range(self.num_agents):
                    train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
                    for k, v in train_infos[agent_id].items():
                        agent_k = "agent%i/" % agent_id + k
                        self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
            else:
                train_infos[-1]["average_step_rewards"] = np.mean(self.buffer[-1].rewards)
                for k, v in train_infos[-1].items():
                        agent_k = "last_agent/"+k
                        self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        else:
            if self.n_objective > 1:
                for i in range(self.n_objective):
                    train_infos["average_step_objective_{}".format(i)] = np.mean(self.buffer.objectives[:,:,:,i])
                    print("average_step_objective_{} is {}.".format(i,train_infos["average_step_objective_{}".format(i)]))
            else:    
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
            for k, v in train_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps, stride = 2):
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_delays = []
        eval_episode_payments = []
        one_episode_rewards = [0 for _ in range(self.all_args.n_eval_rollout_threads)]
        eval_episode_scores = []
        one_episode_scores = [0 for _ in range(self.all_args.n_eval_rollout_threads)]
        eval_episode_delays = []
        one_episode_delays = [0 for _ in range(self.all_args.n_eval_rollout_threads)]
        eval_episode_payments = []
        one_episode_payments = [0 for _ in range(self.all_args.n_eval_rollout_threads)]

        eval_obs, eval_share_obs, ava = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.all_args.n_eval_rollout_threads, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        inference_times = []
        for i in range(total_num_steps):
            end_time = None
            start_time = time.time() 
            if self.all_args.algorithm_name == "random":
                value, eval_actions, action_log_prob, eval_rnn_states, rnn_state_critic = \
                    self.trainer.policy.get_actions(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            available_actions = np.concatenate(ava))
                eval_actions = np.array(np.split((eval_actions), self.all_args.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split((eval_rnn_states), self.all_args.n_eval_rollout_threads))
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava = self.eval_envs.step(eval_actions)
            elif self.all_args.algorithm_name == "happo":
                eval_actions_collector=[]
                eval_rnn_states_collector=[]
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                                eval_rnn_states[:,agent_id],
                                                eval_masks[:,agent_id],
                                                ava[:,agent_id],
                                                deterministic=True)
                    eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                end_time = time.time()
                eval_actions = np.array(eval_actions_collector).transpose(1,0,2)
                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            elif self.all_args.algorithm_name == "ppo": 
                masks = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32)
                value, action, action_log_prob, rnn_state, rnn_state_critic \
                    = self.trainer.policy.get_actions(eval_share_obs,
                                                eval_obs.reshape(self.n_eval_rollout_threads, -1),
                                                eval_rnn_states,
                                                eval_masks,
                                                masks,
                                                ava,
                                                deterministic = True)
                # [self.envs, agents, dim]
                end_time = time.time()
                # print("Inference Time: ",end_time - start_time)
                eval_actions = np.array((_t2n(action)))
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava = self.eval_envs.step(eval_actions)
            else:
                self.trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(ava),
                                            deterministic=True,
                                            stride = stride)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.all_args.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.all_args.n_eval_rollout_threads))
                end_time = time.time()
                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava = self.eval_envs.step(eval_actions)
                # print(eval_obs[0])
            if end_time is not None:
                inference_times.append(end_time-start_time)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_delay = [t_info[0]["delay"] for t_info in eval_infos]
            eval_payment = [t_info[0]["payment"] for t_info in eval_infos]
            one_episode_delays += np.array(eval_delay)
            one_episode_payments += np.array(eval_payment)

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                        dtype=np.float32)

            for eval_i in range(self.all_args.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    
                    one_episode_rewards[eval_i] = 0

                    eval_episode_delays.append(one_episode_delays[eval_i])
                    one_episode_delays[eval_i] = 0
                    
                    eval_episode_payments.append(one_episode_payments[eval_i])
                    one_episode_payments[eval_i] = 0
                    
                    # eval_episode_scores.append(one_episode_scores[eval_i])
                    # one_episode_scores[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = '/eval_average_episode_rewards'
                key_max = '/eval_max_episode_rewards'
                key_delays = '/eval_average_episode_delays'
                key_payments = '/eval_average_episode_payments'
                eval_env_infos = {key_average: eval_episode_rewards,
                                key_max: [np.max(eval_episode_rewards)],
                                key_delays: eval_episode_delays,
                                key_payments:eval_episode_payments}
                # self.log_env(eval_env_infos, total_num_steps)
            if i %100 == 0:
                print("eval average episode rewards: {}, delays: {}, payments: {}."
                    .format(np.mean(eval_episode_rewards), np.mean(eval_episode_delays), np.mean(eval_episode_payments)))
                print("Inference time: ",np.mean(inference_times))
                # break
        
        return np.mean(eval_episode_rewards) ,np.mean(eval_episode_delays), np.mean(eval_episode_payments),np.mean(inference_times)

# %%
