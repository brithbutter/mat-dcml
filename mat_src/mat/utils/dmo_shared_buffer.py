import torch
import numpy as np
import torch.nn.functional as F
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

def _cal_weighted_gae(gae,objective_coefficients):
    weighted_gae=None
    n_thread = gae.shape[0]
    for i in range(n_thread):
        if weighted_gae is None:
            weighted_gae = np.expand_dims(gae[i] * objective_coefficients[i],axis=0)
            
        else:
            weighted_gae = np.vstack((weighted_gae,np.expand_dims(gae[i] * objective_coefficients[i],axis=0)))
    return weighted_gae

class DMOSharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, env_name,n_objective = 2):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.use_advantage_norm = args.use_advantage_norm
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.env_name = env_name
        self.n_objective = n_objective
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.objective_coefficients = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.n_objective), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)


        # Reward Buffer
        self.objectives = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, self.n_objective), dtype=np.float32)
        self.objective_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents,self.n_objective), dtype=np.float32)
        self.returns = np.zeros_like(self.objective_preds)
        self.advantages = np.zeros((self.episode_length, self.n_rollout_threads, num_agents,self.n_objective), dtype=np.float32)
        
        
        

        # Action Buffer
        if act_space.__class__.__name__ == 'Discrete' or act_space.__class__.__name__ == 'Action_Space' or act_space.__class__.__name__ == 'Available_Continous_Space':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads
                                            , num_agents, act_space.n),dtype=np.float32)
        else:
            self.available_actions = None

        act_shape,act_prob_shape = get_shape_from_act_space(act_space)
        if act_prob_shape is None:
            act_prob_shape = act_shape
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_prob_shape), dtype=np.float32)
        
        
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, self.n_objective), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0
    
    def save(self,file_name = "mo_shared_buffer.npy"):
        with open(file_name,"wb") as writter:
            np.save(writter,self.share_obs)
            np.save(writter,self.obs)
            np.save(writter,self.rnn_states)
            np.save(writter,self.rnn_states_critic)
            np.save(writter,self.actions)
            np.save(writter,self.action_log_probs)
            np.save(writter,self.masks)
            np.save(writter,self.available_actions)
            for single_obj_value_preds in self.objective_preds:
                np.save(writter,single_obj_value_preds)
            for single_obj_rewards in self.objectives:
                np.save(writter,single_obj_rewards)

    def load(self,file_name = "mo_shared_buffer.npy"):
        with open(file_name,"wb") as reader:
            self.share_obs = np.load(reader)
            self.obs = np.load(reader)
            self.rnn_states = np.load(reader)
            self.rnn_states_critic = np.load(reader)
            self.actions = np.load(reader)
            self.action_log_probs = np.load(reader)
            self.masks = np.load(reader)
            self.available_actions = np.load(reader)
            self.objective_preds = []
            for i in range(self.n_objective):
                self.objective_preds.append(np.load(reader))
            self.objectives = []
            for i in range(self.n_objective):
                self.objectives.append(np.load(reader))
            
    
    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,objective_preds, objectives, masks, bad_masks=None, active_masks=None, available_actions=None,objective_coefficients=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param objective_preds: (np.ndarray) objective function prediction at each step.
        :param objectives: (np.ndarray) objective collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.objective_preds[self.step] = objective_preds.copy()
        self.objectives[self.step] = objectives.copy()
        self.masks[self.step + 1] = masks.copy()
        if objective_coefficients is not None:
            self.objective_coefficients[self.step] = objective_coefficients.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,objective_preds, objectives, masks, bad_masks=None, active_masks=None, available_actions=None,objective_coefficients=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param objective_preds: (np.ndarray) objective value function prediction at each step.
        :param objectives: (np.ndarray) objective collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        for i in range(self.n_objective):
            self.objective_preds[i][self.step] = objective_preds[i].copy()
            self.objectives[i][self.step] = objectives[i].copy()
        self.masks[self.step + 1] = masks.copy()
        if objective_coefficients is not None:
            self.objective_coefficients[self.step] = objective_coefficients.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_objective_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.objective_preds[-1] = next_objective_value
        gae = 0
        # Check Normalize
        for step in reversed(range(self.objectives.shape[0])):
            if self._use_popart or self._use_valuenorm:
                if self.use_advantage_norm:
                    # Use Normalized Advantage Value
                    delta = value_normalizer.normalize(self.objectives[step]).cpu().numpy() \
                        + (self.gamma * self.objective_preds[step + 1] * self.masks[step + 1])\
                        - self.objective_preds[step]
                else:
                    # Use Original Advantage Value
                    delta = self.objectives[step] + self.gamma * value_normalizer.denormalize(
                    self.objective_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.objective_preds[step])
                    
                    
                
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                
                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.objectives.shape[0] - 1:
                    gae = 0
                # Here is a implementation for weighted gae.
                # if self.objective_coefficients is not None:
                #     obj_coefficients = self.objective_coefficients[step]
                #     weighted_gae = _cal_weighted_gae(gae=gae,objective_coefficients=obj_coefficients)
                #     self.advantages[step] = weighted_gae
                # else:
                #     self.advantages[step] = gae
                # Here is a implementation for normal gae.
                self.advantages[step] = gae
                if value_normalizer.updated:
                    if self.use_advantage_norm:
                        # Original Return with Normalize GAE
                        self.returns[step] = value_normalizer.denormalize(gae +self.objective_preds[step])
                    else:
                        # Original Return with Original GAE
                        self.returns[step] = gae + value_normalizer.denormalize(self.objective_preds[step])           
                else:
                    self.returns[step] = self.objectives[step]
            else:
                delta = self.objectives[step] + self.gamma * self.objective_preds[step + 1] * \
                        self.masks[step + 1] - self.objective_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.objectives.shape[0] - 1:
                    gae = 0

                # Here is a implementation for weighted gae. 
                # Due the effect issue, we comments this implementation
                # if self.objective_coefficients is not None:
                #     obj_coefficients = self.objective_coefficients[step]
                #     weighted_gae = _cal_weighted_gae(gae=gae,objective_coefficients=obj_coefficients)
                #     self.advantages[step] = weighted_gae
                # else:
                #     self.advantages[step] = gae
                # Here is a implementation for normal gae.
                self.advantages[step] = gae
                self.returns[step] = gae + self.objective_preds[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.objectives.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        objective_value_preds = self.objective_preds[:-1].reshape(-1, *self.objective_preds.shape[2:])
        objective_value_preds = objective_value_preds[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            objective_value_preds_batch = objective_value_preds[indices].reshape(-1, *objective_value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  objective_value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
