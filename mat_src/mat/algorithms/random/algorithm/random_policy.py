#%%
import torch
import torch
import numpy as np
import random 
from mat.utils.util import update_linear_schedule
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from mat.algorithms.utils.util import check
#%%
class Random_Policy:
    """
    HAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for HAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,num_agents, device=torch.device("cpu")):
        self.device = device
        self.act_space = act_space
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        semi_index= None
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        elif act_space.__class__.__name__ == 'Action_Space':
            if act_space.semi_index != 0:
                self.action_type = 'Semi_Discrete'
                semi_index = act_space.semi_index
            elif act_space.multi_discrete:
                self.action_type = 'Multi_Discrete'
            else:
                self.action_type = 'Discrete'
        else:
            self.action_type = 'Discrete'
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == 'Discrete' or self.action_type == 'Semi_Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        self.num_agents = num_agents

    def lr_decay(self, episode, episodes):
        pass

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,mask=False, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)
        values = []
        actions = []
        action_log_probs = []
        for j in range(available_actions.shape[0]):
            thread_actions = []
            thread_values = []
            thread_action_log_probs = []
            for i in range(self.num_agents):
                if i> self.act_space.semi_index + self.num_agents:
                    action = random.uniform(0,1)
                else:
                    # print(available_actions[i])
                    index = random.randint(0,sum(available_actions[j,i])-1)
                    action = np.argwhere(np.array(available_actions[j,i])==1)[index,0]
                # thread_actions.append(action)
                # thread_values.append(0)
                # thread_action_log_probs.append(0) 
                actions.append(action)
                values.append(0)
                action_log_probs.append(0) 
            # actions.append(thread_actions)
            # values.append(thread_values)
            # action_log_probs.append(thread_action_log_probs)
        values = np.array(values).reshape(-1,1)
        actions = np.array(actions).reshape(-1,1)
        action_log_probs = np.array(action_log_probs).reshape(-1,1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values = [0]*self.num_agents
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        action_log_probs = None
        dist_entropy = None

        values = None
        return values, action_log_probs, dist_entropy


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
    def eval(self):
        pass
# %%
