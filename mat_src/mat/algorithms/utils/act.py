from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.action_type = action_space.__class__.__name__
        if action_space.__class__.__name__ == "Discrete": 
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Action_Space":
            self.action_out_type = None
            if action_space.mixed:
                self.action_out_type = "MIX_ACTION"
                self.mixed_action = True
                self.semi_index = action_space.semi_index
                self.log_std = torch.nn.Parameter(torch.ones(-self.semi_index))
                # self.countious_action_out = torch.distributions.Normal(inputs_dim, -self.semi_index, use_orthogonal, gain, args)
                self.multi_discrete = True
                self.action_dims = action_space.high - action_space.low
                self.action_range = action_space.n
            if action_space.extra:
                self.action_out_type = "CONTINUOUS_ACTION"
                action_dim = 1
                self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
            elif action_space.multi_discrete:
                self.action_out_type = "MULTI_DISCRETE"
                self.multi_discrete = True
                action_dims = action_space.high - action_space.low
                self.action_outs = []
                for action_dim in action_dims:
                    self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
                self.action_outs = nn.ModuleList(self.action_outs)
            else:
                self.action_out_type = "DISCRETE"
                action_dim = action_space.n
                self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain, args),
                                              Categorical(inputs_dim, discrete_dim, use_orthogonal, gain)])
    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action :
            actions = []
            action_log_probs = []
            for i in range(self.action_dims):
                logit = x[:,i*self.action_range:(i+1)*self.action_range]
                logit[available_actions[:,i,:]== 0] = -1e10 
                distri = torch.distributions.Categorical(logits=logit)
                action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
                action_log_prob = distri.log_prob(action)
                actions.append(action.view(-1,1).float())
                action_log_probs.append(action_log_prob.view(-1,1))
            # continous_action_logits = self.countious_action_out(x, available_actions)
            continous_action_mean = x[:,self.semi_index:]
            continous_action_std = torch.sigmoid(self.log_std)*0.5
            distri = torch.distributions.Normal(continous_action_mean,continous_action_std)
            continous_action = continous_action_mean if deterministic else distri.sample() 
            continous_action_log_prob = distri.log_prob(continous_action)
            actions.append(continous_action.view(-1,-self.semi_index).float())
            action_log_probs.append(continous_action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            # action_log_probs = torch.cat(action_log_probs, -1).min(dim=-1,keepdim=True).values

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((100, 1), -1)
            x[:,:self.semi_index][available_actions.view(available_actions.shape[0],-1)[:,:self.semi_index*self.action_range] == 0] = -1e10 
            a = a.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            final_dist_entropy = []
            for i in range((self.action_dims)):
                act = a[:,i]
                logit = x[:,i*self.action_range:(i+1)*self.action_range]
                
                distri = torch.distributions.Categorical(logits=logit)
                action_log_probs.append(distri.log_prob(act).view(-1,1))
                if active_masks is not None:
                    if len(distri.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((distri.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((distri.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(distri.entropy().mean())
            final_dist_entropy.append(torch.mean(torch.tensor(dist_entropy)))
            continous_action_mean =  x[:,self.semi_index:]
            continous_action_std = torch.sigmoid(self.log_std)*0.5
            distri = torch.distributions.Normal(continous_action_mean,continous_action_std)
            action_log_probs.append(distri.log_prob(b))
            
            ent = distri.entropy()
            if active_masks is not None:
                if len(distri.entropy().shape) == len(active_masks.shape):
                    dist_entropy.append((distri.entropy() * active_masks).sum()/active_masks.sum()) 
                else:
                    dist_entropy.append((distri.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
            else:
                dist_entropy.append(distri.entropy().mean())
            final_dist_entropy.append(dist_entropy[-1])
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            # action_log_probs = torch.cat(action_log_probs, -1).min(dim=-1,keepdim=True).values
            dist_entropy = final_dist_entropy[0] / 0.98 + final_dist_entropy[1] / 0.98 

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) 
            dist_entropy = torch.tensor(dist_entropy).mean()
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                if self.action_type=="Discrete" or (self.action_type== "Action_Space" and self.action_out_type == "DISCRETE"):
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            mu_collector = []
            std_collector = []
            probs_collector = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                mu = action_logit.mean
                std = action_logit.stddev
                action_log_probs.append(action_logit.log_probs(act))
                mu_collector.append(mu)
                std_collector.append(std)
                probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector,-1)
            action_std = torch.cat(std_collector,-1)
            all_probs = torch.cat(probs_collector,-1)
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type=="Discrete":
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type=="Discrete":
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
