import torch
import numpy as np
from template_policy import TemplatePolicy
from environment import Environment
import torch.nn.functional as F


class ActorCritic(TemplatePolicy):
    def __init__(
        self,
        env: Environment,
        model: torch.nn.Module,
        lr,
        gamma,
        step_size:int,
        baseline:bool,
        entropy,
        name,
        M: int = 1,
        budget: int = 5000,
    ):
        super().__init__(env, model, lr, gamma, name, M)
        self.step_size = step_size
        self.baseline = baseline
        self.entropy_eta, self.eta_decrease_factor = entropy
        self.budget = budget

    def update_target_weights(self, epoch):
        pass
    
    def optimize_per_step(self, t, episode_return, selected_action_probs):
      pass

    def optimize_per_epoch(self, m_epoch_experiences, m_selected_action_probs):
      # Calculate episode returns
      policy_losses = []
      value_losses = []
      for epoch_experiences, selected_action_probs in zip(m_epoch_experiences, m_selected_action_probs):
        bs_rewards = []
        for t in range(len(epoch_experiences)):
          episode_return = 0
          bootstrap_length =  min(len(epoch_experiences), t+self.step_size)
          for i in range(t, bootstrap_length):
            episode_return += self.gamma**(i-t) * epoch_experiences[i].reward
          episode_return += selected_action_probs[bootstrap_length-1].v_val.item()
          bs_rewards.append(episode_return)
          
        eps = np.finfo(np.float32).eps.item()
        bs_rewards = (bs_rewards - np.mean(bs_rewards)) / (np.std(bs_rewards) + eps)

        
        n = min(self.step_size, len(bs_rewards))
        for (log_prob, v_val, entropy), ep_return, _ in zip(selected_action_probs, bs_rewards, range(n)):
          not_loss = ep_return
          if self.baseline:
            not_loss -= v_val.item() 
            
          policy_losses.append(-log_prob * not_loss + self.entropy_eta * entropy)
          value_losses.append(
                          F.smooth_l1_loss(
                            v_val, 
                            torch.tensor([ep_return])
                          )
                        )
        
      self.optimizer.zero_grad()
      loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
      loss.backward()
      # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
      self.optimizer.step()
      
      self.entropy_eta = self.entropy_eta * self.eta_decrease_factor