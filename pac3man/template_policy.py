import torch
from tqdm import tqdm
from dataclasses import dataclass

from rl_environment import Environment, Experience

@dataclass
class TrainResults:
  rewards_per_epoch: list

class TemplatePolicy():
  def __init__(
    self,
    env: Environment,
    model: torch.nn.Module,
    lr,
    gamma,
    name,
    M=1
  ):
    self.episode_length_threshold = 200
    self.negative_reward_threshold = None

    self.env = env
    self.device = 'cpu'# torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model.to(device=self.device)
    self.lr = lr
    self.gamma = gamma
    self.M = M
    self.name = name
    self.loss_function = torch.nn.MSELoss(reduction='mean')
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # , betas=(0.9, 0.999)
    self.best_epoch = -1
    # self.restore_optimal_weights()

  def reset(self):
    self.model = self.model.reset()
    self.model.to(device=self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.best_epoch = -1
    self.store_weights()

  def evaluate(self, demo_mode=False, validate=False):
    if demo_mode:
      self.env.enable_demo_mode()

    done = False
    rewards = 0
    ep_length = 0
    self.env.reset()
    while not done:
      ep_length += 1
      action = self.model.select_greedy_action(torch.tensor(self.env.get_state(), device=self.device))
      # action = argmax(q_values.cpu().detach().numpy())
      experience = self.env.step(action, evaluate_mode=True)
      done = experience.done
      rewards += experience.reward

      if ep_length > 200:
        break


    if validate==False:
      print(rewards)
    return rewards

  def train(self, max_epochs):
    # self.prepare_training()
    epoch_rewards = []
    max_epoch_reward = 0
    time_steps = 0
    m_selected_action_probs = []
    m_epoch_experiences = []
    prev_mean = 0
    for epoch in tqdm(range(max_epochs)):
      done, epoch_reward, ep_length = False, 0, 0
      self.env.reset()
      # self.update_target_weights(epoch)
      selected_action_probs = []
      epoch_experiences = []

      while not done:

        if self.episode_length_threshold is not None and ep_length > self.episode_length_threshold:
          break
        if self.negative_reward_threshold is not None and epoch_reward <= self.negative_reward_threshold:
          break

        time_steps += 1
        ep_length += 1
        if ep_length % 10000 == 0:
          print(self.env.state)

        actions, prob = self.model(torch.tensor(self.env.get_state(), device=self.device))

        # action = self.exploration.select_action(q_values.clone().cpu().detach().numpy(), epoch)
        experience: Experience = self.env.step(actions)
        done = experience.done
        selected_action_probs.append(prob)
        epoch_experiences.append(experience)
        epoch_reward += experience.reward
      
      for t in range(ep_length):
        self.optimize_per_step(t, epoch_experiences, selected_action_probs[t])

      m_epoch_experiences.append(epoch_experiences)
      m_selected_action_probs.append(selected_action_probs)
      
      epoch_rewards.append(epoch_reward)
      if epoch_reward >= max_epoch_reward:
        reward = 0
        for _ in range(10):
          reward += self.evaluate(validate=True)
        if reward//10 >= prev_mean:
          # print('stored')
          prev_mean = reward // 10
          self.store_weights()
          self.best_epoch = epoch
          max_epoch_reward = epoch_reward

      if len(m_epoch_experiences) == self.M:
        self.optimize_per_epoch(m_epoch_experiences, m_selected_action_probs)
        m_epoch_experiences, m_selected_action_probs = [], []


      if epoch % 100 == 0:
        print(f'Run {self.name} episode {epoch}\treward: {epoch_reward}\tep_length: {ep_length}')

    self.store_weights()
    # self.restore_optimal_weights()
    reward = 0
          
    
    # print(f'Best epoch: {self.best_epoch}')
    return TrainResults(
      rewards_per_epoch=epoch_rewards,
    )

  def prepare_training(self):
    pass

  def update_target_weights(self, epoch):
    pass

  def optimize_per_step(self, t, episode_return, selected_action_probs):
    pass

  def optimize_per_epoch(self, ep_returns, selected_action_probs):
    pass

  def store_weights(self):
    torch.save(self.model.state_dict(), f'./pacman_weights_{self.name}.h5')
    # self.optimal_weights = self.model.state_dict()

  def restore_optimal_weights(self):
    self.model.load_state_dict(torch.load(f'./pacman_weights_{self.name}.h5'))

    # self.model.load_state_dict(self.optimal_weights)
