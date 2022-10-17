from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

SavedProb = namedtuple('SavedProb', field_names=['log_prob', 'v_val', 'entropy'])

class PackmanNet(nn.Module):
    def __init__(self, outputs=4):
        super(PackmanNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 1), stride=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.hidden = nn.Linear(1920, 64)
        self.action_head = nn.Linear(64, outputs)
        self.value_head = nn.Linear(64, 1)
        self.device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = x.to(self.device)
        x = x.view(1, 1, *x.shape).float()
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = x.flatten()
        x = F.relu(self.hidden(x))
        action_probs = F.softmax(self.action_head(x))
        value = self.value_head(x)

        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)

        return action.item(), SavedProb(action_log_prob, value, action_distribution.entropy())

    def select_greedy_action(self, x):
        x = x.to(self.device)
        x = x.view(1, 1, *x.shape).float()
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = x.flatten()
        x = F.relu(self.hidden(x))
        action_probs = F.softmax(self.action_head(x))
        return action_probs.argmax().item()
