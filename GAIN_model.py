import torch
from torch import nn
import numpy as np

torch.cuda.set_device(0)
# 算法参数设置
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.1
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, input_size),
            nn.Sigmoid()
        )

    def forward(self, new_x, h):
        inputs = torch.cat(dim=1, tensors=[new_x, h])
        D_prob = self.d(inputs)
        return D_prob


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(Generator, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, input_size),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, new_x, m):
        # Mask + Data Concatenate
        inputs = torch.cat(dim=1, tensors=[new_x, m])
        G_prob = self.g(inputs)
        return G_prob


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        # nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.0)

