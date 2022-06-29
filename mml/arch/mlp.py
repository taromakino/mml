import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, target_dim):
        super(MLP, self).__init__()
        module_list = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(prev_dim, hidden_dim))
            module_list.append(nn.ReLU())
            prev_dim = hidden_dim
        module_list.append(nn.Linear(hidden_dims[-1], target_dim))
        self.module_list = nn.Sequential(*module_list) # Don't call this self.modules

    def forward(self, x):
        return self.module_list(x)