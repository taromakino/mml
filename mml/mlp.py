import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden, output_dim):
        super().__init__()
        module_list = []
        module_list.append(nn.Linear(input_dim, hidden_dim))
        module_list.append(nn.ReLU())
        module_list.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_hidden):
            module_list.append(nn.Linear(hidden_dim, hidden_dim))
            module_list.append(nn.ReLU())
            module_list.append(nn.BatchNorm1d(hidden_dim))
        module_list.append(nn.Linear(hidden_dim, output_dim))
        self.module_list = nn.Sequential(*module_list) # Don't call this self.modules

    def forward(self, x):
        return self.module_list(x)