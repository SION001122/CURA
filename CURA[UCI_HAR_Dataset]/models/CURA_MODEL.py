import torch
import torch.nn as nn
import torch.nn.functional as F

class CURACORE(nn.Module):
    def __init__(self, hidden_dim): 
        super().__init__()
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_fc(x))
        residual = self.residual_fc(x)
        x = gate * residual + residual
        x = F.relu(self.relu_linear(x))
        x = self.conv(x.unsqueeze(1)).squeeze(1)
        return x


    
class CURASTACK(nn.Module):
    def __init__(self, seq_len, hidden_dim, output_dim=1, num_cores=1):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, hidden_dim)  
        self.cores = nn.Sequential(*[
            CURACORE(hidden_dim) for _ in range(num_cores)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        x = self.input_proj(x)
        x = self.cores(x)
        return self.output(x).squeeze(-1)
