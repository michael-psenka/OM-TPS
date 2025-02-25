import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import radius_graph

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        # self.cutoff = 5.0
        self.max_num_neighbors = 12
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, return_energy=False):
        B, N, D = x.shape
        batch = torch.arange(B).repeat_interleave(N).to(x.device)
        x = x.view(-1, D)
        edge_index = torch.stack([torch.arange(N-1), torch.arange(1, N)], dim=0).to(x.device)
        edge_index = edge_index.repeat(1, B) + N*torch.arange(B).repeat_interleave(N-1).to(x.device)
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(self.lin(x), batch)  # [batch_size, hidden_channels]
        
        
        return x