import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BotnetGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def _create_masks(num_nodes: int, train_ratio: float = 0.8):
    '''Създава train/test маски за GNN обучение.'''
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    return train_mask, test_mask

def train_gnn(model: BotnetGNN, data, train_mask, epochs: int = 100, lr: float = 0.01):
    """Обучава GNN модела."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}  Loss: {loss.item():.4f}")

    return model