import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, confusion_matrix

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
    """Обучава GNN модела с тегла за class imbalance."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Изчисли тегла — по-рядкият клас получава по-висока тежест
    y_train = data.y[train_mask]
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2  # нормализирай

    print(f"Class weights: Normal={class_weights[0]:.4f}, Botnet={class_weights[1]:.4f}")

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weights)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}  Loss: {loss.item():.4f}")

    return model

def evaluate_gnn(model: BotnetGNN, data, test_mask):
    """Оценява GNN модела върху test данните."""
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    y_true = data.y[test_mask].numpy()
    y_pred = pred[test_mask].numpy()

    print("\n=== GNN Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Botnet']))

    print("=== GNN Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    return y_pred