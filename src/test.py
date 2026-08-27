from data_loader import load_ctu13
from graph_builder import build_graph
from feature_extractor import extract_features
from model import _split_data, _train_model, _evaluate_model
from gnn_dataset import build_pyg_data
from model_gnn import BotnetGNN, _create_masks, train_gnn
from model_gnn import evaluate_gnn
import torch
torch.manual_seed(42)
df = load_ctu13('/Users/bobitro/Desktop/Desktop/GraduationProject/botnet-gnn/data/CTU-13-Dataset/1/capture20110810.binetflow')
'''G = build_graph(df)
features_df = extract_features(G)
X_train, X_test, y_train, y_test = _split_data(features_df)
model = _train_model(X_train, y_train)
y_pred = _evaluate_model(model, X_test, y_test)
pyg_data = build_pyg_data(G, features_df)'''

# Временно, само за тест
df_small = df.sample(n=100_000, random_state=42)
G_small = build_graph(df_small)
features_df_small = extract_features(G_small)
pyg_data = build_pyg_data(G_small, features_df_small)

print(f"\nx shape: {pyg_data.x.shape}")
print(f"edge_index shape: {pyg_data.edge_index.shape}")
print(f"y shape: {pyg_data.y.shape}")

gnn_model = BotnetGNN(in_channels=4, hidden_channels=32, out_channels=2)

print(gnn_model)

# Тестово forward pass с нашите PyG данни
output = gnn_model(pyg_data.x, pyg_data.edge_index)
print(f"\nOutput shape: {output.shape}")

train_mask, test_mask = _create_masks(pyg_data.num_nodes)

gnn_model = BotnetGNN(in_channels=4, hidden_channels=32, out_channels=2)
gnn_model = train_gnn(gnn_model, pyg_data, train_mask, epochs=100)
y_pred_gnn = evaluate_gnn(gnn_model, pyg_data, test_mask)