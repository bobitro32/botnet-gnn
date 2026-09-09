import torch
torch.manual_seed(42)

from data_loader import load_ctu13
from graph_builder import build_graph
from feature_extractor import extract_features
from model import _split_data, _train_model, _evaluate_model
from gnn_dataset import build_pyg_data
from model_gnn import BotnetGNN, _create_masks, train_gnn, evaluate_gnn
from visualizer import visualize_graph

FILEPATH = '/Users/bobitro/Desktop/Desktop/botnet-gnn/CTU-13-Dataset/1/capture20110810.binetflow'

df = load_ctu13(FILEPATH)
df_small = df.sample(n=100_000, random_state=42)

G_small = build_graph(df_small)
features_df_small = extract_features(G_small)

# --- Random Forest ---
print("\n========== RANDOM FOREST ==========")
X_train, X_test, y_train, y_test = _split_data(features_df_small)
rf_model = _train_model(X_train, y_train)
y_pred_rf = _evaluate_model(rf_model, X_test, y_test)

# --- GCN ---
print("\n========== GCN ==========")
pyg_data = build_pyg_data(G_small, features_df_small)
train_mask, test_mask = _create_masks(pyg_data.num_nodes)

gnn_model = BotnetGNN(in_channels=4, hidden_channels=32, out_channels=2)
gnn_model = train_gnn(gnn_model, pyg_data, train_mask, epochs=100)
y_pred_gnn = evaluate_gnn(gnn_model, pyg_data, test_mask)

visualize_graph(G_small, output_path='botnet_graph.html')