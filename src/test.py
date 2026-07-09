from data_loader import load_ctu13
from graph_builder import build_graph
from feature_extractor import extract_features
from model import _split_data, _train_model, _evaluate_model


df = load_ctu13('/Users/bobitro/Desktop/Desktop/GraduationProject/botnet-gnn/data/CTU-13-Dataset/1/capture20110810.binetflow')
G = build_graph(df)
features_df = extract_features(G)
X_train, X_test, y_train, y_test = _split_data(features_df)
model = _train_model(X_train, y_train)
y_pred = _evaluate_model(model, X_test, y_test)

