from data_loader import load_ctu13
from graph_builder import build_graph
from feature_extractor import extract_features
from model import _split_data
df = load_ctu13('/Users/bobitro/Desktop/Desktop/GraduationProject/botnet-gnn/data/CTU-13-Dataset/1/capture20110810.binetflow')
df_small = df.sample(n=100_000, random_state=42)


G = build_graph(df_small)

features_df = extract_features(G)
print(features_df.head(10))
print(f"\nБотнет възли: {features_df['is_botnet'].sum()}")

X_train, X_test, y_train, y_test = _split_data(features_df)

print(f"\nBotnet в train: {y_train.sum()}")
print(f"Botnet в test:  {y_test.sum()}")