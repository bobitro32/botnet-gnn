from data_loader import load_ctu13
from graph_builder import build_graph
from feature_extractor import _degree_features

df = load_ctu13('/Users/bobitro/Desktop/Desktop/GraduationProject/botnet-gnn/data/CTU-13-Dataset/1/capture20110810.binetflow')
df_small = df.sample(n=100_000, random_state=42)
G = build_graph(df_small)

features = _degree_features(G)

# Покажи топ 5 IP-та по out_degree
top5 = sorted(features.items(), key=lambda x: x[1]['out_degree'], reverse=True)[:5]
for ip, feat in top5:
    print(f"{ip:20s}  in={feat['in_degree']:5d}  out={feat['out_degree']:5d}")