import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def _split_data(features_df: pd.DataFrame):
    """Разделя данните на train и test."""
    X = features_df[['in_degree', 'out_degree', 'pagerank', 'betweenness']]
    y = features_df['is_botnet']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% за тест, 80% за обучение
        random_state=42,    # за възпроизводимост
        stratify=y           # запази пропорцията botnet/normal и в двете части
    )

    print(f"Train: {len(X_train):,} реда")
    print(f"Test:  {len(X_test):,} реда")

    return X_train, X_test, y_train, y_test