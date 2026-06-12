import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def load_ctu13(filepath):
    print(f"Зареждане: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"Заредени редове {len(df):,}")
    return df