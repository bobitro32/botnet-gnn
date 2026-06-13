import numpy as np
import pandas as pd 

#constants 
USECOLS = [
    'SrcAddr', 'DstAddr', 'Proto', 'Dur',
    'TotBytes', 'TotPkts', 'SrcBytes', 'State', 'Label'
]

NUMERIC_COLS = ['Dur', 'TotBytes', 'TotPkts', 'SrcBytes']
def _read_csv(filepath: str) -> pd.DataFrame:
    """read the file and clean the names of the columns"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"Заредени редове {len(df):,}")
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Saves only the neccessary columns
    return df[USECOLS]

def _drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Removes rows with empty values(NaN)
    before = len(df)
    df = df.dropna()
    return df

def _fix_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    # Converts the numeric_cols into float
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    #Adds label "Is_botnet": 1 = Botnet, 0 = Normal
    df['is_botnet'] = df['Label'].str.contains('Botnet').astype(int)
    return df

def load_ctu13(filepath: str) -> pd.DataFrame:
    
    
    df = _read_csv(filepath)
    df = _select_columns(df)
    df = _drop_missing(df)
    df = _fix_numeric_types(df)
    df = _drop_missing(df)        
    df = _add_label(df)

    return df