import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

def load_cic_ids(file):

    if hasattr(file, "getvalue"):
        bytes_data = file.getvalue()
        df = pd.read_csv(io.BytesIO(bytes_data))
    else:
        df = pd.read_csv(file)

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    possible_labels = ["Label", "Class", "Attack", "Category", "TrafficType"]

    label_column = None
    for col in possible_labels:
        if col in df.columns:
            label_column = col
            break

    if label_column is None:
        raise ValueError(
            f"No label column found. Available columns: {list(df.columns)}"
        )

    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # 🔥 Convert to Binary Classification
    y = y.apply(lambda x: "BENIGN" if str(x).upper() == "BENIGN" else "ATTACK")

    X = X.select_dtypes(include=[np.number])
    X = X.clip(lower=-1e10, upper=1e10)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.3, random_state=42)