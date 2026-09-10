import warnings

import numpy as np
import pandas as pd


def sanitise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-Inf with NaN, then fill numeric NaN values with zero."""
    df = df.copy()
    num = df.select_dtypes(include=np.number).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def safe_gradient(styler, cmap="Blues", subset=None):
    """Apply a pandas background gradient only where numeric data has a real range."""
    df_data = styler.data.copy()
    num_all = df_data.select_dtypes(include=np.number).columns
    df_data[num_all] = df_data[num_all].replace([np.inf, -np.inf], np.nan)

    candidates = (
        [c for c in subset if c in df_data.columns]
        if subset is not None
        else [c for c in df_data.columns if pd.api.types.is_numeric_dtype(df_data[c])]
    )

    cols = []
    for c in candidates:
        s = df_data[c].dropna()
        if len(s) > 0 and (s.max() - s.min()) > 1e-12:
            cols.append(c)

    if cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return df_data.style.background_gradient(cmap=cmap, subset=cols)
    return styler
