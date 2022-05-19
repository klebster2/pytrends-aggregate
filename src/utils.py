import pandas as pd
import numpy as np

def drop_rows_ispartial_false(df):
    if "isPartial" in df.columns:
        return df[df.isPartial=="False"]
    else:
        return df

def drop_col_ispartial(df):
    """
    drops index timeframes for data where column "isPartial" is "False"
    also drops column "isPartial"
    """
    if "isPartial" in df.columns:
        return df.drop(
            "isPartial",
            axis=1
        )
    else:
        return df

def last_index_nonzero(df) -> pd.Index:
    return (df[(df!=0).any(axis=1)].iloc[-1]!=0)

def get_cols_where_last_index_nonzero(df):
    return df.columns[last_index_nonzero(df)]

def get_pivot_col(df) -> str:
    """
    df.max() will get the maximum columns
    then the sort descending and index[0] will get the maximum value column
    """
    return df.max().sort_values(ascending=False).index[0]

def partition_cols(df, cutoff_pct:float) -> tuple:
    above_cutoff = (df.nunique()>cutoff_pct)
    return (
        df.nunique()[above_cutoff],
        df.nunique()[~above_cutoff],
    )

def get_data_granularity(self) -> float:
    return 100 * ( self.df.nunique().sum() / np.dot(*self.df.shape) )

def get_new_group(df, kw_list, too_little_data) -> list:
    """
    This is bound to the maximum number of keywords you can sent using the
    API (5)
    """
    if not df.empty:
        group = [df.columns[-1]]
        group.extend(kw_list[:4])
    else:
        group = kw_list[:5]
    if len(too_little_data)>0 and too_little_data[-1] == group[0]:
        # better reaction when there's a removal
        group[0] = df.columns[-1]
    return group
