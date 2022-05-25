from .timeframes.timeframes import Timeframes
from .kw_list import KwList

from .utils import (
    drop_col_ispartial,
    drop_rows_ispartial_false,
    last_index_nonzero,
    get_cols_where_last_index_nonzero,
    get_pivot_col,
    partition_cols,
    get_data_granularity,
    get_new_group,
)

from typing import Iterator, Tuple

import math

from pytrends.request import TrendReq
import pytrends

import pandas as pd

import copy

class PyTrendTableManager(TrendReq):
    def __init__(
        self,
        timeframes: Timeframes,
        cutoff_pct:float = 10.0,
        sleep:float = 60.0,
        verbose:bool = True,
    ) -> None:

        super().__init__()

        self.timeframes=timeframes
        self.cutoff_pct=cutoff_pct
        self.sleep=sleep
        self.batch_counter=0
        self.verbose=verbose

    def _pivot_ok(self, pivot: float) -> bool:
        if not pivot:
            print(f"Error. Pivot is zero or None") if self.verbose else None
            return False
        elif math.isnan(pivot):
            print(f"Error. Pivot is nan") if self.verbose else None

            return False
        elif math.isinf(pivot):
            print(f"Error. Pivot is inf") if self.verbose else None

            return False
        else:
            print("Pivot ok") if self.verbose else None
            return True

    @staticmethod
    def _get_next_pivots(cols: pd.Series) -> list:
        return cols.sort_values(ascending=False).index.tolist()

    def build_payload_get_interest_over_intervals(self, kw_list):
        df = pd.DataFrame([])
        error = False

        for timeframe in self.timeframes:
            df2 = self.build_payload_get_interest(
                kw_list=kw_list,
                timeframe=str(timeframe),
            )

            if not df.empty:
                pivot_col = get_pivot_col(
                    df[get_cols_where_last_index_nonzero(df)],
                )

                assert df.iloc[-1,:].name == df2.iloc[0,:].name

                error, pivot = self.get_pivot(
                    ref_df=df.loc[df.iloc[-1,:].name],
                    new_df=df2.loc[df2.iloc[0,:].name],
                    pivot_symbol=pivot_col,
                )

                if not self._pivot_ok(pivot) or error:
                    pivot_index = last_index_nonzero(df).name

                    # set a new start time using pivot_index
                    retry_timeframe = copy.deepcopy(timeframe)
                    retry_timeframe.datetime_start = pd.to_datetime(pivot_index)

                    df2 = self.build_payload_get_interest(
                        kw_list=kw_list,
                        timeframe=str(retry_timeframe),
                    )

                    df = drop_col_ispartial(drop_rows_ispartial_false(df))
                    pivot_col = self.get_pivot_col(df)

                    error, pivot = self.get_pivot(
                        ref_df=df.loc[pivot_index],
                        new_df=df2.loc[pivot_index],
                        pivot_symbol=pivot_col,
                    )
                    df = df.loc[:pivot_index]

                df2.drop(df.iloc[-1,:].name, axis=0, inplace=True)
            else:
                pivot = 1

            df2 = drop_col_ispartial(df2)
            df2 = df2.mul(pivot)

            df = pd.concat([df, df2], axis=0)

        return df, error


    def build_payload_get_interest(
        self,
        kw_list:KwList,
        timeframe:str
    ) -> pd.DataFrame:
        self.build_payload(
            kw_list=kw_list,
            timeframe=timeframe,
        )
        try:
            df = pd.DataFrame(self.interest_over_time())
        except pytrends.exceptions.ResponseError as pytrend_response_error:
            print(pytrend_response_error)
        return df

    def get_pivot(
        self,
        ref_df:pd.DataFrame,
        new_df:pd.DataFrame,
        pivot_symbol:str,
    ) -> Iterator[Tuple[bool, float]]:
        """
        Assume pivot is being joined where ref_df and new_df have single index
        """

        ref_max = ref_df[pivot_symbol].max()
        new_max = new_df[pivot_symbol].max()

        if (ref_max == 0 or new_max == 0):
            # axis=0 axis is time:        {pivot_index}@{df2.iloc[0,:].index}
            # axis=1 axis is search term: {pivot_column}@{pivot_symbol}
            print("Error, either the numerator or denomintator was not ok.") if self.verbose else None

            error = True
            pivot = None
        else:
            error = False
            pivot = float(ref_max/new_max)

        return error, pivot

    def scale_update_df(
        self,
        ref_df: pd.DataFrame,
        new_df: pd.DataFrame,
        pivot_symbol: str,
        axis: int,
        drop_pivot:bool=True
    ) -> Iterator[Tuple[pd.DataFrame, float, bool]]:
        """
        """
        error, pivot = self.get_pivot(
            ref_df=ref_df,
            new_df=new_df,
            pivot_symbol=pivot_symbol
        )
        new_df = new_df.drop(
            [pivot_symbol],
            axis=axis
        ).mul(pivot) if drop_pivot else new_df.mul(pivot)

        return new_df, pivot, error

    def scale_update_df_split_columns(self, df, df2, pivot):
        """
        Divide and conquer strategy
        """
        # zeros group (with predefined pivot on far right of df, left of df2):
        subgroup_1 = df2.max()[df2.max()==0].index.tolist()
        subgroup_2 = df2.max()[df2.max()!=0].index.tolist()
        subgroup_1_element = df.columns[df.columns.isin(subgroup_1)].item()
        # this may go wrong if timeframe gets updated during runtime
        # (e.g. if between 08/04/2021 and 09/04/2021)
        df2_1, error1 = self.build_payload_get_interest_over_intervals(subgroup_1)
        df2_1, pivot2_1, error2 = self.scale_update_df(
            ref_df=df,
            new_df=df2_1,
            pivot_symbol=subgroup_1_element,
            axis=1,
        )

        subgroup_2 = df2.max()[df2.max()!=0].index.tolist()
        subgroup_2.append(df.max()[df.max()>=100].index.tolist()[0])
        # this may go wrong when timeframe gets updated
        # (e.g. if between 08/04/2021 and 09/04/2021)
        df2_2, error3 = self.build_payload_get_interest_over_intervals(subgroup_2)
        df2_2, pivot2_2, error4 = self.scale_update_df(
            ref_df=df,
            new_df=df2_2,
            pivot_symbol=subgroup_2[-1],
            axis=1,
        )

        return pd.concat([df, df2_1, df2_2], axis=1)

    def check_split_columns(self, df:pd.DataFrame, df2:pd.DataFrame, kw:str) -> bool:
        """
        df:  main dataframe
        df2: dataframe used to update main dataframe should contain a 'linker' column
        """
        group_has_zeros = (df2[kw]).all()
        return bool(group_has_zeros and not df.empty)

    def update_df(
        self,
        df: pd.DataFrame,
        df2: pd.DataFrame,
        pivot_col: str,
    ) -> pd.DataFrame:
        """
        currently just for axis=1 (column updates)
        update df with the contents of df2 (includes the scaling)
        """

        if not df.empty:
            df2_scaled, pivot, error = self.scale_update_df(
                ref_df=df,
                new_df=df2,
                pivot_symbol=pivot_col,
                axis=1,
            )
            if error:
                # Attempt divide and conquer approach
                print("Error ", error, end=' ')  if self.verbose else None
                try:
                    df = self.scale_update_df_split_columns(df, df2, pivot)
                except Exception as e:
                    print("Error ", e, end=' ') if self.verbose else None
            else:
                # no error
                df = pd.concat([df, df2_scaled], axis=1)
        else:
            # the 'first time' case
            df = df2
        return df

    def run_groups(self, kw_list: list):
        """
        Create new columns in dataframe by running google trends api
        All columns should get updated
        """

        self.i=0
        df  = pd.DataFrame()
        df2 = pd.DataFrame()

        too_little_data = []
        break_flag = False

        while True:
            group=get_new_group(
                df,
                kw_list,
                too_little_data
            )
            self.i+=1
            print(f"Generated kw_list of {len(group)}") if self.verbose else None

            df2, error = self.build_payload_get_interest_over_intervals(
                kw_list=KwList._kwlist_unique_elements(group)
            )

            # TODO: extract retry logic to an outer loop
            df = self.update_df(df, df2, group[0])

            # remove done columns from kw_list
            for column_done in set(df.columns.tolist()).intersection(kw_list):
                kw_list.remove(column_done)

            df_empties = (df==0).all()

            # remove columns that are empties (if any exist); append to a list
            little_data = df.T[df_empties].T.columns.tolist()
            if little_data:
                print("Not enough data found for ", little_data) if self.verbose else None
                too_little_data.extend(little_data)

            # break clause
            if break_flag:
                break
            if len(kw_list)<=5:
                break_flag=True

            df = df.T[~df_empties].T

        return df, too_little_data

    def run_backoff(self, kw_list, granularity):
        kw_list_pass = kw_list # 1st pass
        df  = pd.DataFrame([])

        for pass_idx in range(1, granularity + 1):
            print(f"Pass {pass_idx} out of granularity passes: {granularity}") if self.verbose else None
            df2, too_little_data = self.run_groups(kw_list_pass)

            # the following line will get data that has low 'granularity'
            cols_above_cutoff, cols_below_cutoff = partition_cols(
                df=df2,
                cutoff_pct=self.cutoff_pct,
            )

            keep_cols = self._get_next_pivots(cols=cols_above_cutoff)
            retry_cols = self._get_next_pivots(cols=cols_below_cutoff)

            # get 'low quality' kw (columns) to retry as defined by cutoff_pct
            # and columns there was too little data for
            too_little_data.extend(retry_cols)
            kw_list_pass = too_little_data
            # keep_cols[-1] is used to pivot and join the data
            ### TODO: UPDATE THE FOLLOWING TWO LINES
            kw_list_pass.insert(0, keep_cols[-1])

            if not df.empty:
                df2 = drop_col_ispartial(drop_rows_ispartial_false(df2[keep_cols]))
                df = self.update_df(df.drop(keep_cols[:-1],axis=1), df2, keep_cols[-1])

                #df2_scaled, pivot, error = self.scale_update_df(
                #    ref_df=df,
                #    new_df=df2,
                #    pivot_symbol=pivot_col,
                #    axis=1,
                #)
            else:
                df = df2

        return df
