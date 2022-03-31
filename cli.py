from src.timeframes.timeframes import Timeframes
from src.kw_list import KwList

import sys

import pandas as pd

from typing import Iterator, Tuple
from abc import abstractmethod

import json
import re

from pytrends.request import TrendReq
import pytrends

from pathlib import Path

import numpy as np
import math

from datetime import date
from datetime import timedelta
from datetime import datetime

import copy

class PyTrendTableManager(TrendReq):
    def __init__(
        self,
        timeframes: Timeframes,
        cutoff_pct: 10,
        sleep=60,
    ) -> None:

        super().__init__()

        self.timeframes=timeframes
        self.cutoff_pct=cutoff_pct
        self.sleep=sleep
        self.batch_counter=0

    @staticmethod
    def df_drop_partial(df:pd.DataFrame) -> pd.DataFrame:
        return df[df.isPartial=="False"].drop("isPartial", axis=1)

    @staticmethod
    def last_index_nonzero(df:pd.DataFrame) -> pd.Index:
        return (df[(df!=0).any(axis=1)].iloc[-1]!=0)

    def get_cols_where_last_index_nonzero(self, df:pd.DataFrame) -> pd.DataFrame:
        return df.columns[self.last_index_nonzero(df)]

    @staticmethod
    def get_pivot_col(df:pd.DataFrame) -> str:
        """
        df.max() will get the maximum columns
        then the sort and index[0] will get the maximum value column
        """
        return df.max().sort_values(ascending=False).index[0]

    @staticmethod
    def _pivot_ok(pivot: float) -> bool:
        if math.isnan(pivot):
            print(f"Error. Pivot is nan")
            return True
        elif math.isinf(pivot):
            print(f"Error. Pivot is inf")
            return True
        elif pivot==0:
            print(f"Error. Pivot is zero")
            return True
        else:
            print("Pivot ok")
            return False

    @staticmethod
    def get_new_group(df:pd.DataFrame, kw_list, too_little_data) -> list:
        """
        This is bound to the maximum number of keywords 5 you can sent using the
        API
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

    @staticmethod
    def get_data_granularity(df: pd.DataFrame) -> float:
        return 100 * df.nunique().sum() / np.dot(*df.shape)

    @staticmethod
    def _get_next_pivots(cols: pd.Series) -> list:
        return cols.sort_values(ascending=False).index.tolist()

    @staticmethod
    def _partition_cols(df, cutoff_pct):
        above_cutoff = (df.nunique()>cutoff_pct)
        return df.nunique()[above_cutoff], df.nunique()[~above_cutoff]

    def build_payload_get_interest_over_intervals(self, kw_list):
        df = pd.DataFrame([])
        error = False

        for timeframe in self.timeframes:
            #2. build_payload_get_interest
            df2 = self.build_payload_get_interest(kw_list, str(timeframe))
            pivot_col = self.get_pivot_col(self.df_drop_partial(df=df2))

            if not df.empty:

                pivot_col = self.get_pivot_col(
                    df=df[self.get_cols_where_last_index_nonzero(df)],
                )

                assert df.iloc[-1,:].name == df2.iloc[0,:].name

                pivot, error = self.get_pivot(
                    df.loc[df.iloc[-1,:].name],
                    df2.loc[df2.iloc[0,:].name],
                    pivot_col,
                )

                print(f"Error? {pivot} {error}")

                if not self._pivot_ok(pivot) or error:
                    pivot_index = self.last_index_nonzero(df).name

                    # set a new start time using pivot_index
                    retry_timeframe = copy.deepcopy(timeframe)
                    retry_timeframe.datetime_start = pd.to_datetime(pivot_index)

                    df2 = self.build_payload_get_interest(
                        kw_list,
                        str(retry_timeframe),
                    )

                    pivot_col = self.get_pivot_col(
                        self.df_drop_partial(df2)
                    )

                    pivot, error = self.get_pivot(
                        df.loc[pivot_index],
                        df2.loc[pivot_index],
                        pivot_col,
                    )
                    df = df.loc[:pivot_index]

                df2.drop(df.iloc[-1,:].name, axis=0, inplace=True)
            else:
                pivot = 1

            # df2 = self.df_drop_partial(df2)
            df2.drop("isPartial", axis=1, inplace=True)

            df2 = df2.mul(pivot)
            df2.loc[df2.iloc[0,:].name][pivot_col]

            # update_timeframe
            df = pd.concat([df, df2], axis=0)

            if True in df2.isna().any().unique():
                import pdb; pdb.set_trace()

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
            df = self.interest_over_time()
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
        assumes pivot is being joined where ref_df and new_df have single index
        """

        error = False
        ref_max = ref_df[pivot_symbol].max()
        new_max = new_df[pivot_symbol].max()

        pivot = float(ref_max/new_max)

        if ref_max == 0 or new_max == 0:
            error = True

        if not self._pivot_ok(pivot):
            # axis=0 axis is time:        {pivot_index}@{df2.iloc[0,:].index}", end=" ")
            # axis=1 axis is search term: {pivot_column}@{pivot_symbol}", end=" ")
            print("Exit data aquisition")
            error = True

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
        error, pivot = self.get_pivot(ref_df, new_df, pivot_symbol)

        if drop_pivot:
            new_df = new_df.drop([pivot_symbol], axis=axis).mul(pivot)
        else:
            new_df = new_df.mul(pivot)

        return new_df, pivot, error

    def scale_update_df_split_columns(self, df, df2, pivot):
        """
        Divide and conquer strategy
        """
        # job 1
        # zeros group (with predefined pivot on far right of df, left of df2):
        subgroup_1 = df2.max()[df2.max()==0].index.tolist()
        subgroup_2 = df2.max()[df2.max()!=0].index.tolist()
        subgroup_1_element = df.columns[df.columns.isin(subgroup_1)].item()

        #pivot = df.columns[df.columns.isin((df2.max()!=0).index.tolist())].item()
        # this may go wrong when timeframe gets updated
        df2_1, error1 = self.build_payload_get_interest_over_intervals(subgroup_1)
        df2_1, pivot2_1, error2 = self.scale_update_df(
            ref_df=df,
            new_df=df2_1,
            pivot_symbol=subgroup_1_element,
            axis=1,
        )

        subgroup_2 = df2.max()[df2.max()!=0].index.tolist()
        subgroup_2.append(df.max()[df.max()>=100].index.tolist()[0])

        # this may go wrong when timeframe is variable
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
                print("Error ", error, end=' ')
                # divide and conquer approach
                try:
                    df = self.scale_update_df_split_columns(df, df2, pivot)
                except Exception as e:
                    print("Error ", e, end=' ')
            else:
                # no error
                df = pd.concat([df, df2_scaled], axis=1)

            print()
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
        df  = pd.DataFrame([])
        df2 = pd.DataFrame([])

        too_little_data = []
        break_flag = False

        while True:
            group=self.get_new_group(
                df,
                kw_list,
                too_little_data
            )
            self.i+=1
            print(f"Generated kw_list of {len(group)}, done shape:{df.shape}")

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
                print("not enough data found for ", little_data)
                too_little_data.extend(little_data)

            # break clause
            if break_flag:
                break
            if len(kw_list)<=5:
                break_flag=True

            df = df.T[~df_empties].T

        return df, too_little_data

    @staticmethod
    def drop_intersecting_cols_from_df(
        df:pd.DataFrame,
        df2:pd.DataFrame,
        except_col:str,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        else:
            import pdb; pdb.set_trace()
            df = df[df.columns.difference(df2.columns).tolist() + [except_col]]
        return df


    def run_backoff(self, kw_list, granularity):
        kw_list_pass = kw_list # 1st pass
        df  = pd.DataFrame([])

        for pass_idx in range(1, granularity + 1):
            print(f"pass {pass_idx} out of {granularity}")
            df2, too_little_data = self.run_groups(kw_list_pass)

            # the following line will get data that has low 'granularity'
            cols_above_cutoff, cols_below_cutoff = self._partition_cols(
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

#            df2 = df2[keep_cols]

            import pdb; pdb.set_trace()
            if not df.empty:
                df = self.update_df(df, df2[keep_cols], keep_cols[-1])
            else:
                df = df2

        return df

def get_pytrends(pytrends_terms: list, days_ago=365):

    from_date=(date.today() - timedelta(days=days_ago))
    to_date=date.today()
    pytrends = TrendReq()

    timeframes = Timeframes(
            datetime_start=from_date,
            datetime_end=to_date,
            max_interval_days=240,
    ).get_datetime_intervals()

    pytrend_table_manager = PyTrendTableManager(
            timeframes,
            cutoff_pct=10,
            sleep=60,
    )

    df = pytrend_table_manager.run_backoff(
            kw_list=pytrends_terms,
            granularity=2,
    )

    return df

def rstrip_split(l):
    return l.rstrip().split('\t')

if __name__=="__main__":
    for days in [365]:
        trends_df = get_pytrends(
                [
                    "MFST", #1
                    "Pytrends", #2
                    "Google Trends", #3
                    "George Bush", #4
                    "9-11", #5
                    "rehabilitation", #7
                    "AAPL", #8
                ],
                days,
        )
        import pdb; pdb.set_trace()
        print(trends_df)
