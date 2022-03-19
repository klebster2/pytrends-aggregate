from src.google_cloud_client.sheet_connector_service.update_sheets import (
        GoogleSheetsCredentialManager,
        GoogleSheetsManager,
)

from src.timeframes.timeframes import Timeframes
import gspread

import sys

import pandas as pd

from abc import ABC, abstractmethod

import json
import re

from collections import deque
from itertools import islice

from pytrends.request import TrendReq
import pytrends

from pathlib import Path

import numpy as np
import math

from datetime import date
from datetime import timedelta
from datetime import datetime


import pdb
import logging

FORMAT = '%(asctime)s %(clientip-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

class KwList:
    @staticmethod
    def _kwlist_unique_elements(kw_list:list) -> list:
        return [element for element in set(kw_list)]


class PyTrendTableManager(TrendReq):
    def __init__(
        self,
        timeframes: Timeframes,
        sleep=60,
    ) -> None:

        super().__init__()

        self.sleep=sleep
        self.batch_counter=0
        self.timeframes=timeframes
        logging.info("__init__ of PyTrendTableManager")

    def check_pivot_nan_inf(self, pivot):
        if math.isnan(pivot) or math.isinf(pivot) or pivot==0:
            print(f"Error! pivot is {pivot}", end=' ')
            return True
        else:
            return False


    def build_payload_get_interest_over_intervals(self, kw_list):
        logging.info("build_payload_get_interest_over_intervals of PyTrendTableManager")
        logging.info("kw_list:[{', '.join(kw_list)}]")

        df = pd.DataFrame([])
        error = False

        for timeframe in self.timeframes:
            #2. build_payload_get_interest
            df2 = self.build_payload_get_interest(kw_list, str(timeframe))

            no_partial =  df2[df2.isPartial=="False"].drop("isPartial", axis=1)
            #last_index_nonzero_cols = df.columns[(df[(df!=0).any(axis=1)].iloc[-1]!=0)]
            pivot_col = no_partial.max().sort_values(ascending=False).index[0]

            if not df.empty:
                assert df.iloc[-1,:].name == df2.iloc[0,:].name

                last_index_nonzero_cols = df.columns[(df[(df!=0).any(axis=1)].iloc[-1]!=0)]
                #pivot_index = no_partial.max().sort_values(ascending=False).index[0]
                pivot_col = df[last_index_nonzero_cols].max().sort_values(ascending=False).index[0]
#                df2_1, pivot2_1, error2 = self.scale_update_df(
#                    ref_df=df,
#                    new_df=df2,
#                    pivot_symbol=df.iloc[-1,:].name,
#                    axis=0,
#                )
                prev_index = df2.iloc[0,:].name
                ref_max = df.loc[df.iloc[-1,:].name][pivot_col]
                new_max = df2.loc[df2.iloc[0,:].name][pivot_col]

                pivot = float(ref_max/new_max)
                if self.check_pivot_nan_inf(pivot):
                    print(ref_max, new_max)
                    pivot_index = df[(df!=0).any(axis=1)].iloc[-1].name

                    retry_timeframe = timeframe._get_str_interval(
                        pd.to_datetime(pivot_index), # from time
                        timeframe.datetime_end,      # to time
                    )

                    df2 = self.build_payload_get_interest(kw_list, str(retry_timeframe))

                    no_partial =  df2[df2.isPartial=="False"].drop("isPartial", axis=1)
                    pivot_col = no_partial.max().sort_values(ascending=False).index[0]
                    ref_max = df.loc[pivot_index][pivot_col]
                    new_max = df2.loc[pivot_index][pivot_col]
                    pivot = float(ref_max/new_max)
                    df = df.loc[:pivot_index]

                #ref_max = df.loc[df.iloc[-1,:].name][pivot_col]
                #new_max = df2.loc[df2.iloc[0,:].name][pivot_col]

                df2.drop(df.iloc[-1,:].name, axis=0, inplace=True)
            else:
                pivot = 1

            df2.drop("isPartial", axis=1, inplace=True)

            df2 = df2.mul(pivot)
            df2.loc[df2.iloc[0,:].name][pivot_col]

            # update_timeframe
            df = pd.concat(
                [df, df2],
                axis=0,
            )
            if True in df2.isna().any().unique():
                import pdb; pdb.set_trace()

        return df, error


    def build_payload_get_interest(self, kw_list:KwList, timeframe:str) -> pd.DataFrame:
        logging.info("build_payload_get_interest of PyTrendTableManager")
        #2. build_payload_get_interest
        self.build_payload(
            kw_list=kw_list,
            timeframe=timeframe,
        )
        try:
            df = self.interest_over_time()
        except pytrends.exceptions.ResponseError as pytrend_response_error:
            print(pytrend_response_error)

        return df


    def scale_update_df(self, ref_df, new_df, pivot_symbol, axis, drop_pivot:bool=True):
        logging.info("scaling_updated_df of PyTrendTableManager")
        error = False
        ref_max = ref_df[pivot_symbol].max()
        new_max = new_df[pivot_symbol].max()
        if ref_max == 0 or new_max == 0:
            error = True

        pivot = float(ref_max/new_max)
        if self.check_pivot_nan_inf(pivot):
            # axis=0 e.g. {pivot_index}@{df2.iloc[0,:].index}", end=" ")
            # axis=1 e.g. {pivot_column}@{pivot_symbol}", end=" ")
            print("Exit data aquisition")
            error = True

        if drop_pivot:
            new_df = new_df.drop([pivot_symbol], axis=axis).mul(pivot)
        else:
            new_df = new_df.mul(pivot)

        return new_df, pivot, error

    def scale_update_df_split_columns(self, df, df2, pivot):
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
        group_has_zeros = (df2[kw]).all()
        return bool(group_has_zeros and not df.empty)

    def run_groups(self, kw_list):
        logging.info("run_groups of PyTrendTableManager")
        self.i=0
        df  = pd.DataFrame([])
        df2 = pd.DataFrame([])

        too_little_data = []
        break_flag = False

        while True:
            if not df.empty:
                group = [df.columns[-1]]
                group.extend(kw_list[:4])
            else:
                group = kw_list[:5]

            #group = set(group).union(set(too_little_data))
            if len(too_little_data)>0 and too_little_data[-1] == group[0]:
                # better reaction when there's a removal
                group[0] = df.columns[-1]

            self.i+=1
            self.batch_counter+=1
            print(f"generating kw_list of {len(group)}", end =' ')

            df2, error = self.build_payload_get_interest_over_intervals(
                KwList._kwlist_unique_elements(group)
            )

            if not df.empty:
                print("elif not df.empty, error?=", end='')
                df2_scaled, pivot, error = self.scale_update_df(
                        ref_df=df,
                        new_df=df2,
                        pivot_symbol=group[0],
                        axis=1,
                )
                if error:
                    print(error, "\ntrying backoff strategy 1")
                    try:
                        df = self.scale_update_df_split_columns(df, df2, pivot)
                    except Exception as identifier:
                        print(identifier)
                        import pdb; pdb.set_trace()
                        sys.exit(1)
                else:
                    df = pd.concat([df, df2_scaled], axis=1)
            else:
                df = df2

            # update kw_list
            [kw_list.remove(x) for x in set(df.columns.tolist()).intersection(kw_list)]

            # remove empties if any exist, append them to list
            little_data = df.T[(df==0).all()].T.columns.tolist()
            if little_data:
                print("not enough data found for ", little_data)
                too_little_data.extend(little_data)

            # break clause
            if break_flag:
                break
            if len(kw_list)<=5:
                break_flag=True

            df = df.T[~(df==0).all()].T
        return df, too_little_data

    def run_backoff(self, kw_list):

        # first pass
        df, too_little_data = self.run_groups(kw_list)
        print("first pass")
        print(f"{100*df.nunique().sum()/np.dot(*df.shape):.2f}% granularity")

        # second pass
        # keep values only, that have a maxmimum
        df_first_pass = df[df.nunique()[(df.nunique()>10)].index]
        pivot = df.nunique()[(df.nunique()>10)].sort_values(ascending=False).index[-1]
        low_quality_kws = df.nunique()[(df.nunique()<=10)].index.tolist()
        low_quality_kws.extend(too_little_data)
        low_quality_kws.insert(0, pivot)

        df, too_little_data = self.run_groups(low_quality_kws)

        print("second pass")
        print(f"{100*df.nunique().sum()/np.dot(*df.shape):.2f}% granularity")

        df_second_pass = df
        linking_symbol_fraction = df_first_pass[pivot].max()/df_second_pass[pivot].max()

        df = pd.concat(
            [df_first_pass, df_second_pass.mul(linking_symbol_fraction)],
            axis=1
        )
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
            sleep=60,
    )

    df = pytrend_table_manager.run_backoff(
            kw_list=pytrends_terms
    )

    return df

def rstrip_split(l):
    return l.rstrip().split('\t')


def sliding_window(iterable, size=5, step=4, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))


def get_worksheet_from_worksheets(sheets, name):
    worksheet=[
        ws for ws in sheets.worksheets()
        if ws.title == name
    ].pop()
    return worksheet


if __name__=="__main__":

    google_sheets_credentials = GoogleSheetsCredentialManager(
        Path('google-key.json')
    ).resolve_creds()

    gsheets_client = gspread.authorize(
        google_sheets_credentials
    )

    rpi_watchlist_sheets = gsheets_client.open("RPI Watchlist Updater")

    rpi_watchlist_ticker_sheet = rpi_watchlist_sheets.get_worksheet(0)
    sheet_values = rpi_watchlist_ticker_sheet.get_all_values()

    df_ticker_info = pd.DataFrame(
        sheet_values[2:],
        columns=sheet_values[1]
    )

    company_search_terms_sheet = get_worksheet_from_worksheets(
        sheets=rpi_watchlist_sheets,
        name="CompanySearchTerms",
    )

    df_search_terms = pd.DataFrame(company_search_terms_sheet.get_all_values()).set_index(0).T

    #rpi_watchlist_sheet.clear()
    #rpi_watchlist_sheet.update(some_matrix)

    # with open('categories','w') as f: f.write(json.dumps(pytrend.categories()))
    # currently using

    # get mcap too
    flatten = lambda x: [i for j in x for i in j]

    search_term_list = flatten([
        df_search_terms[ticker].to_numpy().flatten().tolist()
        for ticker in df_search_terms.columns
    ])

    df_search_terms_dict = df_search_terms.to_dict("list")
    term2ticker = {v2:k for k,v in df_search_terms_dict.items() for v2 in v}

    for days in [365]:
        worksheet=[
            ws for ws in rpi_watchlist_sheets.worksheets()
            if ws.title == f"PyTrends{days}"
        ].pop()
        trends_df = get_pytrends(search_term_list, days)

        if days == 365:
            trends_df = trends_df.resample('D').interpolate('cubic')

        trends_df.reset_index(inplace=True)

        date_series = trends_df.date.apply(lambda x: x.strftime("%d/%m/%Y"))

        aggregation = pd.DataFrame(
            {
                k: trends_df[
                    set(v).intersection(trends_df.columns)
                ].sum(axis=1) for k,v in df_search_terms_dict.items()
            }
        )

        aggregation.index=date_series
        aggregation.reset_index(level=0, inplace=True)

        trend_matrix = [aggregation.columns.tolist()] + aggregation.to_numpy().tolist()

        worksheet.update(trend_matrix)

    #platform_trends_df_365.T.to_csv(sys.argv[1]+"/pytrends_matrix_365.tsv", sep='\t')
    #rpi_watchlist_ticker_sheet = rpi_watchlist_sheets.get_worksheet(0)

    #platform_trends_df_90 = get_pytrends(stock_list, 90)
    #platform_trends_df_90.T.to_csv(sys.argv[1]+"/pytrends_matrix_90.tsv", sep='\t')

    #platform_trends_df_30 = get_pytrends(stock_list, 30)
    #platform_trends_df_30.T.to_csv(sys.argv[1]+"/pytrends_matrix_30.tsv", sep='\t')

    #platform_trends_df_7 = get_pytrends(stock_list, 7)
    #platform_trends_df_7.T.to_csv(sys.argv[1]+"/pytrends_matrix_7.tsv", sep='\t')

