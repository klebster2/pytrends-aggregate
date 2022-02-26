import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from pathlib import Path

from abc import ABC, abstractmethod

##ptrend

import json

import re

from collections import deque
from itertools import islice

import time

from datetime import date
from datetime import datetime
from datetime import timedelta

import pandas as pd

from pytrends.request import TrendReq
import pytrends

from pathlib import Path

import numpy as np


class PyTrendTableManager(TrendReq):

    # afaik...  this is a kind of heuristic
    MAX_DAYS=240
    def __init__(
        self,
        year_start:int,
        month_start:int,
        day_start:int,
        hour_start:int,
        year_end:int,
        month_end:int,
        day_end:int,
        hour_end:int,
        backoff_factor=60,
        sleep=60,
    ) -> None:

        super().__init__()

        print(
            year_start,
            month_start,
            day_start,
            hour_start,
            year_end,
            month_end,
            day_end,
            hour_end,
        )

        self.year_start=year_start
        self.month_start=month_start
        self.day_start=day_start
        self.hour_start=hour_start
        self.year_end=year_end
        self.month_end=month_end
        self.day_end=day_end
        self.hour_end=hour_end

        self.datetime_start = datetime(self.year_start, self.month_start, self.day_start)
        self.datetime_end   = datetime(self.year_end, self.month_end, self.day_end)

        self.sleep=sleep

        self.batch_counter=0

    def get_datetime_intervals(self):

        datetime_chunks = []
        start_interval_datetime = self.datetime_start

        while True:
            end_interval_datetime = start_interval_datetime + timedelta(self.MAX_DAYS)

            if end_interval_datetime > self.datetime_end:
                end_interval_datetime = self.datetime_end
                datetime_chunks.append(
                    tuple([
                        start_interval_datetime,
                        end_interval_datetime
                    ])
                )
                break
            else:
                datetime_chunks.append(
                    tuple([
                        start_interval_datetime,
                        end_interval_datetime
                    ])
                )

            start_interval_datetime = end_interval_datetime

        return datetime_chunks

    def build_payload_get_interest(self, kw_list, timeframe):
        self.build_payload(
            kw_list=kw_list,
            timeframe=timeframe,
        )
        return self.interest_over_time()

    def build_payload_get_interest_over_intervals(self, kw_list):
        df = pd.DataFrame([])

        for start_datetime, end_datetime in self.get_datetime_intervals():
            timeframe = "{} {}".format(
                start_datetime.strftime("%Y-%m-%d"),
                end_datetime.strftime("%Y-%m-%d"),
            )
            # update_timeframe
            df = pd.concat(
                [df, self.build_payload_get_interest(kw_list, timeframe)],
                axis=1
            )


    def scale_update_df(self, ref_df, new_df, pivot_symbol):
        ref_max = ref_df[pivot_symbol].max()
        new_max = new_df[pivot_symbol].max()
        pivot = float(ref_max/new_max)
        dropcols = ["isPartial", pivot_symbol]
        return new_df.drop(dropcols, axis=1).mul(pivot)


    def run_groups(self, groups):
        self.i=0
        df  = pd.DataFrame([])
        df2 = pd.DataFrame([])

        too_little_data = []

        for group in groups:

            if len(too_little_data)>0 and too_little_data[-1] == group[0]:
                group[0] = df.columns[-1]

            self.i+=1
            self.batch_counter+=1


            df2 = self.build_payload_get_interest_over_intervals(group)
            df2.drop(["isPartial"], axis=1, inplace=True)

            # if 'linker' is all 0...
            if (df2[group[0]]==0).all() and not df.empty:
                # divide and conquer approach

                # split search terms into two lists (zeros and > 100) and 2 jobs

                # job 1
                # zeros group (with predefined pivot on far right of df, left of df2):
                subgroup_1 = df2.max()[df2.max()==0].index.tolist()
                # this may go wrong when timeframe gets updated
                df2_1 = self.build_payload_get_interest(subgroup_1)
                df2_1 = self.scale_update_df(
                    ref_df=df,
                    new_df=df2_1,
                    pivot_symbol=subgroup_1[0],
                )

                # job 2
                # reuse a 'pivot' from the main array:
                subgroup_2 = df2.iloc[:,~df2.columns.isin(subgroup_1)].columns.tolist()
                subgroup_2.append(df.max()[df.max()>=100].index.tolist()[0])

                # this may go wrong when timeframe is variable
                df2_2 = self.build_payload_get_interest(subgroup_2)

                df2_2 = self.scale_update_df(
                    ref_df=df,
                    new_df=df2_2,
                    pivot_symbol=subgroup_2[-1],
                )

                df = pd.concat(
                    [df, df2_1, df2_2],
                    axis=1,
                )

            elif not df.empty:
                linking_symbol_fraction = float(df[group[0]].max()/df2[group[0]].max())

                df = pd.concat(
                    [df, df2[group[1:]].mul(linking_symbol_fraction)],
                    axis=1
                )
            else:
                df = df2

            # remove empties if any exist, append them to list
            little_data = df.T[(df==0).all()].T.columns.tolist()
            if little_data:
                print("not enough data found for ", little_data)
                too_little_data.extend(little_data)

            df = df.T[~(df==0).all()].T
        return df, too_little_data

    def run_backoff(self, kw_list):

        # first pass
        groups = [
            [s for s in list(group) if s!=None]
            for group in sliding_window(kw_list)
        ]
        df, too_little_data = self.run_groups(groups)
        print("first pass")
        print(f"{100*df.nunique().sum()/np.dot(*df.shape):.2f}% granularity")

        # second pass
        # keep values only, that have a maxmimum
        df_first_pass = df[df.nunique()[(df.nunique()>10)].index]
        pivot = df.nunique()[(df.nunique()>10)].sort_values(ascending=False).index[-1]
        low_quality_kws = df.nunique()[(df.nunique()<=10)].index.tolist()
        low_quality_kws.extend(too_little_data)
        low_quality_kws.insert(0, pivot)

        groups = [
            [s for s in list(group) if s!=None]
            for group in sliding_window(low_quality_kws)
        ]

        df, too_little_data = self.run_groups(groups)

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
    from_date_fmt = from_date.strftime("%Y-%m-%d")
    to_date_fmt = to_date.strftime("%Y-%m-%d")
    pytrends = TrendReq()

    pytrend_table_manager = PyTrendTableManager(
        year_start=datetime.now().year-1,
        month_start=datetime.now().month,
        day_start=datetime.now().day,
        hour_start=0,
        year_end=datetime.now().year,
        month_end=datetime.now().month,
        day_end=datetime.now().day,
        hour_end=datetime.now().hour,
        backoff_factor=5,
        sleep=60,
    )

    import pdb; pdb.set_trace()

    df = pytrend_table_manager.run_backoff(kw_list=pytrends_terms)

    return df

def rstrip_split(l):
    return l.rstrip().split('\t')


SCOPE = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]

class GoogleSheetsCredentialManager:
    def __init__(self, google_key_path:Path):
        self._google_key_path = google_key_path

    def resolve_creds(self):
        return ServiceAccountCredentials.from_json_keyfile_name(
            self._google_key_path.resolve(),
            SCOPE,
        )


class GoogleSheetsManager:
    def __init__(self, sheets_name:str):
        self._sheets_name = sheets_name

    @property
    def sheets_name(self):
        return self._sheets_name

    @sheets_name.setter
    def sheets_name(self, value):
        self._sheets_name = value


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

    for days in [365, 90]:
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

