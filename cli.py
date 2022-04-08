import json
import re

from datetime import date
from datetime import timedelta

from src.table_manager import PyTrendTableManager
from src.timeframes.timeframes import Timeframes

def get_pytrends(pytrends_terms: list, days_ago=365):
    from_date=(date.today() - timedelta(days=days_ago))
    to_date=date.today()

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
                "MSFT", #1
                "Pytrends", #2
                "Google Trends", #3
                "George Bush", #4
                "9-11", #5
                "rehabilitation", #7
                "AAPL", #8
            ],
            days,
        )
        print(trends_df)
