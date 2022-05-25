import sys
import json
import re

from datetime import date
from datetime import timedelta

from .table_manager import PyTrendTableManager
from .timeframes.timeframes import Timeframes

from pathlib import Path

import argparse

import pandas as pd

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
        cutoff_pct=10, # TODO move cutoff_pct to run_backoff func
        sleep=60,
        verbose=False,
    )

    df = pytrend_table_manager.run_backoff(
        kw_list=pytrends_terms,
        granularity=2,
    )
    return df


def get_args():
    parser = argparse.ArgumentParser(
        description='Process args for json and days'
    )
    parser.add_argument(
        '--json',
        dest='json',
        required=True,
        type=json.loads,
        help='json to load e.g. \'{<term1>:[<term1a>,<term1b>],<term2>:[<term2a>,<term2b>]}\'',
    )
    parser.add_argument(
        '--aggregate',
        action="store_true",
        help='aggregate over search terms naively (adding them together)',
    )
    parser.add_argument(
        '--sep',
        dest='csv_sep',
        type=str,
        default=',',
        help='separator for csv file'
    )
    parser.add_argument(
        '--outpath',
        dest='outpath',
        required=True,
        type=Path,
        help='file output path'
    )
    parser.add_argument(
        '-D',
        '--days',
        dest='days',
        required=True,
        type=int,
        help='days ago'
    )
    return parser.parse_args()

def main():
    args = get_args()
    trends_dfs_to_aggregate = {}

    if args.aggregate:
        # expand data into aggregation by simply summing
        aggregation = pd.DataFrame(
            {
                main_term: get_pytrends(
                    pytrends_terms=pytrends_terms,
                    days_ago=args.days,
                ).sum(axis=1)
                for main_term, pytrends_terms in args.json.items()
            }
        )

        aggregation.to_csv(args.outpath, sep=args.csv_sep)

    else:

        for pytrends_terms in args.json.values():
            trends_df = get_pytrends(
                pytrends_terms=pytrends_terms,
                days_ago=args.days,
            )

            trends_df.to_csv(args.outpath, sep=args.csv_sep)

if __name__=="__main__":
    main()
