import time

from datetime import date
from datetime import datetime
from datetime import timedelta

class Timeframes:

    def __init__(self, datetime_start: datetime, datetime_end: datetime, max_interval_days:int):
        self.datetime_start = datetime_start
        self.datetime_end   = datetime_end
        self.max_interval_days = max_interval_days

    def get_datetime_intervals(self):

        datetime_chunks = []
        start_interval_datetime = self.datetime_start

        while True:
            end_interval_datetime = start_interval_datetime + timedelta(self.max_interval_days)

            if end_interval_datetime > self.datetime_end:
                datetime_chunks.append(
                        self._get_str_interval(start_interval_datetime, self.datetime_end)
                )
                break
            else:
                datetime_chunks.append(
                        self._get_str_interval(start_interval_datetime, end_interval_datetime)
                )

            start_interval_datetime = end_interval_datetime

        return datetime_chunks

    def _get_str_interval(self, start_datetime: date, end_datetime: date) -> str:
        return "{} {}".format(
            start_datetime.strftime("%Y-%m-%d"),
            end_datetime.strftime("%Y-%m-%d"),
        )



