import time

from datetime import date
from datetime import datetime
from datetime import timedelta

class Timeframe:
    def __init__(self, datetime_start, datetime_end):
        self._datetime_start = datetime_start
        self._datetime_end = datetime_end

    @property
    def datetime_start(self):
        """getter"""
        return self._datetime_start

    @property
    def datetime_end(self):
        """getter"""
        return self._datetime_end

    @datetime_start.setter
    def datetime_start(self, value):
        self._datetime_start = value

    @datetime_end.setter
    def datetime_end(self, value):
        self._datetime_end = value

    def __str__(self):
        return "{} {}".format(
            self.datetime_start.strftime("%Y-%m-%d"),
            self.datetime_end.strftime("%Y-%m-%d"),
        )

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
                    Timeframe(start_interval_datetime, self.datetime_end)
                )
                break
            else:
                datetime_chunks.append(
                    Timeframe(start_interval_datetime, end_interval_datetime)
                )

            start_interval_datetime = end_interval_datetime

        return datetime_chunks
