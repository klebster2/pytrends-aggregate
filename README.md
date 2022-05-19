# pytrends-aggregate

Aggregate 5+ search terms and longer time periods set by the Google Trends API which is accessed by the pytrends python library.

# Installation

```bash
python -m pip install git+https://github.com/klebster2/pytrends-aggregate.git@main
```

# Usage Notes

Note that you should use this sparingly, as you can run into 429 errors (too many requests) if frequently you abuse the (implicit) API quota.

To search for a large group of trends over a long period, you could use:

```bash
python3 ./cli.py --json '{"Market Pulses":["should I buy","FTSE","SP500","index fund","market crash","recession","pandemic"]}' --outpath SP500.tsv -D 5000 --sep '\t'
```

You can aggregate also (simple sum) of the data altogether using the argument:

`--aggregate`

Another option might be to take `pct_change()` day by day, and aggregate correlated groups only.

In the "Market Pulses" data above, the following groups may be correlated:

```
'{"Market Pulses Fear":["market crash","recession","pandemic","war","disaster","inflation","debt"]}'
'{"Market Pulses Greed"}:["should I buy","FTSE","SP500","index fund","earnings","hodl"]'
```

It may be interesting to compute statistics (Moving Averages, etc.) over the trends before providing them as input to a model.
