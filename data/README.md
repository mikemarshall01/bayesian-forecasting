# Data fixtures

Cached Binance public daily klines, committed so the notebook runs offline with no API key
(Binance returns HTTP 451 from many cloud/CI IP ranges, so a live fetch is not always
reproducible):

- `BTCUSDT_1d.parquet`, `ETHUSDT_1d.parquet`, `SOLUSDT_1d.parquet` -- daily OHLCV for the three
  assets the hierarchical model pools across.

Regenerate or extend via `src/data.py`. Public market data; derived OHLCV only.
