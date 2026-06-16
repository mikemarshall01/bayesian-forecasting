# Data fixtures

Cached public market-data (Binance klines and similar), committed so the notebooks
run **fully offline** with no API key and no network: Binance returns HTTP 451 from
many cloud/CI IP ranges, so a live fetch is not reproducible. Regenerate via the
repo data layer. Public/derived market data only.
