"""Free market data from Binance public klines (no API key required).

Binance's ``/api/v3/klines`` endpoint is public, keyless and goes back to 2017 for the
majors. We paginate to pull full history and cache each series to ``data/`` as Parquet,
so a notebook re-run is instant and runs offline from the committed fixtures.

The ``.env`` / API-key pattern is supported for consistency with the other repos in the
handbook (some of which DO need keys, e.g. Etherscan, FRED), but THIS module needs none.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import pandas as pd
import requests

BINANCE = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
_KLINES = f"{BINANCE}/api/v3/klines"
_CACHE = Path(__file__).resolve().parent.parent / "data"
_CACHE.mkdir(exist_ok=True)

_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def get_klines(symbol: str, interval: str = "1d", start: str = "2018-01-01",
               end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Full OHLCV history for ``symbol`` (e.g. 'BTCUSDT') as a tidy DataFrame
    indexed by UTC date. Paginates the 1000-row Binance limit and caches to Parquet."""
    cache = _CACHE / f"{symbol}_{interval}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()
                      ).timestamp() * 1000)
        rows: list = []
        cursor = start_ms
        while cursor < end_ms:
            r = requests.get(_KLINES, params=dict(symbol=symbol, interval=interval,
                             startTime=cursor, limit=1000), timeout=15)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows += batch
            cursor = batch[-1][6] + 1          # next ms after last close_time
            if len(batch) < 1000:
                break
            time.sleep(0.2)                     # be polite to the public endpoint
        df = pd.DataFrame(rows, columns=_COLS)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(cache)
    return df


def get_closes(symbols: list[str], interval: str = "1d", start: str = "2018-01-01",
               end: str | None = None) -> pd.DataFrame:
    """Aligned close-price panel for several symbols (inner-joined on date)."""
    series = {s.replace("USDT", ""): get_klines(s, interval, start, end)["close"]
              for s in symbols}
    return pd.DataFrame(series).dropna()
