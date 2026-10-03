"""Unit tests for engine.data — fixture-based, no network.

The Yahoo JSON parser is tested against a hand-built minimal chart response;
the HTTP layer is monkeypatched so no test touches the network.
"""
import json
from datetime import date

import pandas as pd

import engine.data as data


def _yahoo_payload():
    # 3 trading days of synthetic AAPL-like data + one dividend
    return {
        "chart": {
            "result": [{
                "timestamp": [1609718400, 1609804800, 1609891200],  # 2021-01-04..06
                "indicators": {
                    "quote": [{
                        "open": [100.0, 101.0, 102.0],
                        "high": [101.0, 102.0, 103.0],
                        "low": [99.0, 100.0, 101.0],
                        "close": [100.5, 101.5, 102.5],
                        "volume": [1000, 1100, 1200],
                    }],
                    "adjclose": [{"adjclose": [90.0, 91.0, 92.0]}],
                },
                "events": {"dividends": {"1609804800": {"amount": 0.5, "date": 1609804800}}},
            }],
            "error": None,
        }
    }


def test_fetch_yahoo_daily_parses_fixture(monkeypatch):
    blob = json.dumps(_yahoo_payload()).encode()

    def fake_get(url, timeout=30):
        assert "chart/TEST" in url and "interval=1d" in url
        return blob

    monkeypatch.setattr(data, "_http_get", fake_get)
    df = data.fetch_yahoo_daily("TEST", start=date(2021, 1, 1), end=date(2021, 1, 7))
    assert list(df.columns) == ["open", "high", "low", "close", "adjclose", "volume", "dividends"]
    assert len(df) == 3
    assert df["adjclose"].tolist() == [90.0, 91.0, 92.0]
    assert df["dividends"].iloc[1] == 0.5
    assert df.index[0].date() == date(2021, 1, 4)


def test_fetch_yahoo_daily_raises_on_empty_result(monkeypatch):
    blob = json.dumps({"chart": {"result": None, "error": {"description": "not found"}}}).encode()
    monkeypatch.setattr(data, "_http_get", lambda url, timeout=30: blob)
    try:
        data.fetch_yahoo_daily("NOPE")
    except RuntimeError as exc:
        assert "NOPE" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_load_universe_reads_yaml():
    u = data.load_universe("etfs")
    assert "SPY" in u["tickers"] and "TLT" in u["tickers"]
    assert set(u["sleeves"]["fixed_income"]) == {"TLT", "IEF", "SHY", "TIP", "LQD", "HYG"}


def test_fetch_fred_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert data.fetch_fred("DGS10") is None
