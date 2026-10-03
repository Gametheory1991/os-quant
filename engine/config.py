"""Strategy configuration loader.

A strategy config is a small YAML file, e.g. configs/value_momentum_everywhere.yaml:

    name: "Value and Momentum Everywhere (ETF)"
    universe: etfs            # -> universes/etfs.yaml
    signal:
      value_lookback_months: 60
      value_skip_months: 12
      momentum_lookback_months: 12
      momentum_skip_months: 1
      terciles: 3
    costs_bps: 10.0           # one-way transaction cost per unit turnover
    borrow_bps_annual: 50.0   # borrow cost on short notional (long-short books)
    benchmark: blend_50_50_TLT_IEF
    start: "2012-06-01"

``load_config(name)`` returns the parsed dict; ``resolve_universe`` merges the
universe definition in.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .data import load_universe

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["universe_def"] = load_universe(cfg["universe"])
    return cfg
