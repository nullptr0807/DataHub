from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from Backtest.framework import (
        BacktestContext,
        BacktestEngine,
        BacktestResult,
        Candle,
        Strategy,
        filter_candles,
        load_candles,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from Backtest.framework import (  # type: ignore  # noqa: WPS433
        BacktestContext,
        BacktestEngine,
        BacktestResult,
        Candle,
        Strategy,
        filter_candles,
        load_candles,
    )


def build_grid(lower: float, upper: float, levels: int) -> list[float]:
    # Construct equally spaced price checkpoints the bot will trade between.
    if lower <= 0 or upper <= 0 or upper <= lower:
        raise ValueError("Grid bounds must be positive and upper > lower")
    if levels < 2:
        raise ValueError("Grid level count must be at least 2")
    step = (upper - lower) / (levels - 1)
    return [lower + step * idx for idx in range(levels)]


@dataclass
class GridBasicStrategy(Strategy):
    grid_levels: Sequence[float]

    def __post_init__(self) -> None:
        # Keep levels sorted so the strategy can scan from low to high quickly.
        self.levels = sorted(self.grid_levels)
        if len(self.levels) < 2:
            raise ValueError("Grid requires at least two levels")
        self.share_value = 0.0
        self.positions: list[float] = [0.0] * (len(self.levels) - 1)

    def prepare(self, candles: Sequence[Candle], context: BacktestContext) -> None:
        if context.initial_equity is None:
            raise RuntimeError("Context initial equity not set prior to prepare")
        # Allocate equal quote capital to each interval between grid levels.
        self.share_value = context.initial_equity / (len(self.levels) - 1)
        self.positions = [0.0] * (len(self.levels) - 1)

    def on_candle(
        self,
        candle: Candle,
        previous: Optional[Candle],
        context: BacktestContext,
    ) -> None:
        prev_price = previous.close if previous else candle.open
        current = prev_price
        low_extreme = min(candle.low, candle.open, prev_price)
        high_extreme = max(candle.high, candle.open, prev_price)

        # Walk the price movement through the grid to trigger fills in order.
        if low_extreme < current:
            self._traverse_down(current, low_extreme, candle.timestamp, context)
            current = low_extreme

        if high_extreme > current:
            self._traverse_up(current, high_extreme, candle.timestamp, context)
            current = high_extreme

        close_price = candle.close
        if close_price < current:
            self._traverse_down(current, close_price, candle.timestamp, context)
        elif close_price > current:
            self._traverse_up(current, close_price, candle.timestamp, context)

    def finalize(self, context: BacktestContext) -> dict[str, Any]:
        return {
            "share_value": self.share_value,
            "grid_levels": list(self.levels),
            "open_positions": list(self.positions),
        }

    def _traverse_down(
        self,
        start_price: float,
        end_price: float,
        timestamp: dt.datetime,
        context: BacktestContext,
    ) -> None:
        if end_price >= start_price:
            return
        # Step through each interval the price crossed while moving down.
        for idx in reversed(range(len(self.levels) - 1)):
            level = self.levels[idx]
            if end_price <= level < start_price:
                self._execute_buy(idx, timestamp, context)

    def _traverse_up(
        self,
        start_price: float,
        end_price: float,
        timestamp: dt.datetime,
        context: BacktestContext,
    ) -> None:
        if end_price <= start_price:
            return
        # Step through each interval the price crossed while moving up.
        for idx in range(len(self.levels) - 1):
            upper = self.levels[idx + 1]
            if start_price < upper <= end_price:
                self._execute_sell(idx, timestamp, context)

    def _execute_buy(self, idx: int, timestamp: dt.datetime, context: BacktestContext) -> None:
        price = self.levels[idx]
        if self.positions[idx] > 0:
            return
        # Buy enough BTC at the lower bound to commit one share of capital.
        quantity = self.share_value / price
        executed = context.buy(
            quantity,
            price,
            timestamp,
            metadata={"level": price, "index": idx},
        )
        if executed:
            self.positions[idx] = quantity

    def _execute_sell(self, idx: int, timestamp: dt.datetime, context: BacktestContext) -> None:
        if idx >= len(self.positions):
            return
        quantity = self.positions[idx]
        if quantity <= 0:
            return
        price = self.levels[idx + 1]
        # Sell the inventory captured in the lower band once price re-enters the upper band.
        executed = context.sell(
            quantity,
            price,
            timestamp,
            metadata={"level": price, "index": idx + 1},
        )
        if executed:
            self.positions[idx] = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a basic grid strategy using historical BTC data.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Collector/data/btc_1m.csv"),
        help="Path to the candle CSV produced by BinanceDownloader.",
    )
    parser.add_argument(
        "--lower",
        type=float,
        required=True,
        help="Lower price bound for the grid.",
    )
    parser.add_argument(
        "--upper",
        type=float,
        required=True,
        help="Upper price bound for the grid.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=21,
        help="Number of grid levels (default: 21).",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Trading fee rate as a decimal (default: 0.001).",
    )
    parser.add_argument(
        "--quote-balance",
        type=float,
        default=10_000.0,
        help="Starting quote balance in USDT (default: 10000).",
    )
    parser.add_argument(
        "--base-balance",
        type=float,
        default=0.0,
        help="Starting base balance in BTC (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of candles to process after filtering (most recent first).",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        help="Restrict the backtest to the most recent N days (overrides --limit ordering).",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.02,
        help="Annual risk-free rate as a decimal for Sharpe calculation (default: 0.02).",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="Choose text for console summary or json for machine-readable output.",
    )
    return parser.parse_args(argv)


def run_backtest(args: argparse.Namespace) -> tuple[BacktestResult, list[Candle]]:
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    candles = load_candles(data_path)
    candles = filter_candles(
        candles,
        days_back=args.days_back,
        limit=args.limit,
    )
    grid_levels = build_grid(args.lower, args.upper, args.levels)

    engine = BacktestEngine(
        GridBasicStrategy(grid_levels),
        fee_rate=args.fee,
        quote_balance=args.quote_balance,
        base_balance=args.base_balance,
        risk_free_rate=args.risk_free,
    )
    return engine.run(candles), candles


def render_text(result: BacktestResult) -> None:
    print("Grid Backtest Summary")
    print("---------------------")
    print(f"Initial equity: {result.initial_equity:.2f} USDT")
    print(f"Final equity: {result.final_equity:.2f} USDT")
    print(f"Return: {result.roi * 100:.2f}%")
    print(f"BTC price return: {result.price_return * 100:.2f}%")
    print(f"Alpha: {result.alpha * 100:.2f}%")
    sharpe = result.sharpe_ratio
    if isinstance(sharpe, float):
        print(f"Sharpe ratio: {sharpe:.4f}")
    else:
        print("Sharpe ratio: n/a (insufficient data)")
    share_value = result.strategy.get("share_value")
    if isinstance(share_value, (int, float)):
        print(f"Share value per grid interval: {share_value:.2f} USDT")
    print(f"Initial mark price: {result.initial_price:.2f} USDT")
    print(f"Final mark price: {result.final_price:.2f} USDT")
    print(f"Quote balance: {result.context['quote_balance']:.2f} USDT")
    print(f"Base balance: {result.context['base_balance']:.6f} BTC")
    print(f"Buys executed: {result.context['buy_count']}")
    print(f"Sells executed: {result.context['sell_count']}")


def format_json_payload(
    result: BacktestResult,
    candles: Sequence[Candle],
) -> dict[str, Any]:
    payload = result.as_dict()
    context_snapshot = payload.pop("context", {})
    strategy_snapshot = payload.pop("strategy", {})

    payload["equity"] = payload.get("final_equity", payload.get("initial_equity"))
    payload["quote_balance"] = float(context_snapshot.get("quote_balance", 0.0))
    payload["base_balance"] = float(context_snapshot.get("base_balance", 0.0))
    payload["buy_count"] = int(context_snapshot.get("buy_count", 0))
    payload["sell_count"] = int(context_snapshot.get("sell_count", 0))
    payload["skipped_buys"] = int(context_snapshot.get("skipped_buys", 0))
    payload["skipped_sells"] = int(context_snapshot.get("skipped_sells", 0))

    payload["share_value"] = float(strategy_snapshot.get("share_value", 0.0))
    payload["grid_levels"] = list(strategy_snapshot.get("grid_levels", []))
    payload["open_positions"] = list(strategy_snapshot.get("open_positions", []))

    payload["candles"] = [
        {
            "timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
        }
        for candle in candles
    ]

    trades: list[dict[str, Any]] = []
    for trade in payload.get("trades", []):
        metadata = trade.get("metadata") or {}
        trades.append(
            {
                "timestamp": trade.get("timestamp"),
                "type": trade.get("side") or trade.get("type"),
                "price": trade.get("price"),
                "quantity": trade.get("quantity"),
                "level": metadata.get("level"),
                "metadata": metadata,
            }
        )
    payload["trades"] = trades

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result, candles = run_backtest(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1

    if args.output_format == "json":
        json.dump(format_json_payload(result, candles), sys.stdout, indent=2)
        print()
        return 0

    render_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
