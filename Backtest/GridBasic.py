from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class Candle:
	timestamp: dt.datetime
	open: float
	high: float
	low: float
	close: float


def load_candles(path: Path, limit: int | None = None) -> list[Candle]:
	candles: list[Candle] = []
	with path.open("r", encoding="ascii", newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			try:
				timestamp = dt.datetime.fromisoformat(row["timestamp_utc"])
				candle = Candle(
					timestamp=timestamp,
					open=float(row["open"]),
					high=float(row["high"]),
					low=float(row["low"]),
					close=float(row["close"]),
				)
			except (KeyError, ValueError) as exc:
				raise RuntimeError(f"Malformed row in {path}: {row}") from exc
			candles.append(candle)
			if limit is not None and len(candles) >= limit:
				break
	if not candles:
		raise RuntimeError(f"No candles loaded from {path}")
	return candles


def build_grid(lower: float, upper: float, levels: int) -> list[float]:
	if lower <= 0 or upper <= 0 or upper <= lower:
		raise ValueError("Grid bounds must be positive and upper > lower")
	if levels < 2:
		raise ValueError("Grid level count must be at least 2")
	step = (upper - lower) / (levels - 1)
	return [lower + step * idx for idx in range(levels)]


def compute_sharpe_ratio(
	equities: Sequence[float],
	timestamps: Sequence[dt.datetime],
	annual_risk_free: float,
) -> float | None:
	if len(equities) < 2 or len(equities) != len(timestamps):
		return None
	seconds_per_year = 365.0 * 24 * 3600
	excess_returns: list[float] = []
	step_seconds: list[float] = []
	for idx in range(1, len(equities)):
		prev_equity = equities[idx - 1]
		curr_equity = equities[idx]
		if prev_equity <= 0:
			continue
		delta_seconds = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
		if delta_seconds <= 0:
			continue
		period_return = (curr_equity / prev_equity) - 1.0
		risk_free_period = annual_risk_free * (delta_seconds / seconds_per_year)
		excess_returns.append(period_return - risk_free_period)
		step_seconds.append(delta_seconds)
	if len(excess_returns) < 2:
		return None
	mean_excess = sum(excess_returns) / len(excess_returns)
	variance = sum((value - mean_excess) ** 2 for value in excess_returns) / (
		len(excess_returns) - 1
	)
	if variance <= 0:
		return None
	sharpe_period = mean_excess / math.sqrt(variance)
	avg_step = sum(step_seconds) / len(step_seconds)
	if avg_step <= 0:
		return sharpe_period
	periods_per_year = seconds_per_year / avg_step
	if periods_per_year <= 0:
		return sharpe_period
	return sharpe_period * math.sqrt(periods_per_year)


class GridBacktester:
	def __init__(
		self,
		grid_levels: Sequence[float],
		fee_rate: float,
		quote_balance: float,
		base_balance: float,
		initial_equity: float,
	) -> None:
		if fee_rate < 0:
			raise ValueError("Fee rate must be non-negative")
		self.levels = sorted(grid_levels)
		self.fee_rate = fee_rate
		self.quote_balance = quote_balance
		self.base_balance = base_balance
		if len(self.levels) < 2:
			raise ValueError("Grid requires at least two levels")
		if initial_equity <= 0:
			raise ValueError("Initial equity must be positive for share-based sizing")
		self.share_value = initial_equity / (len(self.levels) - 1)
		self.positions = [0.0] * (len(self.levels) - 1)
		self.buy_count = 0
		self.sell_count = 0
		self.skipped_buys = 0
		self.skipped_sells = 0

	def run(self, candles: Iterable[Candle]) -> list[float]:
		equities: list[float] = []
		iterator = iter(candles)
		try:
			first = next(iterator)
		except StopIteration as exc:
			raise RuntimeError("No candle data supplied") from exc
		prev_price = first.open
		equities.append(self.snapshot(prev_price)["equity"])
		self._process_candle(prev_price, first)
		equities.append(self.snapshot(first.close)["equity"])
		prev_price = first.close
		for candle in iterator:
			self._process_candle(prev_price, candle)
			equities.append(self.snapshot(candle.close)["equity"])
			prev_price = candle.close
		return equities

	def _process_candle(self, prev_price: float, candle: Candle) -> None:

		current = prev_price
		low_extreme = min(candle.low, candle.open, prev_price)
		high_extreme = max(candle.high, candle.open, prev_price)

		if low_extreme < current:
			self._traverse_down(current, low_extreme)
			current = low_extreme

		if high_extreme > current:
			self._traverse_up(current, high_extreme)
			current = high_extreme

		close_price = candle.close
		if close_price < current:
			self._traverse_down(current, close_price)
		elif close_price > current:
			self._traverse_up(current, close_price)

	def _traverse_down(self, start_price: float, end_price: float) -> None:
		if end_price >= start_price:
			return
		for idx in reversed(range(len(self.levels) - 1)):
			level = self.levels[idx]
			if end_price <= level < start_price:
				self._execute_buy(idx)

	def _traverse_up(self, start_price: float, end_price: float) -> None:
		if end_price <= start_price:
			return
		for idx in range(len(self.levels) - 1):
			upper = self.levels[idx + 1]
			if start_price < upper <= end_price:
				self._execute_sell(idx)

	def _execute_buy(self, idx: int) -> None:
		price = self.levels[idx]
		if self.positions[idx] > 0:
			return
		base_amount = self.share_value / price
		cost = price * base_amount
		total_cost = cost * (1 + self.fee_rate)
		if total_cost > self.quote_balance:
			self.skipped_buys += 1
			return
		self.quote_balance -= total_cost
		self.base_balance += base_amount
		self.positions[idx] += base_amount
		self.buy_count += 1

	def _execute_sell(self, idx: int) -> None:
		if idx >= len(self.positions):
			return
		base_amount = self.positions[idx]
		if base_amount <= 0:
			return
		if self.base_balance + 1e-12 < base_amount:
			self.skipped_sells += 1
			return
		price = self.levels[idx + 1]
		proceeds = price * base_amount
		net_proceeds = proceeds * (1 - self.fee_rate)
		self.base_balance -= base_amount
		self.quote_balance += net_proceeds
		self.positions[idx] = 0.0
		self.sell_count += 1

	def snapshot(self, mark_price: float) -> dict[str, float | int]:
		equity = self.quote_balance + self.base_balance * mark_price
		return {
			"quote_balance": self.quote_balance,
			"base_balance": self.base_balance,
			"equity": equity,
			"share_value": self.share_value,
			"buy_count": self.buy_count,
			"sell_count": self.sell_count,
			"skipped_buys": self.skipped_buys,
			"skipped_sells": self.skipped_sells,
		}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Backtest a basic grid strategy using historical BTC data."
	)
	parser.add_argument(
		"--data",
		type=Path,
		default=Path("Collector/data/btc_1m.csv"),
		help="Path to the minute-level CSV produced by BinanceDownloader.",
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


def run_backtest(args: argparse.Namespace) -> dict[str, float | int]:
	data_path = args.data.resolve()
	if not data_path.exists():
		raise FileNotFoundError(data_path)
	candles = load_candles(data_path)
	if args.days_back is not None:
		if args.days_back <= 0:
			raise ValueError("--days-back must be positive")
		cutoff = candles[-1].timestamp - dt.timedelta(days=args.days_back)
		candles = [candle for candle in candles if candle.timestamp >= cutoff]
		if not candles:
			raise RuntimeError(
				f"No candles available within the last {args.days_back} days in {data_path}"
			)
	if args.limit is not None:
		if args.limit <= 0:
			raise ValueError("--limit must be positive if provided")
		candles = candles[-args.limit :]
		if not candles:
			raise RuntimeError("No candles remain after applying --limit")
	grid = build_grid(args.lower, args.upper, args.levels)
	initial_price = candles[0].open
	initial_equity = args.quote_balance + args.base_balance * initial_price
	engine = GridBacktester(
		grid_levels=grid,
		fee_rate=args.fee,
		quote_balance=args.quote_balance,
		base_balance=args.base_balance,
		initial_equity=initial_equity,
	)
	equities = engine.run(candles)
	final_price = candles[-1].close
	snapshot = engine.snapshot(final_price)
	snapshot["initial_equity"] = initial_equity
	snapshot["final_price"] = final_price
	snapshot["initial_price"] = initial_price
	if candles:
		if len(candles) > 1:
			first_delta_seconds = (candles[1].timestamp - candles[0].timestamp).total_seconds()
			if first_delta_seconds <= 0:
				first_delta_seconds = 60.0
		else:
			first_delta_seconds = 60.0
		equity_times = [
			candles[0].timestamp - dt.timedelta(seconds=first_delta_seconds)
		]
		equity_times.extend(candle.timestamp for candle in candles)
		sharpe_ratio = compute_sharpe_ratio(equities, equity_times, args.risk_free)
	else:
		sharpe_ratio = None
	snapshot["sharpe_ratio"] = sharpe_ratio
	return snapshot


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	try:
		results = run_backtest(args)
	except Exception as exc:  # noqa: BLE001
		print(f"Error: {exc}")
		return 1

	initial_equity = results["initial_equity"]
	final_equity = results["equity"]
	initial_price = results["initial_price"]
	final_price = results["final_price"]
	roi = 0.0 if initial_equity == 0 else (final_equity - initial_equity) / initial_equity
	price_return = 0.0 if initial_price == 0 else (final_price - initial_price) / initial_price
	alpha = roi - price_return
	results["roi"] = roi
	results["price_return"] = price_return
	results["alpha"] = alpha
	results["risk_free_rate"] = args.risk_free

	if args.output_format == "json":
		json.dump(results, sys.stdout, indent=2, default=str)
		print()
		return 0

	print("Grid Backtest Summary")
	print("---------------------")
	print(f"Initial equity: {initial_equity:.2f} USDT")
	print(f"Final equity: {final_equity:.2f} USDT")
	print(f"Return: {roi * 100:.2f}%")
	print(f"BTC price return: {price_return * 100:.2f}%")
	print(f"Alpha: {alpha * 100:.2f}%")
	sharpe_ratio = results.get("sharpe_ratio")
	if isinstance(sharpe_ratio, float):
		print(f"Sharpe ratio: {sharpe_ratio:.4f}")
	else:
		print("Sharpe ratio: n/a (insufficient data)")
	print(f"Share value per grid interval: {results['share_value']:.2f} USDT")
	print(f"Initial mark price: {initial_price:.2f} USDT")
	print(f"Final mark price: {final_price:.2f} USDT")
	print(f"Quote balance: {results['quote_balance']:.2f} USDT")
	print(f"Base balance: {results['base_balance']:.6f} BTC")
	print(f"Buys executed: {results['buy_count']}")
	print(f"Sells executed: {results['sell_count']}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
