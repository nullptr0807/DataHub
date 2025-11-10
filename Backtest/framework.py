from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence


@dataclass
class Candle:
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class TradeEvent:
    timestamp: dt.datetime
    side: str
    price: float
    quantity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "metadata": self.metadata,
        }


class Strategy(Protocol):
    def prepare(self, candles: Sequence[Candle], context: BacktestContext) -> None:  # noqa: D401
        """Allow the strategy to initialise internal state before execution."""

    def on_candle(
        self,
        candle: Candle,
        previous: Optional[Candle],
        context: BacktestContext,
    ) -> None:
        """Process the next candle."""

    def finalize(self, context: BacktestContext) -> dict[str, Any]:
        """Return strategy-specific metrics (optional)."""


class BacktestContext:
    def __init__(self, fee_rate: float, quote_balance: float, base_balance: float) -> None:
        if fee_rate < 0:
            raise ValueError("Fee rate must be non-negative")
        self.fee_rate = fee_rate
        self.quote_balance = quote_balance
        self.base_balance = base_balance
        self.initial_equity: float | None = None
        self.trade_log: list[TradeEvent] = []
        self.buy_count = 0
        self.sell_count = 0
        self.skipped_buys = 0
        self.skipped_sells = 0

    def set_initial_equity(self, price: float) -> None:
        equity = self.mark_to_market(price)
        if equity <= 0:
            raise ValueError("Initial equity must be positive")
        self.initial_equity = equity

    def mark_to_market(self, price: float) -> float:
        return self.quote_balance + self.base_balance * price

    def buy(
        self,
        quantity: float,
        price: float,
        timestamp: dt.datetime,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if quantity <= 0:
            raise ValueError("Quantity must be positive for buy orders")
        cost = quantity * price
        total_cost = cost * (1 + self.fee_rate)
        if total_cost > self.quote_balance + 1e-12:
            self.skipped_buys += 1
            return False
        self.quote_balance -= total_cost
        self.base_balance += quantity
        self.buy_count += 1
        self.trade_log.append(
            TradeEvent(
                timestamp=timestamp,
                side="buy",
                price=price,
                quantity=quantity,
                metadata=metadata or {},
            )
        )
        return True

    def sell(
        self,
        quantity: float,
        price: float,
        timestamp: dt.datetime,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if quantity <= 0:
            raise ValueError("Quantity must be positive for sell orders")
        if quantity > self.base_balance + 1e-12:
            self.skipped_sells += 1
            return False
        proceeds = quantity * price
        net_proceeds = proceeds * (1 - self.fee_rate)
        self.base_balance -= quantity
        self.quote_balance += net_proceeds
        self.sell_count += 1
        self.trade_log.append(
            TradeEvent(
                timestamp=timestamp,
                side="sell",
                price=price,
                quantity=quantity,
                metadata=metadata or {},
            )
        )
        return True

    def snapshot(self) -> dict[str, float | int]:
        return {
            "quote_balance": self.quote_balance,
            "base_balance": self.base_balance,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "skipped_buys": self.skipped_buys,
            "skipped_sells": self.skipped_sells,
        }


def load_candles(path: Path, limit: int | None = None) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", encoding="ascii", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                timestamp = dt.datetime.fromisoformat(row["timestamp_utc"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
                else:
                    timestamp = timestamp.astimezone(dt.timezone.utc)
                candles.append(
                    Candle(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Malformed row in {path}: {row}") from exc
            if limit is not None and len(candles) >= limit:
                break
    if not candles:
        raise RuntimeError(f"No candles loaded from {path}")
    return candles


def filter_candles(
    candles: Sequence[Candle],
    *,
    days_back: int | None = None,
    limit: int | None = None,
) -> list[Candle]:
    filtered = list(candles)
    if days_back is not None:
        if days_back <= 0:
            raise ValueError("--days-back must be positive")
        cutoff = filtered[-1].timestamp - dt.timedelta(days=days_back)
        filtered = [candle for candle in filtered if candle.timestamp >= cutoff]
        if not filtered:
            raise RuntimeError("No candles remain after applying --days-back")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive if provided")
        filtered = filtered[-limit:]
        if not filtered:
            raise RuntimeError("No candles remain after applying --limit")
    return filtered


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
    sharpe_period = mean_excess / variance ** 0.5
    avg_step = sum(step_seconds) / len(step_seconds)
    if avg_step <= 0:
        return sharpe_period
    periods_per_year = seconds_per_year / avg_step
    if periods_per_year <= 0:
        return sharpe_period
    return sharpe_period * periods_per_year ** 0.5


@dataclass
class BacktestResult:
    initial_equity: float
    final_equity: float
    initial_price: float
    final_price: float
    roi: float
    price_return: float
    alpha: float
    sharpe_ratio: float | None
    risk_free_rate: float
    equity_curve: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    context: dict[str, float | int]
    strategy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "initial_price": self.initial_price,
            "final_price": self.final_price,
            "roi": self.roi,
            "price_return": self.price_return,
            "alpha": self.alpha,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_free_rate": self.risk_free_rate,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "context": self.context,
            "strategy": self.strategy,
        }


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        *,
        fee_rate: float,
        quote_balance: float,
        base_balance: float,
        risk_free_rate: float,
    ) -> None:
        self.strategy = strategy
        self.context = BacktestContext(
            fee_rate=fee_rate,
            quote_balance=quote_balance,
            base_balance=base_balance,
        )
        self.risk_free_rate = risk_free_rate

    def run(self, candles: Sequence[Candle]) -> BacktestResult:
        if not candles:
            raise RuntimeError("No candle data supplied")

        first_candle = candles[0]
        self.context.set_initial_equity(first_candle.open)
        self.strategy.prepare(candles, self.context)

        initial_equity = self.context.initial_equity
        if initial_equity is None:
            raise RuntimeError("Strategy failed to establish initial equity")
        equity_points: list[float] = []
        time_points: list[dt.datetime] = []

        baseline_ts = first_candle.timestamp
        if len(candles) > 1:
            delta = candles[1].timestamp - baseline_ts
            if delta.total_seconds() <= 0:
                delta = dt.timedelta(minutes=1)
        else:
            delta = dt.timedelta(minutes=1)
        equity_points.append(initial_equity)
        time_points.append(baseline_ts - delta)

        previous: Optional[Candle] = None
        for candle in candles:
            self.strategy.on_candle(candle, previous, self.context)
            equity_points.append(self.context.mark_to_market(candle.close))
            time_points.append(candle.timestamp)
            previous = candle

        final_equity = equity_points[-1]
        initial_price = first_candle.open
        final_price = candles[-1].close
        roi = 0.0 if initial_equity == 0 else (final_equity - initial_equity) / initial_equity
        price_return = 0.0 if initial_price == 0 else (final_price - initial_price) / initial_price
        alpha = roi - price_return
        sharpe_ratio = compute_sharpe_ratio(equity_points, time_points, self.risk_free_rate)

        result = BacktestResult(
            initial_equity=initial_equity,
            final_equity=final_equity,
            initial_price=initial_price,
            final_price=final_price,
            roi=roi,
            price_return=price_return,
            alpha=alpha,
            sharpe_ratio=sharpe_ratio,
            risk_free_rate=self.risk_free_rate,
            equity_curve=[
                {"timestamp": stamp.isoformat(), "equity": equity}
                for stamp, equity in zip(time_points, equity_points)
            ],
            trades=[event.as_dict() for event in self.context.trade_log],
            context=self.context.snapshot(),
            strategy=self.strategy.finalize(self.context),
        )
        return result