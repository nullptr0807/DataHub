from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path
from typing import Iterable

from binance.spot import Spot

MAX_BATCH = 1000
BINANCE_INTERVALS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}
AGG_TRADE_LIMIT = 1000
DEFAULT_SECOND_START = dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc)
COMMON_QUOTES = [
    "USDT",
    "BUSD",
    "USDC",
    "FDUSD",
    "TUSD",
    "USDD",
    "USDP",
    "GUSD",
    "BTC",
    "ETH",
    "BNB",
    "TRY",
    "EUR",
    "GBP",
    "AUD",
    "BRL",
    "BIDR",
    "IDRT",
    "RUB",
    "NGN",
    "UAH",
    "ZAR",
    "PAX",
    "DAI",
    "UST",
]

COMMON_QUOTES = [
    "USDT",
    "BUSD",
    "USDC",
    "FDUSD",
    "TUSD",
    "USDD",
    "USDP",
    "GUSD",
    "BTC",
    "ETH",
    "BNB",
    "TRY",
    "EUR",
    "GBP",
    "AUD",
    "BRL",
    "BIDR",
    "IDRT",
    "RUB",
    "NGN",
    "UAH",
    "ZAR",
    "PAX",
    "DAI",
    "UST",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical klines from Binance and write them to CSV."
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair to query (default: BTCUSDT).",
    )
    parser.add_argument(
        "--mode",
        choices=("kline", "second"),
        default="kline",
        help="Data mode: kline (candles) or second (1-second aggregates).",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        choices=sorted(BINANCE_INTERVALS),
        help="Kline interval to request (default: 1m).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to look back from now (default: 365).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("binance_klines.csv"),
        help="Destination CSV path (default: binance_klines.csv).",
    )
    parser.add_argument(
        "--target",
        action="append",
        metavar="INTERVAL=PATH",
        help=(
            "Process multiple interval/output pairs (repeatable, format: interval=path)."
        ),
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        help="Discover CSV targets automatically within a directory (e.g. data).",
    )
    parser.add_argument(
        "--start",
        help="ISO-8601 UTC start timestamp (default varies by mode).",
    )
    parser.add_argument(
        "--end",
        help="ISO-8601 UTC end timestamp (default: now UTC).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append only new klines to an existing CSV (requires --output).",
    )
    parser.add_argument(
        "--api-key",
        help="Optional Binance API key for higher rate limits.",
    )
    parser.add_argument(
        "--api-secret",
        help="Optional Binance API secret to pair with --api-key.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Delay in seconds between requests to stay within rate limits (default: 0.3).",
    )
    return parser.parse_args(argv)


def iter_klines(
    client: Spot,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    sleep_seconds: float,
) -> Iterable[list]:
    step = BINANCE_INTERVALS[interval]
    cursor = start_ms
    while cursor <= end_ms:
        batch = client.klines(
            symbol,
            interval,
            startTime=cursor,
            endTime=end_ms,
            limit=MAX_BATCH,
        )
        if not batch:
            break

        yield from batch

        last_open_time = batch[-1][0]
        next_cursor = last_open_time + step
        if next_cursor <= cursor:
            raise RuntimeError(
                "Binance API did not advance cursor; aborting to avoid infinite loop."
            )
        cursor = next_cursor
        time.sleep(sleep_seconds)


def write_csv(rows: Iterable[list], output: Path, append: bool) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_utc",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    mode = "a" if append else "w"
    file_exists = output.exists()
    with output.open(mode, newline="", encoding="ascii") as csv_file:
        writer = csv.writer(csv_file)
        if not append or not file_exists or output.stat().st_size == 0:
            writer.writerow(header)

        written = 0
        for row in rows:
            open_time_ms = int(row[0])
            close_time_ms = int(row[6])
            open_time = dt.datetime.utcfromtimestamp(open_time_ms / 1000).isoformat()
            close_time = dt.datetime.utcfromtimestamp(close_time_ms / 1000).isoformat()
            writer.writerow(
                [
                    open_time,
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    close_time,
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                ]
            )
            written += 1

    return written


def read_last_open_time(output: Path) -> int:
    if not output.exists():
        raise FileNotFoundError(output)

    with output.open("r", newline="", encoding="ascii") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        last_row: list[str] | None = None
        for row in reader:
            if row:
                last_row = row

    if not last_row:
        raise ValueError("CSV contains no data rows")

    timestamp = dt.datetime.fromisoformat(last_row[0])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return int(timestamp.timestamp() * 1000)


def normalize_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "").replace("/", "")


def parse_targets(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    symbol = normalize_symbol(args.symbol)

    if not args.target:
        interval = "1s" if args.mode == "second" else args.interval
        mode = args.mode
        return [(symbol, interval, args.output, mode)]

    targets: list[tuple[str, str, Path, str]] = []
    for item in args.target:
        if "=" not in item:
            raise ValueError(f"Invalid target '{item}'. Expected format interval=path.")
        interval_raw, path_raw = item.split("=", 1)
        interval = interval_raw.strip()
        if interval == "1s":
            mode = "second"
        elif interval in BINANCE_INTERVALS:
            mode = "kline"
        else:
            raise ValueError(
                f"Interval '{interval}' is not supported. Choose from 1s or {', '.join(sorted(BINANCE_INTERVALS))}."
            )
        path_str = path_raw.strip()
        if not path_str:
            raise ValueError(f"Output path missing in target '{item}'.")
        path = Path(path_str)
        targets.append((symbol, interval, path, mode))

    if not targets:
        raise ValueError("No valid targets supplied.")
    return targets


def split_symbol(symbol: str) -> tuple[str, str]:
    cleaned = symbol.upper().replace("-", "").replace("/", "")
    for quote in COMMON_QUOTES:
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            return cleaned[: -len(quote)], quote
    return cleaned, ""


def discover_targets(directory: Path, default_symbol: str) -> list[tuple[str, str, Path, str]]:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(
            f"Scan directory '{directory}' does not exist or is not a directory."
        )

    targets: list[tuple[str, str, Path, str]] = []
    default_base, default_quote = split_symbol(default_symbol)
    default_symbol_upper = normalize_symbol(default_symbol)

    suffixes = sorted(list(BINANCE_INTERVALS.keys()) + ["1s"], key=len, reverse=True)

    for csv_path in sorted(directory.glob("*.csv")):
        stem = csv_path.stem
        match_interval = None
        symbol_part = ""
        for interval in suffixes:
            if stem.lower().endswith(f"_{interval.lower()}"):
                match_interval = interval
                symbol_part = stem[: -len(interval) - 1]
                break
        if not match_interval or not symbol_part:
            continue
        clean_symbol = symbol_part.replace("-", "").replace("/", "").upper()

        if not clean_symbol:
            symbol = default_symbol_upper
        elif any(clean_symbol.endswith(quote) for quote in COMMON_QUOTES) and len(clean_symbol) > 3:
            symbol = clean_symbol
        elif default_quote:
            if clean_symbol == default_base:
                symbol = default_symbol_upper
            else:
                symbol = f"{clean_symbol}{default_quote}"
        else:
            symbol = clean_symbol

        if match_interval == "1s":
            mode = "second"
        elif match_interval in BINANCE_INTERVALS:
            mode = "kline"
        else:
            continue

        targets.append((symbol, match_interval, csv_path, mode))

    if not targets:
        raise ValueError(f"No CSV files with interval suffix found in {directory}.")
    return targets


def parse_iso_datetime(value: str) -> dt.datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    moment = dt.datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def iter_agg_trades(
    client: Spot,
    symbol: str,
    start_ms: int,
    end_ms: int,
    sleep_seconds: float,
) -> Iterable[dict]:
    next_from_id: int | None = None
    while True:
        params = {"symbol": symbol, "limit": AGG_TRADE_LIMIT}
        if next_from_id is None:
            params["startTime"] = start_ms
        else:
            params["fromId"] = next_from_id

        trades = client.agg_trades(**params)
        if not trades:
            break

        for trade in trades:
            trade_time = int(trade["T"])
            if trade_time > end_ms:
                return
            yield trade

        last_trade = trades[-1]
        last_time = int(last_trade["T"])
        if last_time >= end_ms:
            break

        next_from_id = int(last_trade["a"]) + 1
        time.sleep(sleep_seconds)


def aggregate_trades_to_seconds(trades: Iterable[dict]) -> Iterable[list]:
    bucket_second: int | None = None
    bucket: dict[str, float | int] | None = None

    for trade in trades:
        trade_second = int(trade["T"]) // 1000
        price = float(trade["p"])
        quantity = float(trade["q"])
        count = int(trade["l"]) - int(trade["f"]) + 1
        taker_buy_volume = 0.0 if trade["m"] else quantity
        taker_buy_quote = 0.0 if trade["m"] else quantity * price

        if bucket_second != trade_second:
            if bucket is not None:
                yield _finalize_second_bucket(bucket)
            bucket_second = trade_second
            open_time_ms = trade_second * 1000
            bucket = {
                "open_time_ms": open_time_ms,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": quantity,
                "close_time_ms": open_time_ms + 999,
                "quote_volume": quantity * price,
                "trade_count": count,
                "taker_buy_volume": taker_buy_volume,
                "taker_buy_quote_volume": taker_buy_quote,
            }
        else:
            bucket["high"] = max(float(bucket["high"]), price)
            bucket["low"] = min(float(bucket["low"]), price)
            bucket["close"] = price
            bucket["volume"] = float(bucket["volume"]) + quantity
            bucket["quote_volume"] = float(bucket["quote_volume"]) + quantity * price
            bucket["trade_count"] = int(bucket["trade_count"]) + count
            bucket["taker_buy_volume"] = float(bucket["taker_buy_volume"]) + taker_buy_volume
            bucket["taker_buy_quote_volume"] = float(bucket["taker_buy_quote_volume"]) + taker_buy_quote

    if bucket is not None:
        yield _finalize_second_bucket(bucket)


def _finalize_second_bucket(bucket: dict[str, float | int]) -> list:
    def fmt(value: float) -> str:
        return f"{value:.8f}"

    return [
        int(bucket["open_time_ms"]),
        fmt(float(bucket["open"])),
        fmt(float(bucket["high"])),
        fmt(float(bucket["low"])),
        fmt(float(bucket["close"])),
        fmt(float(bucket["volume"])),
        int(bucket["close_time_ms"]),
        fmt(float(bucket["quote_volume"])),
        str(int(bucket["trade_count"])),
        fmt(float(bucket["taker_buy_volume"])),
        fmt(float(bucket["taker_buy_quote_volume"])),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.scan_dir and args.target:
        print("Error: --scan-dir cannot be combined with --target.")
        return 1

    try:
        start_override = parse_iso_datetime(args.start) if args.start else None
    except ValueError as exc:
        print(f"Error parsing --start: {exc}")
        return 1

    try:
        end_override = parse_iso_datetime(args.end) if args.end else None
    except ValueError as exc:
        print(f"Error parsing --end: {exc}")
        return 1

    default_symbol = normalize_symbol(args.symbol)

    try:
        if args.scan_dir:
            targets = discover_targets(args.scan_dir, default_symbol)
        else:
            targets = parse_targets(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    end_reference = end_override or dt.datetime.now(dt.timezone.utc)

    client = Spot(api_key=args.api_key, api_secret=args.api_secret)

    total_written = 0

    for symbol, interval, output, mode in targets:
        end_dt = end_override or end_reference
        end_ms = int(end_dt.timestamp() * 1000)

        try:
            if mode == "second":
                if args.append:
                    try:
                        last_open_ms = read_last_open_time(output)
                    except (FileNotFoundError, ValueError) as exc:
                        print(f"Error: cannot append to {output}: {exc}")
                        continue
                    start_ms = last_open_ms + 1000
                    if start_ms >= end_ms:
                        print(f"{output} already up to date; skipping.")
                        continue
                else:
                    base_start = start_override or DEFAULT_SECOND_START
                    if base_start >= end_dt:
                        print(
                            f"Requested end precedes start for {output}; nothing to download."
                        )
                        continue
                    start_ms = int(base_start.timestamp() * 1000)

                trades = iter_agg_trades(
                    client,
                    symbol,
                    start_ms,
                    end_ms,
                    args.sleep,
                )
                rows = aggregate_trades_to_seconds(trades)
                written = write_csv(rows, output, append=args.append)
                dataset_label = "1s"
            else:
                if interval not in BINANCE_INTERVALS:
                    print(f"Skipping unsupported kline interval '{interval}' for {output}.")
                    continue

                if args.append:
                    try:
                        last_open_ms = read_last_open_time(output)
                    except (FileNotFoundError, ValueError) as exc:
                        print(f"Error: cannot append to {output}: {exc}")
                        continue
                    step = BINANCE_INTERVALS[interval]
                    start_ms = last_open_ms + step
                    if start_ms > end_ms:
                        print(f"{output} already up to date; skipping.")
                        continue
                else:
                    base_start = start_override or (end_dt - dt.timedelta(days=args.days))
                    if base_start >= end_dt:
                        print(
                            f"Requested end precedes start for {output}; nothing to download."
                        )
                        continue
                    start_ms = int(base_start.timestamp() * 1000)

                rows = iter_klines(
                    client,
                    symbol,
                    interval,
                    start_ms,
                    end_ms,
                    args.sleep,
                )
                written = write_csv(rows, output, append=args.append)
                dataset_label = interval
        except Exception as exc:  # noqa: BLE001
            print(f"Error while processing {symbol} {interval} -> {output}: {exc}")
            continue

        if written == 0:
            print(
                f"No data returned for {symbol} {dataset_label}; {output} left unchanged."
            )
            continue

        verb = "Appended" if args.append else "Downloaded"
        suffix = "bars" if mode == "second" else "klines"
        print(f"{verb} {written} {symbol} {dataset_label} {suffix} into {output}")
        total_written += written

    if total_written == 0 and args.append:
        print("All referenced CSV files appear up to date.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
