from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


COINGECKO_SIMPLE_PRICE_ENDPOINT = (
	"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={currency}"
)

COINGECKO_MARKET_CHART_ENDPOINT = (
	"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency={currency}&days=max&interval=daily"
)


def _normalize_currency(currency: str) -> str:
	normalized = currency.lower().strip()
	if not normalized:
		raise ValueError("Currency symbol must not be empty.")
	return normalized


def _build_headers(api_key: str | None) -> dict[str, str]:
	headers = {"User-Agent": "DataHub-BitcoinPriceCollector/1.0"}
	if api_key:
		headers["x-cg-demo-api-key"] = api_key
	return headers


def fetch_bitcoin_price(currency: str, api_key: str | None = None) -> float:
	"""Retrieve the latest Bitcoin price for the requested currency."""

	normalized_currency = _normalize_currency(currency)

	request = Request(
		COINGECKO_SIMPLE_PRICE_ENDPOINT.format(currency=normalized_currency),
		headers=_build_headers(api_key),
	)

	try:
		with urlopen(request, timeout=10) as response:
			payload = json.loads(response.read().decode("utf-8"))
	except HTTPError as exc:
		message = f"CoinGecko request failed: HTTP {exc.code}"
		if exc.code == 401:
			message += " (supply a valid API key via --api-key or COINGECKO_API_KEY)"
		raise RuntimeError(message) from exc
	except URLError as exc:
		raise RuntimeError("CoinGecko request failed: network unreachable") from exc

	try:
		price = payload["bitcoin"][normalized_currency]
	except (KeyError, TypeError) as exc:
		raise RuntimeError(
			"CoinGecko response is missing the expected price field"
		) from exc

	if not isinstance(price, (int, float)):
		raise RuntimeError("CoinGecko price field is not numeric")

	return float(price)


def fetch_bitcoin_history(currency: str, api_key: str | None = None) -> list[tuple[dt.datetime, float]]:
	"""Retrieve daily Bitcoin closing prices for the entire history."""

	normalized_currency = _normalize_currency(currency)

	request = Request(
		COINGECKO_MARKET_CHART_ENDPOINT.format(currency=normalized_currency),
		headers=_build_headers(api_key),
	)

	try:
		with urlopen(request, timeout=30) as response:
			payload = json.loads(response.read().decode("utf-8"))
	except HTTPError as exc:
		message = f"CoinGecko history request failed: HTTP {exc.code}"
		if exc.code == 401:
			message += " (supply a valid API key via --api-key or COINGECKO_API_KEY)"
		raise RuntimeError(message) from exc
	except URLError as exc:
		raise RuntimeError("CoinGecko history request failed: network unreachable") from exc

	try:
		price_points = payload["prices"]
	except (KeyError, TypeError) as exc:
		raise RuntimeError(
			"CoinGecko history response is missing the expected price list"
		) from exc

	history: list[tuple[dt.datetime, float]] = []
	for entry in price_points:
		if not isinstance(entry, list) or len(entry) != 2:
			continue
		timestamp_ms, price = entry
		if not isinstance(timestamp_ms, (int, float)) or not isinstance(price, (int, float)):
			continue
		# API returns milliseconds since epoch; convert to UTC-aware datetimes.
		timestamp = dt.datetime.fromtimestamp(timestamp_ms / 1000, tz=dt.timezone.utc)
		history.append((timestamp, float(price)))

	if not history:
		raise RuntimeError("CoinGecko history response contained no usable data")

	return history


def persist_price(price: float, currency: str, output: Path) -> None:
	"""Append the price and timestamp to a CSV file."""

	output.parent.mkdir(parents=True, exist_ok=True)
	timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
	header = ["timestamp_utc", "currency", "price"]
	record = [timestamp, currency.upper(), f"{price:.8f}"]

	file_exists = output.exists()
	with output.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		if not file_exists:
			writer.writerow(header)
		writer.writerow(record)


def persist_history(history: list[tuple[dt.datetime, float]], currency: str, output: Path) -> None:
	"""Write historical price data to a CSV file."""

	output.parent.mkdir(parents=True, exist_ok=True)
	header = ["timestamp_utc", "currency", "price"]

	with output.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(header)
		for timestamp, price in history:
			writer.writerow([timestamp.isoformat(), currency.upper(), f"{price:.8f}"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fetch the latest Bitcoin price or download the full historical series."
	)
	parser.add_argument(
		"-c",
		"--currency",
		default="usd",
		help="Fiat currency code to query (default: usd).",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="Path to the CSV file where data should be written.",
	)
	parser.add_argument(
		"--no-console",
		action="store_true",
		help="Suppress printing data to stdout.",
	)
	parser.add_argument(
		"--history",
		action="store_true",
		help="Download full historical daily prices (requires --output).",
	)
	parser.add_argument(
		"--api-key",
		help="CoinGecko API key (falls back to COINGECKO_API_KEY env var if omitted).",
	)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	api_key = args.api_key or os.getenv("COINGECKO_API_KEY")

	if args.history:
		if not args.output:
			print("Error: --history requires --output", file=sys.stderr)
			return 1
		try:
			history = fetch_bitcoin_history(args.currency, api_key=api_key)
		except (RuntimeError, ValueError) as exc:
			print(f"Error: {exc}", file=sys.stderr)
			return 1

		persist_history(history, args.currency, args.output)

		if not args.no_console:
			print(
				f"Fetched {len(history)} daily price points for {args.currency.upper()} and wrote to {args.output}"
			)
		return 0

	try:
		price = fetch_bitcoin_price(args.currency, api_key=api_key)
	except (RuntimeError, ValueError) as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	if not args.no_console:
		print(f"BTC price in {args.currency.upper()}: {price:.2f}")

	if args.output:
		persist_price(price, args.currency, args.output)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
