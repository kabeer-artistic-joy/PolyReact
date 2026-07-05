#!/usr/bin/env python3
"""
PolyReact
====================================
Two variants:
  --variant predict  Decides direction from the PRIOR window's close, buys
                      instantly at the new window's open.
  --variant react     Watches the NEW window's own real price action for
                      REACT_WAIT_SEC before deciding, then buys at whatever
                      price results.

This version rebuilds the "react" variant around a longer, ROLLING 30-second
observation instead of a single before/after snapshot, plus a cleanliness
check (does the move look like a real trend, or a whipsaw that happened to
net out directional) — the same concept already validated on the predict
variant, applied here because a 30-second window is long enough for a real
whipsaw to fool a simple two-point comparison. This is NOT a second,
independent veto like the old momentum-disagreement gate — it's a quality
check on the SAME signal, and the bot still enters whenever a real move is
found; it does not force a trade when there genuinely isn't one.

The sell mechanism now places a RESTING limit order immediately after
buying, instead of polling the book and reacting after the fact — this was
already proven out and fixed on the Down/Up bots in this project, but had
never been ported back here until now.

IMPORTANT — read before running live:
  Both variants carry the same unbounded loss risk as before: if price never
  reaches the sell trigger, force-exit can cost close to your full stake on
  that trade. Run --dry-run for a meaningful sample before ever using --live.

Usage:
  python spread_bot.py --dry-run --variant react
  python spread_bot.py --live --amount 2 --variant react
"""

import time
import json
import csv
import argparse
import threading
import os
import collections
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────

GAMMA_API   = "https://gamma-api.polymarket.com"
CLOB_API    = "https://clob.polymarket.com"
BINANCE_API = "https://api.binance.com"

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

MARKETS = {
    "btc-updown-5m": "BTC",
    "eth-updown-5m": "ETH",
}

BUY_TARGET_PRICE   = 0.50
BUY_CEILING_PRICE  = 0.52
BUY_TIMEOUT_SEC    = 3.0

SELL_TARGET_PRICE  = 0.60
SELL_FLOOR_PRICE   = 0.58

FORCE_EXIT_SECONDS_LEFT = 30

# ─── "REACT" VARIANT CONFIG ──────────────────────────────────────────────────
REACT_WAIT_SEC        = 30   # raised from 4s per explicit request — long enough that a rolling
                               # observation + cleanliness check matters, not just a single snapshot
REACT_POLL_SEC         = 2    # how often to sample price during the observation window
REACT_MIN_DELTA_PCT    = 0.005  # minimum % move over the observation window before trusting direction
REACT_CLEANLINESS_MIN  = 0.5    # net move vs total high-low range during observation — same concept
                                  # already validated on the predict variant (caught its biggest-signal
                                  # loss). NOT a second independent gate — a quality check on this same
                                  # signal. A real move that's just noisy/whipsawing still gets skipped
                                  # for being unclear, same as a move that's simply too small.
REACT_BUY_TIMEOUT_SEC  = 3.0
REACT_BUY_CEILING_BUFFER = 0.02  # buy ceiling = observed price + this, NOT a fixed absolute ceiling —
                                    # after 30s of real movement, price could be almost anywhere, so a
                                    # fixed ceiling like the old $0.62 no longer makes sense here
REACT_PROFIT_MARGIN    = 0.05    # sell trigger = entry price + this, per explicit request (0.60 -> 0.65 example)

POLL_INTERVAL_FAST = 0.05
POLL_INTERVAL_SLOW = 1.0

# ─── UTILITIES ───────────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def ts_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg, crypto=""):
    prefix = f"[{crypto}] " if crypto else ""
    with _print_lock:
        print(f"[{ts_str()}] {prefix}{msg}", flush=True)

def now_unix():
    return time.time()


def get_window_market(slug_prefix: str, start_ts: int) -> dict | None:
    slug = f"{slug_prefix}-{start_ts}"
    try:
        r = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=3)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        event = data[0]
    except Exception:
        return None

    markets = event.get("markets", [])
    if not markets:
        return None
    market = markets[0]

    try:
        outcomes       = json.loads(market.get("outcomes", "[]"))
        clob_token_ids = json.loads(market.get("clobTokenIds", "[]"))
    except Exception:
        return None

    if len(outcomes) < 2 or len(clob_token_ids) < 2:
        return None

    tokens = dict(zip(outcomes, clob_token_ids))
    if "Down" not in tokens or "Up" not in tokens:
        return None

    return {
        "slug":         slug,
        "crypto":       MARKETS[slug_prefix],
        "start_ts":     start_ts,
        "close_ts":     start_ts + 300,
        "down_token":   tokens["Down"],
        "up_token":     tokens["Up"],
        "condition_id": market.get("conditionId", ""),
        "title":        event.get("title", ""),
    }


def get_order_book(token_id: str) -> dict:
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def best_ask(book: dict):
    asks = book.get("asks", [])
    if not asks:
        return None, None
    cheapest = min(asks, key=lambda a: float(a["price"]))
    return float(cheapest["price"]), float(cheapest["size"])


def best_bid(book: dict):
    bids = book.get("bids", [])
    if not bids:
        return None, None
    highest = max(bids, key=lambda b: float(b["price"]))
    return float(highest["price"]), float(highest["size"])


ENTRY_MIN_DELTA_PCT = 0.05

def get_entry_signal(crypto: str) -> dict:
    """Predict variant's signal — unchanged from before."""
    symbol = SYMBOLS.get(crypto)
    result = {"side": None, "shadow_side": None, "delta_pct": 0.0, "momentum_agrees": False, "reason": ""}
    if not symbol:
        result["reason"] = "no symbol mapping"
        return result
    try:
        r = requests.get(f"{BINANCE_API}/api/v3/klines", params={"symbol": symbol, "interval": "1m", "limit": 5}, timeout=3)
        r.raise_for_status()
        candles = r.json()
    except Exception as e:
        result["reason"] = f"binance error: {e}"
        return result
    if len(candles) < 5:
        result["reason"] = "insufficient candle data"
        return result
    window_open    = float(candles[0][1])
    current_price  = float(candles[-1][4])
    prev_close     = float(candles[-2][4])
    delta_pct      = abs(current_price - window_open) / window_open * 100
    magnitude_dir  = "Up" if current_price > window_open else "Down"
    momentum_dir   = "Up" if current_price > prev_close else "Down"
    result["delta_pct"]       = round(delta_pct, 4)
    result["momentum_agrees"] = (magnitude_dir == momentum_dir)
    result["shadow_side"]     = magnitude_dir
    if delta_pct < ENTRY_MIN_DELTA_PCT:
        result["reason"] = f"delta {delta_pct:.4f}% < {ENTRY_MIN_DELTA_PCT}% — too weak to trust"
        return result
    result["side"]   = magnitude_dir
    result["reason"] = f"delta {delta_pct:.4f}% (momentum {'agreed' if result['momentum_agrees'] else 'disagreed'}, no longer gating)"
    return result


def shadow_track_window(crypto: str, token: str, close_ts: float, window_open_time: float, budget: float) -> dict:
    deadline = window_open_time + BUY_TIMEOUT_SEC
    bought_price = None
    while now_unix() < deadline:
        book = get_order_book(token)
        price, _ = best_ask(book)
        if price is not None and price <= BUY_CEILING_PRICE:
            bought_price = price
            break
        time.sleep(POLL_INTERVAL_FAST)
    if bought_price is None:
        return {"result": "shadow_missed", "pnl": 0.0}
    shares = max(1, round(budget / bought_price))
    while True:
        seconds_left = close_ts - now_unix()
        if seconds_left <= FORCE_EXIT_SECONDS_LEFT:
            break
        book = get_order_book(token)
        price, size = best_bid(book)
        if price is not None and price >= SELL_FLOOR_PRICE and size >= shares:
            pnl = round((price - bought_price) * shares, 4)
            return {"result": "shadow_sold", "sell_price": price, "pnl": pnl}
        time.sleep(POLL_INTERVAL_SLOW)
    book = get_order_book(token)
    price, _ = best_bid(book)
    if price is None:
        return {"result": "shadow_no_bids", "sell_price": None, "pnl": -round(bought_price * shares, 4)}
    pnl = round((price - bought_price) * shares, 4)
    return {"result": "shadow_exited", "sell_price": price, "pnl": pnl}


def get_binance_price(symbol: str) -> float | None:
    try:
        r = requests.get(f"{BINANCE_API}/api/v3/ticker/price", params={"symbol": symbol}, timeout=2)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def collect_react_signal(symbol: str, wait_sec: float, poll_sec: float) -> dict:
    """
    Replaces the old single before/after snapshot. Actively polls price every
    poll_sec throughout the observation window, building a real rolling
    history, then computes move + cleanliness together — same concept
    already validated on the predict variant, applied here because a 30s
    window is long enough for a real whipsaw to fool a two-point comparison.
    """
    result = {"side": None, "move_pct": 0.0, "cleanliness": 0.0, "reason": ""}
    prices = []
    deadline = now_unix() + wait_sec
    while now_unix() < deadline:
        p = get_binance_price(symbol)
        if p is not None:
            prices.append(p)
        time.sleep(poll_sec)

    if len(prices) < 3:
        result["reason"] = "insufficient price samples collected during observation"
        return result

    oldest, current = prices[0], prices[-1]
    move_pct = (current - oldest) / oldest * 100
    price_range = max(prices) - min(prices)
    net_move = abs(current - oldest)
    cleanliness = round(net_move / price_range, 4) if price_range > 0 else 1.0

    result["move_pct"] = round(move_pct, 4)
    result["cleanliness"] = cleanliness

    if abs(move_pct) < REACT_MIN_DELTA_PCT:
        result["reason"] = f"move {move_pct:+.4f}% < {REACT_MIN_DELTA_PCT}% — too weak to trust"
        return result
    if cleanliness < REACT_CLEANLINESS_MIN:
        result["reason"] = f"move {move_pct:+.4f}% OK, but cleanliness {cleanliness:.2f} < {REACT_CLEANLINESS_MIN} — too much whipsaw"
        return result

    result["side"] = "Up" if move_pct > 0 else "Down"
    result["reason"] = f"move {move_pct:+.4f}%, cleanliness {cleanliness:.2f} — both pass"
    return result


def next_window_start(now: float) -> int:
    return int((now // 300) + 1) * 300


# ─── PERSISTENT CSV LOG ──────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "bot_name", "mode", "crypto", "slug",
    "target_side", "signal_delta_pct", "signal_cleanliness", "signal_reason",
    "buy_result", "buy_price", "buy_shares", "buy_elapsed_ms",
    "num_opportunities", "sell_result", "sell_price",
    "pnl_usd", "notes",
]

SHADOW_CSV_FIELDS = [
    "timestamp", "bot_name", "crypto", "slug", "skip_reason",
    "shadow_side", "delta_pct", "shadow_result", "shadow_sell_price", "shadow_pnl",
]

class ShadowLogger:
    def __init__(self, bot_name: str):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow_log.csv")
        self.lock = threading.Lock()
        self.bot_name = bot_name
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(SHADOW_CSV_FIELDS)

    def write(self, row: dict):
        row = {**{k: "" for k in SHADOW_CSV_FIELDS}, **row}
        with self.lock:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in SHADOW_CSV_FIELDS])


class TradeLogger:
    def __init__(self, bot_name: str):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.csv")
        self.lock = threading.Lock()
        self.bot_name = bot_name
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def write(self, row: dict):
        row = {**{k: "" for k in CSV_FIELDS}, **row}
        with self.lock:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in CSV_FIELDS])


# ─── CORE BOT ────────────────────────────────────────────────────────────────

class SpreadBot:
    def __init__(self, dry_run: bool, amount: float, variant: str = "predict"):
        self.dry_run  = dry_run
        self.amount   = amount
        self.variant  = variant
        self.bot_name = os.getenv("BOT_NAME", "spread_bot")
        self.mode_str = "dry_run" if dry_run else "live"
        self.stop_event = threading.Event()
        self.trades = []
        self.trades_lock = threading.Lock()
        self.logger = TradeLogger(self.bot_name)
        self.shadow_logger = ShadowLogger(self.bot_name)

        self.client = None
        if not dry_run:
            self._init_client()

        log("=" * 70)
        log(f"PolyReact | {self.mode_str.upper()} | ${amount:.2f}/trade | bot_name={self.bot_name}")
        if self.variant == "react":
            log(f"Variant: REACT | observe {REACT_WAIT_SEC}s (rolling, poll every {REACT_POLL_SEC}s), "
                f"then buy within {REACT_BUY_TIMEOUT_SEC}s at observed price + ${REACT_BUY_CEILING_BUFFER}")
            log(f"Sell: entry price + ${REACT_PROFIT_MARGIN} margin (resting order) | force-exit last {FORCE_EXIT_SECONDS_LEFT}s")
            log(f"Min move: {REACT_MIN_DELTA_PCT}% | Min cleanliness: {REACT_CLEANLINESS_MIN}")
        else:
            log(f"Variant: PREDICT | Buy: target ${BUY_TARGET_PRICE} ceiling ${BUY_CEILING_PRICE} timeout {BUY_TIMEOUT_SEC}s")
            log(f"Sell: target ${SELL_TARGET_PRICE} floor ${SELL_FLOOR_PRICE} | force-exit last {FORCE_EXIT_SECONDS_LEFT}s")
            log(f"Min delta to trust direction: {ENTRY_MIN_DELTA_PCT}%")
        log(f"Trade log: {self.logger.path}")
        log("=" * 70)

    def _init_client(self):
        from py_clob_client_v2 import ClobClient, AssetType, BalanceAllowanceParams
        signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "3"))
        self.client = ClobClient(
            host=CLOB_API, key=os.environ["POLY_PRIVATE_KEY"], chain_id=137,
            signature_type=signature_type, funder=os.environ["POLY_PROXY_WALLET"],
        )
        self.client.set_api_creds(self.client.create_or_derive_api_key())
        self.client.update_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, signature_type=signature_type,
        ))

    # ── BUY (shared by both variants) ───────────────────────────────────────

    def _attempt_buy(self, market: dict, window_open_time: float, ceiling: float = None, timeout: float = None) -> dict:
        ceiling = BUY_CEILING_PRICE if ceiling is None else ceiling
        timeout = BUY_TIMEOUT_SEC if timeout is None else timeout
        crypto = market["crypto"]
        token  = market["target_token"]
        MIN_SHARES = 5  # CONFIRMED via a real live API error: "Size (4) lower than the minimum: 5"

        if self.dry_run:
            deadline = window_open_time + timeout
            last_seen_price = None
            while now_unix() < deadline:
                book = get_order_book(token)
                price, size = best_ask(book)
                if price is not None:
                    last_seen_price = price
                elapsed_ms = (now_unix() - window_open_time) * 1000
                if price is not None and price <= ceiling:
                    shares = max(MIN_SHARES, round(self.amount / price))
                    log(f"[DRY] BUY would fill: ask ${price:.3f} (size {size}) at {elapsed_ms:.0f}ms", crypto)
                    return {"result": "bought", "price": price, "shares": shares, "elapsed_ms": elapsed_ms}
                time.sleep(POLL_INTERVAL_FAST)
            elapsed_ms = (now_unix() - window_open_time) * 1000
            price_info = f"last real ask seen was ${last_seen_price:.3f}" if last_seen_price is not None else "no asks seen at all"
            log(f"[DRY] BUY missed: no ask <= ${ceiling} within {timeout}s ({price_info})", crypto)
            return {"result": "missed", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}

        from py_clob_client_v2 import OrderArgsV2, Side, OrderType, OrderPayload
        size = max(MIN_SHARES, round(self.amount / ceiling))
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=ceiling, size=size, side=Side.BUY),
                order_type=OrderType.GTC,
            )
        except Exception as e:
            log(f"❌ BUY order failed to submit: {e}", crypto)
            return {"result": "error", "price": None, "shares": 0, "elapsed_ms": 0}

        order_id = resp.get("orderID", "")
        deadline = window_open_time + timeout
        last_known_size = 0.0
        while now_unix() < deadline:
            try:
                detail = self.client.get_order(order_id)
            except Exception:
                detail = None
            if detail is None:
                break
            try:
                current_size = float(detail.get("size_matched", 0))
                if current_size > last_known_size:
                    last_known_size = current_size
                    log(f"BUY fill update: {last_known_size} shares matched so far", crypto)
            except (TypeError, ValueError):
                pass
            time.sleep(0.25)

        elapsed_ms = (now_unix() - window_open_time) * 1000
        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
        except Exception:
            pass

        if last_known_size <= 0:
            try:
                from py_clob_client_v2 import AssetType, BalanceAllowanceParams
                bal_resp = self.client.get_balance_allowance(BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token,
                    signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "3")),
                ))
                real_balance = float(bal_resp.get("balance", 0)) / 1_000_000
                if real_balance >= 0.5:
                    log(f"⚠️ get_order() showed no fill, but balance check found {real_balance} shares — correcting course", crypto)
                    return {"result": "bought", "price": ceiling, "shares": real_balance, "elapsed_ms": elapsed_ms}
            except Exception as e:
                log(f"⚠️ Final balance safety-check failed ({e})", crypto)
            log(f"❌ BUY timed out with no confirmed fill after {timeout}s ({elapsed_ms:.0f}ms)", crypto)
            return {"result": "missed", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}

        log(f"✅ BUY confirmed: {last_known_size} total shares matched (ceiling was ${ceiling}), "
            f"order {order_id[:16]}... ({elapsed_ms:.0f}ms)", crypto)
        return {"result": "bought", "price": ceiling, "shares": last_known_size, "elapsed_ms": elapsed_ms}

    # ── SELL (shared by both variants) — now places a RESTING order ────────

    def _watch_for_sell(self, market: dict, buy_info: dict, sell_floor: float) -> dict:
        crypto    = market["crypto"]
        token     = market["target_token"]
        close_ts  = market["close_ts"]
        buy_price = buy_info["price"]
        raw_shares = buy_info["shares"]
        shares = int(raw_shares)
        if shares != raw_shares:
            log(f"⚠️ Buy partially filled: held {raw_shares}, flooring to {shares} whole shares to keep sells valid", crypto)
        if shares < 1:
            log("⚠️ Partial fill left less than 1 whole share — forcing immediate exit", crypto)
            exit_result = self._force_exit(token, raw_shares, crypto)
            pnl = -round(buy_price * raw_shares, 4)
            return {**exit_result, "opportunities": 0, "pnl_usd": pnl, "notes": "sub-1-share partial fill"}

        log(f"Sell trigger: ${sell_floor} (bought ${buy_price})", crypto)

        if self.dry_run:
            while True:
                if close_ts - now_unix() <= FORCE_EXIT_SECONDS_LEFT:
                    break
                book = get_order_book(token)
                price, size = best_bid(book)
                if price is not None and price >= sell_floor and size >= shares:
                    pnl = round((price - buy_price) * shares, 4)
                    log(f"[DRY] SELL would fill: bid ${price:.3f}", crypto)
                    return {"result": "sold", "price": price, "opportunities": 1, "pnl_usd": pnl, "notes": "sold"}
                time.sleep(POLL_INTERVAL_SLOW)
            exit_result = self._force_exit(token, shares, crypto)
            pnl = round((exit_result["price"] - buy_price) * shares, 4) if exit_result["price"] is not None else -round(buy_price * shares, 4)
            return {**exit_result, "opportunities": 0, "pnl_usd": pnl, "notes": "force-exit"}

        # LIVE: rest a limit sell immediately instead of polling and reacting —
        # proven fix from the Down/Up bots, ported back here.
        from py_clob_client_v2 import AssetType, BalanceAllowanceParams
        try:
            self.client.update_balance_allowance(BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token,
                signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "3")),
            ))
        except Exception as e:
            log(f"⚠️ Could not sync conditional balance ({e})", crypto)

        from py_clob_client_v2 import OrderArgsV2, Side, OrderType, OrderPayload
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=sell_floor, size=shares, side=Side.SELL),
                order_type=OrderType.GTC,
            )
            sell_order_id = resp.get("orderID", "")
            log(f"Resting SELL placed at ${sell_floor}, order {sell_order_id[:16]}...", crypto)
        except Exception as e:
            log(f"⚠️ Could not place resting sell ({e}) — forcing exit immediately", crypto)
            exit_result = self._force_exit(token, shares, crypto)
            pnl = round((exit_result["price"] - buy_price) * shares, 4) if exit_result["price"] is not None else -round(buy_price * shares, 4)
            return {**exit_result, "opportunities": 0, "pnl_usd": pnl, "notes": "resting sell placement failed"}

        last_known_sold = 0.0
        while True:
            if close_ts - now_unix() <= FORCE_EXIT_SECONDS_LEFT:
                break
            try:
                detail = self.client.get_order(sell_order_id)
            except Exception:
                detail = None
            if detail is None:
                last_known_sold = shares
                break
            try:
                current_sold = float(detail.get("size_matched", 0))
                if current_sold > last_known_sold:
                    last_known_sold = current_sold
                    log(f"SELL fill update: {last_known_sold}/{shares} shares sold so far", crypto)
            except (TypeError, ValueError):
                pass
            time.sleep(POLL_INTERVAL_SLOW)

        if last_known_sold >= shares:
            pnl = round((sell_floor - buy_price) * shares, 4)
            return {"result": "sold", "price": sell_floor, "opportunities": 1, "pnl_usd": pnl, "notes": "sold via resting order"}

        try:
            self.client.cancel_order(OrderPayload(orderID=sell_order_id))
        except Exception:
            pass
        remaining = round(shares - last_known_sold, 4)
        if remaining < 1:
            pnl = round((sell_floor - buy_price) * last_known_sold, 4)
            return {"result": "sold", "price": sell_floor, "opportunities": 1, "pnl_usd": pnl, "notes": "dust remainder left"}
        exit_result = self._force_exit(token, int(remaining), crypto)
        sold_pnl = round((sell_floor - buy_price) * last_known_sold, 4)
        exit_pnl = round((exit_result["price"] - buy_price) * int(remaining), 4) if exit_result["price"] is not None else -round(buy_price * int(remaining), 4)
        return {**exit_result, "opportunities": 1, "pnl_usd": round(sold_pnl + exit_pnl, 4), "notes": "partial via resting order + force-exit"}

    def _force_exit(self, token: str, shares: float, crypto: str) -> dict:
        if self.dry_run:
            book = get_order_book(token)
            price, size = best_bid(book)
            if price is None:
                log("[DRY] No bids at all for force-exit — would be a total loss this window", crypto)
                return {"result": "no_bids", "price": None}
            log(f"[DRY] Force-exit would fill at ${price:.3f}", crypto)
            return {"result": "exited", "price": price}

        from py_clob_client_v2 import MarketOrderArgsV2, Side, OrderType
        try:
            resp = self.client.create_and_post_market_order(
                MarketOrderArgsV2(token_id=token, amount=shares, side=Side.SELL),
                order_type=OrderType.FAK,
            )
        except Exception as e:
            log(f"⚠️ Force-exit order failed: {e}", crypto)
            return {"result": "error", "price": None}
        status = str(resp.get("status", "")).lower()
        if status == "matched":
            try:
                cost = float(resp.get("makingAmount", 0)) / 1_000_000
                exit_price = round(cost / shares, 4) if shares else None
            except Exception:
                exit_price = None
            log(f"✅ Force-exit matched, order {resp.get('orderID','')[:16]}...", crypto)
            return {"result": "exited", "price": exit_price}
        log("⚠️ Force-exit did not match — position may still be open, check account manually", crypto)
        return {"result": "unmatched", "price": None}

    # ── PREDICT VARIANT WINDOW HANDLER ──────────────────────────────────────

    def _handle_window(self, slug_prefix: str, start_ts: int):
        crypto = MARKETS[slug_prefix]
        while now_unix() < start_ts - 1:
            time.sleep(0.2)
        while now_unix() < start_ts:
            time.sleep(0.005)
        window_open_time = now_unix()

        market = None
        find_deadline = window_open_time + 3
        while now_unix() < find_deadline:
            market = get_window_market(slug_prefix, start_ts)
            if market:
                break
            time.sleep(0.1)
        if not market:
            log(f"Could not find market for window starting {start_ts} within 3s of open — skipping this window", crypto)
            return

        signal = get_entry_signal(crypto)
        log(f"Entry signal: {signal['reason']}", crypto)
        if signal["side"] is None:
            log(f"Skipping this window — shadow-tracking what {signal['shadow_side']} would have done", crypto)
            shadow_token = market["down_token"] if signal["shadow_side"] == "Down" else market["up_token"]
            shadow = shadow_track_window(crypto, shadow_token, market["close_ts"], window_open_time, self.amount)
            self.shadow_logger.write({
                "timestamp": ts_str(), "bot_name": self.bot_name, "crypto": crypto,
                "slug": market["slug"], "skip_reason": signal["reason"],
                "shadow_side": signal["shadow_side"], "delta_pct": signal["delta_pct"],
                "shadow_result": shadow["result"], "shadow_sell_price": shadow.get("sell_price"),
                "shadow_pnl": shadow["pnl"],
            })
            log(f"SHADOW RESULT: {shadow['result']} | pnl={shadow['pnl']:+.2f} (not real — this window was skipped)", crypto)
            return

        target_side = signal["side"]
        market["target_side"]  = target_side
        market["target_token"] = market["down_token"] if target_side == "Down" else market["up_token"]

        buy_info = self._attempt_buy(market, window_open_time)
        row = {
            "timestamp": ts_str(), "bot_name": self.bot_name, "mode": self.mode_str, "crypto": crypto,
            "target_side": target_side, "signal_delta_pct": signal["delta_pct"], "signal_cleanliness": "",
            "signal_reason": signal["reason"], "slug": market["slug"], "buy_result": buy_info["result"],
            "buy_price": buy_info["price"], "buy_shares": buy_info["shares"],
            "buy_elapsed_ms": round(buy_info["elapsed_ms"], 1),
        }
        if buy_info["result"] != "bought":
            row.update({"sell_result": "n/a", "sell_price": "", "pnl_usd": 0, "notes": "no buy fill", "num_opportunities": 0})
            self._record(row)
            return

        sell_info = self._watch_for_sell(market, buy_info, SELL_FLOOR_PRICE)
        row.update({
            "sell_result": sell_info["result"], "sell_price": sell_info["price"],
            "num_opportunities": sell_info["opportunities"], "pnl_usd": sell_info["pnl_usd"],
            "notes": sell_info["notes"],
        })
        self._record(row)

    def _record(self, row: dict):
        with self.trades_lock:
            self.trades.append(row)
        self.logger.write(row)
        pnl = row.get("pnl_usd", 0)
        sign = "+" if isinstance(pnl, (int, float)) and pnl >= 0 else ""
        log(f"RECORDED: buy={row['buy_result']}@{row['buy_price']} | sell={row['sell_result']}@{row['sell_price']} "
            f"| opportunities={row['num_opportunities']} | pnl={sign}${pnl}", row["crypto"])

    # ── REACT VARIANT WINDOW HANDLER ────────────────────────────────────────

    def _handle_window_react(self, slug_prefix: str, start_ts: int):
        crypto = MARKETS[slug_prefix]
        symbol = SYMBOLS.get(crypto)

        while now_unix() < start_ts - 1:
            time.sleep(0.2)
        while now_unix() < start_ts:
            time.sleep(0.005)
        window_open_time = now_unix()

        market = None
        find_deadline = window_open_time + 3
        while now_unix() < find_deadline:
            market = get_window_market(slug_prefix, start_ts)
            if market:
                break
            time.sleep(0.1)
        if not market:
            log(f"Could not find market for window starting {start_ts} within 3s of open — skipping this window", crypto)
            return

        log(f"Observing for {REACT_WAIT_SEC}s (rolling, poll every {REACT_POLL_SEC}s)...", crypto)
        signal = collect_react_signal(symbol, REACT_WAIT_SEC, REACT_POLL_SEC)
        log(f"React signal: {signal['reason']}", crypto)
        if signal["side"] is None:
            log("Skipping this window — no confident signal", crypto)
            return

        direction = signal["side"]
        market["target_side"]  = direction
        market["target_token"] = market["down_token"] if direction == "Down" else market["up_token"]

        book = get_order_book(market["target_token"])
        observed_price, _ = best_ask(book)
        if observed_price is None:
            log("Could not read current price to set buy ceiling — skipping this window", crypto)
            return
        ceiling = round(observed_price + REACT_BUY_CEILING_BUFFER, 4)

        buy_start = now_unix()
        buy_info = self._attempt_buy(market, buy_start, ceiling=ceiling, timeout=REACT_BUY_TIMEOUT_SEC)

        row = {
            "timestamp": ts_str(), "bot_name": self.bot_name, "mode": self.mode_str, "crypto": crypto,
            "target_side": direction, "signal_delta_pct": signal["move_pct"], "signal_cleanliness": signal["cleanliness"],
            "signal_reason": signal["reason"], "slug": market["slug"], "buy_result": buy_info["result"],
            "buy_price": buy_info["price"], "buy_shares": buy_info["shares"],
            "buy_elapsed_ms": round(buy_info["elapsed_ms"], 1),
        }
        if buy_info["result"] != "bought":
            row.update({"sell_result": "n/a", "sell_price": "", "pnl_usd": 0, "notes": "no buy fill", "num_opportunities": 0})
            self._record(row)
            return

        sell_floor = round(buy_info["price"] + REACT_PROFIT_MARGIN, 4)
        sell_info = self._watch_for_sell(market, buy_info, sell_floor)
        row.update({
            "sell_result": sell_info["result"], "sell_price": sell_info["price"],
            "num_opportunities": sell_info["opportunities"], "pnl_usd": sell_info["pnl_usd"],
            "notes": sell_info["notes"] + f" (sell floor was ${sell_floor})",
        })
        self._record(row)

    # ── ASSET LOOP ───────────────────────────────────────────────────────────

    def _asset_loop(self, slug_prefix: str):
        crypto = MARKETS[slug_prefix]
        while not self.stop_event.is_set():
            start_ts = next_window_start(now_unix())
            wake_at  = start_ts - 10
            while now_unix() < wake_at and not self.stop_event.is_set():
                time.sleep(1)
            if self.stop_event.is_set():
                break
            log(f"Waking for window starting {datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%H:%M:%S')} UTC", crypto)
            try:
                if self.variant == "react":
                    self._handle_window_react(slug_prefix, start_ts)
                else:
                    self._handle_window(slug_prefix, start_ts)
            except Exception as e:
                log(f"⚠️ Unhandled error this window: {e}", crypto)
            time.sleep(2)

    def run(self):
        threads = [threading.Thread(target=self._asset_loop, args=(prefix,), daemon=True) for prefix in MARKETS]
        for t in threads:
            t.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log("Stopping...")
            self.stop_event.set()
            self._print_summary()

    def _print_summary(self):
        log("-" * 70)
        with self.trades_lock:
            trades = list(self.trades)
        log(f"SUMMARY — {len(trades)} windows attempted")
        bought = [t for t in trades if t["buy_result"] == "bought"]
        sold   = [t for t in bought if t["sell_result"] == "sold"]
        forced = [t for t in bought if t["sell_result"] in ("exited", "no_bids", "unmatched")]
        total_pnl = sum(float(t["pnl_usd"] or 0) for t in trades)
        log(f"  Buy fills: {len(bought)}/{len(trades)}")
        log(f"  Sold: {len(sold)}")
        log(f"  Force-exited: {len(forced)}")
        log(f"  Total PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}")
        log("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyReact")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--variant", choices=["predict", "react"], default="predict")
    args = parser.parse_args()

    bot = SpreadBot(dry_run=args.dry_run, amount=args.amount, variant=args.variant)
    bot.run()
