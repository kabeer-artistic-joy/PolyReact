#!/usr/bin/env python3
"""
Polymarket Down-Spread Scalper Bot
====================================
A completely separate strategy from the delta/momentum bot. This bot does NOT
use price direction signals, delta thresholds, momentum, or ATR. It tests one
specific, narrow hypothesis:

    At the instant a new 5-minute Up/Down window opens, buy the DOWN token as
    cheaply as possible (target 50c, ceiling 52c). Then try to sell it back
    for a profit once the market price moves toward 60c (accepting as low as
    58c). If neither the buy nor the sell mechanics work out in time, exit
    for whatever is available rather than hold to resolution.

This is a SPREAD/VOLATILITY play, not a directional one — it does not care
which side (Up or Down) ultimately wins. It only cares whether the DOWN price
touches the 58-60c range at some point during the window before you're forced
to exit in the closing seconds.

IMPORTANT — read before running live:
  A loss in this bot is NOT bounded like the delta bot's PRICE_MAX-protected
  entries. If DOWN never reaches 58c during the window, this bot force-exits
  in the final FORCE_EXIT_SECONDS at whatever price is available — which could
  be far below your buy price. A single bad window can cost close to your
  full stake. This has NOT been validated with real trade data yet. Run
  --dry-run for a meaningful sample before ever using --live.

Modes:
  --dry-run   No real orders. Polls the REAL, LIVE order book throughout each
              window and computes what WOULD have happened (fill/miss on the
              buy, each 58-60c opportunity, and the eventual exit) using real
              market depth data — not simulated or assumed prices.
  --live      Places real limit orders per the exact logic below, using your
              Polymarket deposit wallet (same auth mechanism as the other bot).

Usage:
  python spread_bot.py --dry-run
  python spread_bot.py --live --amount 2
"""

import time
import json
import csv
import argparse
import threading
import os
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

BUY_TARGET_PRICE   = 0.50   # what we hope to pay
BUY_CEILING_PRICE  = 0.52   # max we're willing to pay — this is the actual limit price used
BUY_TIMEOUT_SEC    = 3.0    # cancel the buy attempt if unfilled after this long. Raised from 2.0 based on real fill-time data: avg fill was 1787ms, and the one observed miss came in at 2322ms — just 322ms late. 3s is the number this specific evidence supports; a jump to 5s was not.

SELL_TARGET_PRICE  = 0.60   # what we hope to sell at
SELL_FLOOR_PRICE   = 0.58   # minimum acceptable profitable sell

FORCE_EXIT_SECONDS_LEFT = 60  # in the final N seconds of the window, exit at any price if still holding

# ─── "REACT" VARIANT CONFIG ──────────────────────────────────────────────────
# An alternative to the "predict" variant above. Instead of deciding direction
# from the PRIOR window and buying instantly at open, this waits a few seconds
# INTO the new window, watches its own real price action to decide direction,
# then buys at whatever price results — targeting a fixed profit MARGIN from
# that actual entry price, rather than fixed absolute floor/target prices.
REACT_WAIT_SEC       = 5     # observe the new window for this long before deciding direction
REACT_BUY_TIMEOUT_SEC = 2    # then buy within this long, same cancel-if-unfilled logic
REACT_BUY_CEILING    = 0.58  # wider than BUY_CEILING_PRICE — price may have already drifted further from 50c by the time we act
REACT_PROFIT_MARGIN  = 0.07  # target this much profit per share above actual entry price (lowered from 0.09 to split the new 0.06-0.08 range, since waiting longer to enter means less exposure to the full move)
REACT_MIN_DELTA_PCT  = 0.005 # minimum % move required in the 3s observation before trusting it as real direction, not noise.
                              # Calibrated from your own BTC observation (~$2-5 real moves) — lands at ~$3.10 on BTC,
                              # ~$0.09 on ETH (same relative move, different price scale — not a typo).

POLL_INTERVAL_FAST = 0.05   # tight poll interval used right at window open (seconds)
POLL_INTERVAL_SLOW = 1.0    # normal poll interval while watching for a 58-60c opportunity

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
    """
    Find the Up/Down market for a window starting at start_ts. Returns both
    outcome token IDs by name (unlike the delta bot, which only tracked the
    'winning' side — this bot always wants the Down token specifically).
    """
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
    """Raw public order book fetch — no auth required. Used for both dry-mode
    simulation and live-mode depth checks."""
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def best_ask(book: dict):
    """Returns (price, size) of the cheapest ask, or (None, None)."""
    asks = book.get("asks", [])
    if not asks:
        return None, None
    cheapest = min(asks, key=lambda a: float(a["price"]))
    return float(cheapest["price"]), float(cheapest["size"])


def best_bid(book: dict):
    """Returns (price, size) of the highest bid, or (None, None)."""
    bids = book.get("bids", [])
    if not bids:
        return None, None
    highest = max(bids, key=lambda b: float(b["price"]))
    return float(highest["price"]), float(highest["size"])


ENTRY_MIN_DELTA_PCT = 0.05  # % move required from the ending window's own open
                             # before its direction is trusted as a signal for
                             # the NEXT window. Starting point is informed by
                             # real validated data from the delta bot (0.06%
                             # eliminated its one real loss while preserving
                             # 68/68 wins in that sample) — but this is a
                             # DIFFERENT question (predicting the next window
                             # vs filtering entries within the same window),
                             # so this value still needs its own validation
                             # against this bot's real dry-run data.


def get_entry_signal(crypto: str) -> dict:
    """
    Looks at the window that's about to close: how far has price moved from
    THAT window's own open (magnitude, as a %), and does the last ~2 minutes
    of momentum agree with that direction? Both must agree, and the
    magnitude must clear ENTRY_MIN_DELTA_PCT, or this window is skipped
    entirely — this bot is meant to take FEWER, more selective trades, not
    force an entry into every window.

    Returns a dict with the decision and the raw numbers, for full logging —
    same transparency the delta bot's skip/enter reasoning always had.
    """
    symbol = SYMBOLS.get(crypto)
    result = {"side": None, "shadow_side": None, "delta_pct": 0.0, "momentum_agrees": False, "reason": ""}
    if not symbol:
        result["reason"] = "no symbol mapping"
        return result

    try:
        # Last 5 one-minute candles ≈ the closing 5-minute window's own span,
        # giving us that window's own open price to measure magnitude from —
        # the same delta_pct concept as the delta bot, applied here to predict
        # the NEXT window instead of filtering entries within this one.
        r = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 5},
            timeout=3,
        )
        r.raise_for_status()
        candles = r.json()
    except Exception as e:
        result["reason"] = f"binance error: {e}"
        return result

    if len(candles) < 5:
        result["reason"] = "insufficient candle data"
        return result

    window_open    = float(candles[0][1])   # open of the oldest candle in this span
    current_price  = float(candles[-1][4])  # close of the most recent candle
    prev_close     = float(candles[-2][4])
    delta_pct      = abs(current_price - window_open) / window_open * 100
    magnitude_dir  = "Up" if current_price > window_open else "Down"
    momentum_dir   = "Up" if current_price > prev_close else "Down"

    result["delta_pct"]       = round(delta_pct, 4)
    result["momentum_agrees"] = (magnitude_dir == momentum_dir)  # still computed and logged, just no longer gates the decision
    result["shadow_side"]     = magnitude_dir  # always set, regardless of confidence — used for shadow-tracking skipped windows

    if delta_pct < ENTRY_MIN_DELTA_PCT:
        result["reason"] = f"delta {delta_pct:.4f}% < {ENTRY_MIN_DELTA_PCT}% — too weak to trust"
        return result

    # Momentum agreement removed as a gate per explicit request — decision is
    # now purely delta-magnitude-based. Not yet validated whether this
    # helps or hurts; momentum_agrees is still logged so this can be
    # checked against real outcomes later even though it no longer blocks entry.
    result["side"]   = magnitude_dir
    result["reason"] = f"delta {delta_pct:.4f}% (momentum {'agreed' if result['momentum_agrees'] else 'disagreed'}, no longer gating)"
    return result


def shadow_track_window(crypto: str, token: str, close_ts: float, window_open_time: float, budget: float) -> dict:
    """
    Pure observation of a SKIPPED window — never places a real order,
    regardless of --dry-run or --live. Answers the actual question raised
    when momentum/delta disagreement causes a skip: was that skip a good
    decision or did it cost a winning trade? Uses the same real order-book
    polling as dry-run, just applied to windows the entry signal rejected.
    """
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
    """Current price, single call — lighter than klines, used by the react variant's before/after comparison."""
    try:
        r = requests.get(f"{BINANCE_API}/api/v3/ticker/price", params={"symbol": symbol}, timeout=2)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def get_react_direction(crypto: str, price_at_open: float) -> tuple[str | None, float]:
    """
    Compares the current price to price_at_open (captured right when the
    window opened) to decide Up or Down, for the "react" variant. Requires
    the move to clear REACT_MIN_DELTA_PCT, ruling out sub-cent noise being
    treated with the same confidence as a real move. Returns (direction, pct)
    — direction is None if data is unavailable, the move is flat, or it's
    below the noise threshold.
    """
    symbol = SYMBOLS.get(crypto)
    if not symbol or price_at_open is None:
        return None, 0.0
    current = get_binance_price(symbol)
    if current is None or current == price_at_open:
        return None, 0.0
    pct = abs(current - price_at_open) / price_at_open * 100
    if pct < REACT_MIN_DELTA_PCT:
        return None, pct
    return ("Up" if current > price_at_open else "Down"), pct


def next_window_start(now: float) -> int:
    """Returns the unix timestamp of the next 5-minute boundary."""
    return int((now // 300) + 1) * 300


# ─── PERSISTENT CSV LOG ──────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "bot_name", "mode", "crypto", "slug",
    "target_side", "signal_delta_pct", "signal_reason",
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
        log(f"Down-Spread Scalper | {self.mode_str.upper()} | ${amount:.2f}/trade | bot_name={self.bot_name}")
        if self.variant == "react":
            log(f"Variant: REACT | wait {REACT_WAIT_SEC}s to observe, then buy within {REACT_BUY_TIMEOUT_SEC}s, ceiling ${REACT_BUY_CEILING}")
            log(f"Sell: entry price + ${REACT_PROFIT_MARGIN} margin | force-exit last {FORCE_EXIT_SECONDS_LEFT}s")
            log(f"Min move to trust direction: {REACT_MIN_DELTA_PCT}%")
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
            host=CLOB_API,
            key=os.environ["POLY_PRIVATE_KEY"],
            chain_id=137,
            signature_type=signature_type,
            funder=os.environ["POLY_PROXY_WALLET"],
        )
        self.client.set_api_creds(self.client.create_or_derive_api_key())
        self.client.update_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=signature_type,
        ))

    # ── BUY PHASE ────────────────────────────────────────────────────────────

    def _attempt_buy(self, market: dict, window_open_time: float, ceiling: float = None, timeout: float = None) -> dict:
        """
        Attempt to buy DOWN at up to ceiling within BUY_TIMEOUT_SEC
        of window open. Returns a dict describing what happened, with real
        elapsed time in milliseconds from window_open_time.
        """
        ceiling = BUY_CEILING_PRICE if ceiling is None else ceiling
        timeout = BUY_TIMEOUT_SEC if timeout is None else timeout
        crypto = market["crypto"]
        token  = market["target_token"]

        if self.dry_run:
            # Poll the REAL live order book repeatedly for up to BUY_TIMEOUT_SEC,
            # looking for a real ask at or below our ceiling. No order is placed.
            deadline = window_open_time + timeout
            while now_unix() < deadline:
                book = get_order_book(token)
                price, size = best_ask(book)
                elapsed_ms = (now_unix() - window_open_time) * 1000
                if price is not None and price <= ceiling:
                    MIN_SHARES = 1  # NOT independently confirmed by Polymarket docs — this is a minimal safety floor only, not a verified exchange rule. The old "5" had no documented source and should not have been asserted with confidence. Test a real small live order to find the true threshold, if one exists.
                    shares = max(MIN_SHARES, round(self.amount / price))
                    actual_cost = round(shares * price, 2)
                    if actual_cost > self.amount * 1.5:
                        log(f"[DRY] Would need ~${actual_cost:.2f} to meet {MIN_SHARES}-share minimum at this price "
                            f"— above ${self.amount:.2f} stake, would skip in live mode", crypto)
                        return {"result": "skipped_min_size", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}
                    log(f"[DRY] BUY would fill: ask ${price:.3f} (size {size}) at {elapsed_ms:.0f}ms", crypto)
                    return {"result": "bought", "price": price, "shares": shares, "elapsed_ms": elapsed_ms}
                time.sleep(POLL_INTERVAL_FAST)
            elapsed_ms = (now_unix() - window_open_time) * 1000
            log(f"[DRY] BUY missed: no ask <= ${ceiling} within {timeout}s "
                f"(waited {elapsed_ms:.0f}ms)", crypto)
            return {"result": "missed", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}

        # LIVE: place a real resting limit buy at the ceiling price, poll for
        # a fill, cancel if it doesn't fill within the timeout.
        from py_clob_client_v2 import OrderArgsV2, Side, OrderType
        # Whole-share sizing guarantees price × size lands on a clean 2-decimal
        # dollar amount (price already has ≤2dp; integer × 2dp never adds more
        # decimal places) — fixes the same maker-amount precision bug found
        # and fixed on the delta bot. This also enforces the exchange's real
        # 5-share minimum order size, which your stake must clear.
        MIN_SHARES = 1  # NOT independently confirmed by Polymarket docs — this is a minimal safety floor only, not a verified exchange rule. The old "5" had no documented source and should not have been asserted with confidence. Test a real small live order to find the true threshold, if one exists.
        size = max(MIN_SHARES, round(self.amount / ceiling))
        actual_cost = round(size * ceiling, 2)
        if actual_cost > self.amount * 1.5:
            log(f"⚠️ To meet the {MIN_SHARES}-share minimum, this order needs ~${actual_cost:.2f}, "
                f"well above your ${self.amount:.2f} stake — skipping this window.", crypto)
            return {"result": "skipped_min_size", "price": None, "shares": 0, "elapsed_ms": 0}
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=ceiling, size=size, side=Side.BUY),
                order_type=OrderType.GTC,
            )
        except Exception as e:
            log(f"❌ BUY order failed to submit: {e}", crypto)
            return {"result": "error", "price": None, "shares": 0, "elapsed_ms": 0}

        order_id = resp.get("orderID", "")
        status   = str(resp.get("status", "")).lower()
        if status == "matched":
            elapsed_ms = (now_unix() - window_open_time) * 1000
            # Parse the ACTUAL fill price/size rather than assuming the ceiling
            # was paid — Polymarket's own matching rules give price improvement
            # to the taker (e.g. a resting ask at 0.50 fills you at 0.50, not
            # your 0.52 ceiling). Confirmed reliable for LIMIT orders on the
            # delta bot's earliest live trade; unlike MARKET orders, this
            # response type has not been observed returning $0 here.
            try:
                fill_cost  = round(float(resp["makingAmount"]) / 1_000_000, 2)
                fill_size  = round(float(resp["takingAmount"]) / 1_000_000, 4)
                fill_price = round(fill_cost / fill_size, 4) if fill_size else ceiling
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                log(f"⚠️ Could not parse actual fill price ({e}), falling back to ceiling — "
                    f"this UNDERSTATES profit if price improvement occurred. Raw resp: {resp}", crypto)
                fill_price, fill_size = ceiling, size
            log(f"✅ BUY matched immediately at ${fill_price} (ceiling was ${ceiling}), "
                f"order {order_id[:16]}... ({elapsed_ms:.0f}ms)", crypto)
            return {"result": "bought", "price": fill_price, "shares": fill_size, "elapsed_ms": elapsed_ms}

        # Still resting — poll until timeout, then cancel if unfilled.
        deadline = window_open_time + timeout
        while now_unix() < deadline:
            time.sleep(0.25)
            try:
                detail = self.client.get_order(order_id)
            except Exception:
                detail = None
            if detail is None:
                # NOTE: get_order() has been observed returning None once an
                # order is no longer in the open-orders index — which appears
                # to happen once an order is fully matched. This is an
                # inference, not confirmed by Polymarket docs for this case.
                #
                # KNOWN GAP: unlike the immediate-match case above, there is no
                # response object here to parse an actual fill price from — the
                # original `resp` was captured while the order was still
                # "live" (resting), before it matched, so its amounts reflect
                # the requested order, not the eventual fill. This falls back
                # to the ceiling price, which may UNDERSTATE profit if the
                # order filled at a better price sometime during the rest
                # period. Not yet fixed — would need a get_trades() lookup.
                elapsed_ms = (now_unix() - window_open_time) * 1000
                log(f"✅ BUY appears filled (order no longer open), order {order_id[:16]}... ({elapsed_ms:.0f}ms) "
                    f"— price assumed at ceiling \\${ceiling}, may understate actual profit", crypto)
                return {"result": "bought", "price": ceiling, "shares": size, "elapsed_ms": elapsed_ms}

        # Timed out — cancel whatever is left resting.
        from py_clob_client_v2 import OrderPayload
        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
        except Exception as e:
            log(f"⚠️ Cancel request failed ({e}) — order may still be resting, check manually.", crypto)

        elapsed_ms = (now_unix() - window_open_time) * 1000
        log(f"❌ BUY timed out after {timeout}s, cancelled ({elapsed_ms:.0f}ms)", crypto)
        return {"result": "missed", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}

    # ── SELL PHASE ───────────────────────────────────────────────────────────

    def _watch_for_sell(self, market: dict, buy_info: dict, window_open_time: float, sell_floor: float = None) -> dict:
        """
        After a successful buy, watch for the price to reach the sell
        floor/target. Tracks every distinct opportunity, attempts to sell on
        each one until successful, and force-exits in the closing seconds if
        no opportunity worked out.

        sell_floor defaults to the module-level SELL_FLOOR_PRICE (the
        "predict" variant's absolute target). The "react" variant passes a
        RELATIVE floor computed from its own actual buy price instead, since
        that variant doesn't buy at a known ~50c price.
        """
        sell_floor = SELL_FLOOR_PRICE if sell_floor is None else sell_floor
        crypto      = market["crypto"]
        token       = market["target_token"]
        close_ts    = market["close_ts"]
        buy_price   = buy_info["price"]
        shares      = buy_info["shares"]
        opportunities = 0
        in_opportunity_zone = False  # tracks edge-triggering so we log each distinct touch, not every poll

        while True:
            seconds_left = close_ts - now_unix()
            if seconds_left <= FORCE_EXIT_SECONDS_LEFT:
                break

            book = get_order_book(token)
            price, size = best_bid(book)

            if price is not None and price >= sell_floor:
                if not in_opportunity_zone:
                    opportunities += 1
                    in_opportunity_zone = True
                    log(f"Opportunity #{opportunities}: bid ${price:.3f} (size {size}) — attempting sell", crypto)

                sell_result = self._attempt_sell(token, shares, price, crypto, sell_floor)
                if sell_result["result"] == "sold":
                    pnl = round((sell_result["price"] - buy_price) * shares, 4)
                    return {**sell_result, "opportunities": opportunities, "pnl_usd": pnl,
                            "notes": f"sold on opportunity #{opportunities}"}
                # sell attempt failed (e.g. size vanished before we could act) — keep watching
            else:
                in_opportunity_zone = False

            time.sleep(POLL_INTERVAL_SLOW)

        # Reached the force-exit window still holding — exit at whatever's available.
        log(f"⏰ Force-exit window reached ({FORCE_EXIT_SECONDS_LEFT}s left), still holding — exiting at best price", crypto)
        exit_result = self._force_exit(token, shares, crypto)
        pnl = round((exit_result["price"] - buy_price) * shares, 4) if exit_result["price"] is not None else -round(buy_price * shares, 4)
        return {**exit_result, "opportunities": opportunities, "pnl_usd": pnl, "notes": "force-exit, no opportunity filled"}

    def _attempt_sell(self, token: str, shares: float, observed_price: float, crypto: str, sell_floor: float = None) -> dict:
        sell_floor = SELL_FLOOR_PRICE if sell_floor is None else sell_floor
        if self.dry_run:
            # Real depth check: would our share count actually have filled at
            # this observed bid, or was there insufficient size resting there?
            book = get_order_book(token)
            price, size = best_bid(book)
            if price is not None and price >= sell_floor and size >= shares:
                log(f"[DRY] SELL would fill: bid ${price:.3f} (sufficient depth for {shares} shares)", crypto)
                return {"result": "sold", "price": price}
            log(f"[DRY] SELL opportunity did not have enough depth ({size} < {shares} needed) — missed", crypto)
            return {"result": "missed", "price": None}

        from py_clob_client_v2 import OrderArgsV2, Side, OrderType
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=sell_floor, size=shares, side=Side.SELL),
                order_type=OrderType.FAK,
            )
        except Exception as e:
            log(f"⚠️ SELL order failed: {e}", crypto)
            return {"result": "missed", "price": None}

        status = str(resp.get("status", "")).lower()
        if status == "matched":
            log(f"✅ SELL matched at order {resp.get('orderID','')[:16]}...", crypto)
            return {"result": "sold", "price": observed_price}
        return {"result": "missed", "price": None}

    def _force_exit(self, token: str, shares: float, crypto: str) -> dict:
        """Exit at any available price — no floor, no ceiling. This is a
        deliberate loss-minimization exit, not a profit attempt."""
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

    # ── WINDOW HANDLER ───────────────────────────────────────────────────────

    def _handle_window(self, slug_prefix: str, start_ts: int):
        crypto = MARKETS[slug_prefix]

        # Busy-wait right up to the window boundary so the buy attempt fires
        # as close to window open as this VPS/network can achieve. Being
        # honest about a real limit: competing for the very first liquidity
        # in the opening 1-2 seconds also depends on network latency to
        # Polymarket's servers and how fast market makers seed that liquidity
        # — this loop cannot guarantee winning that race, only minimize our
        # own added delay.
        while now_unix() < start_ts - 1:
            time.sleep(0.2)
        while now_unix() < start_ts:
            time.sleep(0.005)

        window_open_time = now_unix()

        # The market listing itself may lag slightly behind the actual time
        # boundary — retry briefly rather than assume it's instantly available.
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

        # Decide which side to buy based on the ending window's own delta
        # magnitude + momentum agreement, instead of always buying Down or
        # falling back to Down when unsure. This bot is meant to take FEWER,
        # more selective trades — if the signal isn't confident, we skip
        # the window entirely rather than force a trade. This is a genuinely
        # new, unvalidated hypothesis for THIS use case — it aims to reduce
        # how OFTEN you're on the wrong side, but does NOT change what
        # happens when the signal is wrong: a loss here is still close to
        # full stake, same as before.
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
            "timestamp":   ts_str(),
            "bot_name":    self.bot_name,
            "mode":        self.mode_str,
            "crypto":      crypto,
            "target_side": target_side,
            "signal_delta_pct": signal["delta_pct"],
            "signal_reason": signal["reason"],
            "slug":        market["slug"],
            "buy_result":  buy_info["result"],
            "buy_price":   buy_info["price"],
            "buy_shares":  buy_info["shares"],
            "buy_elapsed_ms": round(buy_info["elapsed_ms"], 1),
        }

        if buy_info["result"] != "bought":
            row.update({"sell_result": "n/a", "sell_price": "", "pnl_usd": 0, "notes": "no buy fill", "num_opportunities": 0})
            self._record(row)
            return

        sell_info = self._watch_for_sell(market, buy_info, window_open_time)
        row.update({
            "sell_result":       sell_info["result"],
            "sell_price":        sell_info["price"],
            "num_opportunities": sell_info["opportunities"],
            "pnl_usd":           sell_info["pnl_usd"],
            "notes":             sell_info["notes"],
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

    # ── "REACT" VARIANT WINDOW HANDLER ──────────────────────────────────────

    def _handle_window_react(self, slug_prefix: str, start_ts: int):
        """
        Alternative to _handle_window: waits REACT_WAIT_SEC into the new
        window, decides direction from the window's OWN real price action
        (not the prior window), buys at whatever price results, then targets
        a fixed profit MARGIN above that actual entry price rather than a
        fixed absolute floor/target. This is a genuinely different, testable
        hypothesis — run alongside --variant predict to compare real results.
        """
        crypto = MARKETS[slug_prefix]
        symbol = SYMBOLS.get(crypto)

        while now_unix() < start_ts - 1:
            time.sleep(0.2)
        while now_unix() < start_ts:
            time.sleep(0.005)

        window_open_time = now_unix()
        price_at_open = get_binance_price(symbol) if symbol else None

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

        # Wait and observe this window's own direction before deciding.
        while now_unix() < window_open_time + REACT_WAIT_SEC:
            time.sleep(0.1)

        direction, observed_pct = get_react_direction(crypto, price_at_open)
        if direction is None:
            log(f"React: no clear/strong-enough direction after observation window (moved {observed_pct:.4f}%, "
                f"need {REACT_MIN_DELTA_PCT}%) — skipping", crypto)
            return

        log(f"React: observed {direction} ({observed_pct:.4f}%) after {REACT_WAIT_SEC}s — buying {direction}", crypto)
        market["target_side"]  = direction
        market["target_token"] = market["down_token"] if direction == "Down" else market["up_token"]

        buy_start = now_unix()
        buy_info = self._attempt_buy(market, buy_start, ceiling=REACT_BUY_CEILING, timeout=REACT_BUY_TIMEOUT_SEC)

        row = {
            "timestamp": ts_str(), "bot_name": self.bot_name, "mode": self.mode_str,
            "crypto": crypto, "target_side": direction, "signal_delta_pct": observed_pct,
            "signal_reason": f"react: observed {direction} ({observed_pct:.4f}%) after {REACT_WAIT_SEC}s",
            "slug": market["slug"], "buy_result": buy_info["result"],
            "buy_price": buy_info["price"], "buy_shares": buy_info["shares"],
            "buy_elapsed_ms": round(buy_info["elapsed_ms"], 1),
        }

        if buy_info["result"] != "bought":
            row.update({"sell_result": "n/a", "sell_price": "", "pnl_usd": 0, "notes": "no buy fill", "num_opportunities": 0})
            self._record(row)
            return

        # Relative sell floor: actual entry price + target margin, not a fixed absolute price.
        relative_floor = round(buy_info["price"] + REACT_PROFIT_MARGIN, 4)
        sell_info = self._watch_for_sell(market, buy_info, window_open_time, sell_floor=relative_floor)
        row.update({
            "sell_result": sell_info["result"], "sell_price": sell_info["price"],
            "num_opportunities": sell_info["opportunities"], "pnl_usd": sell_info["pnl_usd"],
            "notes": sell_info["notes"] + f" (relative floor was ${relative_floor})",
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
            # Sleep briefly past window close before recalculating the next boundary
            time.sleep(2)

    # ── RUN / SUMMARY ────────────────────────────────────────────────────────

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
        bought  = [t for t in trades if t["buy_result"] == "bought"]
        sold_60 = [t for t in bought if t["sell_result"] == "sold" and float(t["sell_price"] or 0) >= SELL_TARGET_PRICE]
        sold_58 = [t for t in bought if t["sell_result"] == "sold" and float(t["sell_price"] or 0) < SELL_TARGET_PRICE]
        forced  = [t for t in bought if t["sell_result"] in ("exited", "no_bids", "unmatched")]
        total_pnl = sum(float(t["pnl_usd"] or 0) for t in trades)

        log(f"  Buy fills: {len(bought)}/{len(trades)}")
        log(f"  Sold at/above target (~60c): {len(sold_60)}")
        log(f"  Sold at floor (58-60c range): {len(sold_58)}")
        log(f"  Force-exited (never reached floor): {len(forced)}")
        log(f"  Total PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}")
        log("-" * 70)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Down-Spread Scalper Bot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Observe real order book data, place no real orders")
    mode.add_argument("--live", action="store_true", help="Place real orders with real funds")
    parser.add_argument("--amount", type=float, default=2.0, help="USDC stake per trade (default: $2)")
    parser.add_argument("--variant", choices=["predict", "react"], default="predict",
                         help="predict: decide direction from the prior window, buy instantly at open. "
                              "react: wait a few seconds into the new window, decide from its own price action.")
    args = parser.parse_args()

    bot = SpreadBot(dry_run=args.dry_run, amount=args.amount, variant=args.variant)
    bot.run()
