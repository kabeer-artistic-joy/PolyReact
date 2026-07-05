#!/usr/bin/env python3
"""
Polymarket Up-Only Scalper Bot
====================================
Buys UP unconditionally at the instant every 5-minute window opens — no
delta, no momentum, no magnitude check, no observation period. Just:

    Buy UP as cheaply as possible (target 50c, ceiling 52c) within 2
    seconds of window open. Sell as soon as the price reaches 55c. If that
    never happens, exit for whatever is available in the closing seconds
    rather than hold to resolution.

This is a SPREAD/VOLATILITY play, not a directional one — it does not care
which side (Up or Down) ultimately wins. It only cares whether the UP price
touches 55c at some point during the window before you're forced to exit.

IMPORTANT — read before running live:
  A loss in this bot is NOT bounded the way a directionally-confirmed entry
  would be. If UP never reaches 55c during the window, this bot force-exits
  in the final FORCE_EXIT_SECONDS at whatever price is available — which
  could be far below your buy price. A single bad window can cost close to
  your full stake. Run --dry-run for a meaningful sample before --live.

Modes:
  --dry-run   No real orders. Polls the REAL, LIVE order book throughout each
              window and computes what WOULD have happened using real market
              depth data — not simulated or assumed prices.
  --live      Places real limit orders per the exact logic below, using your
              Polymarket deposit wallet.

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

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

MARKETS = {
    "btc-updown-5m": "BTC",
    "eth-updown-5m": "ETH",
}

BUY_TARGET_PRICE   = 0.50   # what we hope to pay
BUY_CEILING_PRICE  = 0.54   # max we're willing to pay — this is the actual limit price used
BUY_TIMEOUT_SEC    = 3.0    # cancel the buy attempt if unfilled after this long

PROFIT_MARGIN      = 0.05   # sell as soon as (current bid) >= (actual buy price) + this margin.
                              # Relative to YOUR entry, not a fixed absolute price — if you buy at
                              # 0.47, the trigger is 0.52; if you buy at 0.52, the trigger is 0.57.
                              # Same profit either way, since it's measured from where you actually got in.

FORCE_EXIT_SECONDS_LEFT = 60  # in the final N seconds of the window, exit at any price if still holding

POLL_INTERVAL_FAST = 0.05   # tight poll interval used right at window open (seconds)
POLL_INTERVAL_SLOW = 1.0    # normal poll interval while watching for a sell opportunity

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
    """Find the Up/Down market for a window starting at start_ts."""
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
    """Raw public order book fetch — no auth required."""
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


def next_window_start(now: float) -> int:
    """Returns the unix timestamp of the next 5-minute boundary."""
    return int((now // 300) + 1) * 300


# ─── PERSISTENT CSV LOG ──────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "bot_name", "mode", "crypto", "slug",
    "buy_result", "buy_price", "buy_shares", "buy_elapsed_ms",
    "num_opportunities", "sell_result", "sell_price",
    "pnl_usd", "notes",
]

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
    def __init__(self, dry_run: bool, amount: float):
        self.dry_run  = dry_run
        self.amount   = amount
        self.bot_name = os.getenv("BOT_NAME", "spread_bot")
        self.mode_str = "dry_run" if dry_run else "live"
        self.stop_event = threading.Event()
        self.trades = []
        self.trades_lock = threading.Lock()
        self.logger = TradeLogger(self.bot_name)

        self.client = None
        if not dry_run:
            self._init_client()

        log("=" * 70)
        log(f"Up-Only Scalper | {self.mode_str.upper()} | ${amount:.2f}/trade | bot_name={self.bot_name}")
        log(f"Buy: target ${BUY_TARGET_PRICE} ceiling ${BUY_CEILING_PRICE} timeout {BUY_TIMEOUT_SEC}s")
        log(f"Sell trigger: entry price + ${PROFIT_MARGIN} margin | force-exit last {FORCE_EXIT_SECONDS_LEFT}s")
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

    def _attempt_buy(self, market: dict, window_open_time: float) -> dict:
        """Attempt to buy UP at up to BUY_CEILING_PRICE within BUY_TIMEOUT_SEC of window open."""
        crypto = market["crypto"]
        token  = market["up_token"]

        if self.dry_run:
            deadline = window_open_time + BUY_TIMEOUT_SEC
            last_seen_price = None
            while now_unix() < deadline:
                book = get_order_book(token)
                price, size = best_ask(book)
                if price is not None:
                    last_seen_price = price
                elapsed_ms = (now_unix() - window_open_time) * 1000
                if price is not None and price <= BUY_CEILING_PRICE:
                    MIN_SHARES = 1
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
            price_info = f"last real ask seen was ${last_seen_price:.3f}" if last_seen_price is not None else "no asks seen at all"
            log(f"[DRY] BUY missed: no ask <= ${BUY_CEILING_PRICE} within {BUY_TIMEOUT_SEC}s ({price_info}) "
                f"(waited {elapsed_ms:.0f}ms)", crypto)
            return {"result": "missed", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}

        # LIVE
        from py_clob_client_v2 import OrderArgsV2, Side, OrderType
        MIN_SHARES = 1  # NOT independently confirmed by Polymarket docs — a minimal safety floor only.
        size = max(MIN_SHARES, round(self.amount / BUY_CEILING_PRICE))
        actual_cost = round(size * BUY_CEILING_PRICE, 2)
        if actual_cost > self.amount * 1.5:
            log(f"⚠️ To meet the {MIN_SHARES}-share minimum, this order needs ~${actual_cost:.2f}, "
                f"well above your ${self.amount:.2f} stake — skipping this window.", crypto)
            return {"result": "skipped_min_size", "price": None, "shares": 0, "elapsed_ms": 0}
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=BUY_CEILING_PRICE, size=size, side=Side.BUY),
                order_type=OrderType.GTC,
            )
        except Exception as e:
            log(f"❌ BUY order failed to submit: {e}", crypto)
            return {"result": "error", "price": None, "shares": 0, "elapsed_ms": 0}

        order_id = resp.get("orderID", "")
        status   = str(resp.get("status", "")).lower()
        if status == "matched":
            elapsed_ms = (now_unix() - window_open_time) * 1000
            try:
                fill_cost  = round(float(resp["makingAmount"]) / 1_000_000, 2)
                fill_size  = round(float(resp["takingAmount"]) / 1_000_000, 4)
                fill_price = round(fill_cost / fill_size, 4) if fill_size else BUY_CEILING_PRICE
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                log(f"⚠️ Could not parse actual fill price ({e}), falling back to ceiling — "
                    f"this UNDERSTATES profit if price improvement occurred. Raw resp: {resp}", crypto)
                fill_price, fill_size = BUY_CEILING_PRICE, size
            log(f"✅ BUY matched immediately at ${fill_price} (ceiling was ${BUY_CEILING_PRICE}), "
                f"order {order_id[:16]}... ({elapsed_ms:.0f}ms)", crypto)
            return {"result": "bought", "price": fill_price, "shares": fill_size, "elapsed_ms": elapsed_ms}

        deadline = window_open_time + BUY_TIMEOUT_SEC
        while now_unix() < deadline:
            time.sleep(0.25)
            try:
                detail = self.client.get_order(order_id)
            except Exception:
                detail = None
            if detail is None:
                elapsed_ms = (now_unix() - window_open_time) * 1000
                log(f"✅ BUY appears filled (order no longer open), order {order_id[:16]}... ({elapsed_ms:.0f}ms) "
                    f"— price assumed at ceiling ${BUY_CEILING_PRICE}, may understate actual profit", crypto)
                return {"result": "bought", "price": BUY_CEILING_PRICE, "shares": size, "elapsed_ms": elapsed_ms}

        from py_clob_client_v2 import OrderPayload
        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
        except Exception as e:
            log(f"⚠️ Cancel request failed ({e}) — order may still be resting, check manually.", crypto)

        elapsed_ms = (now_unix() - window_open_time) * 1000
        log(f"❌ BUY timed out after {BUY_TIMEOUT_SEC}s, cancelled ({elapsed_ms:.0f}ms)", crypto)
        return {"result": "missed", "price": None, "shares": 0, "elapsed_ms": elapsed_ms}

    # ── SELL PHASE ───────────────────────────────────────────────────────────

    def _watch_for_sell(self, market: dict, buy_info: dict, window_open_time: float) -> dict:
        """After a successful buy, watch for the UP price to reach buy_price + PROFIT_MARGIN."""
        crypto      = market["crypto"]
        token       = market["up_token"]
        close_ts    = market["close_ts"]
        buy_price   = buy_info["price"]
        shares      = buy_info["shares"]
        sell_trigger = round(buy_price + PROFIT_MARGIN, 4)  # relative to THIS trade's actual entry, not a fixed price
        log(f"Sell trigger for this trade: ${sell_trigger} (bought ${buy_price} + ${PROFIT_MARGIN} margin)", crypto)
        opportunities = 0
        in_opportunity_zone = False

        while True:
            seconds_left = close_ts - now_unix()
            if seconds_left <= FORCE_EXIT_SECONDS_LEFT:
                break

            book = get_order_book(token)
            price, size = best_bid(book)

            if price is not None and price >= sell_trigger:
                if not in_opportunity_zone:
                    opportunities += 1
                    in_opportunity_zone = True
                    log(f"Opportunity #{opportunities}: bid ${price:.3f} (size {size}) — attempting sell", crypto)

                sell_result = self._attempt_sell(token, shares, price, crypto, sell_trigger)
                if sell_result["result"] == "sold":
                    pnl = round((sell_result["price"] - buy_price) * shares, 4)
                    return {**sell_result, "opportunities": opportunities, "pnl_usd": pnl,
                            "notes": f"sold on opportunity #{opportunities}"}
            else:
                in_opportunity_zone = False

            time.sleep(POLL_INTERVAL_SLOW)

        log(f"⏰ Force-exit window reached ({FORCE_EXIT_SECONDS_LEFT}s left), still holding — exiting at best price", crypto)
        exit_result = self._force_exit(token, shares, crypto)
        pnl = round((exit_result["price"] - buy_price) * shares, 4) if exit_result["price"] is not None else -round(buy_price * shares, 4)
        return {**exit_result, "opportunities": opportunities, "pnl_usd": pnl, "notes": "force-exit, no opportunity filled"}

    def _attempt_sell(self, token: str, shares: float, observed_price: float, crypto: str, sell_trigger: float) -> dict:
        if self.dry_run:
            book = get_order_book(token)
            price, size = best_bid(book)
            if price is not None and price >= sell_trigger and size >= shares:
                log(f"[DRY] SELL would fill: bid ${price:.3f} (sufficient depth for {shares} shares)", crypto)
                return {"result": "sold", "price": price}
            log(f"[DRY] SELL opportunity did not have enough depth ({size} < {shares} needed) — missed", crypto)
            return {"result": "missed", "price": None}

        from py_clob_client_v2 import OrderArgsV2, Side, OrderType
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=sell_trigger, size=shares, side=Side.SELL),
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
        """Exit at any available price — no floor, no ceiling."""
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

        buy_info = self._attempt_buy(market, window_open_time)

        row = {
            "timestamp":   ts_str(),
            "bot_name":    self.bot_name,
            "mode":        self.mode_str,
            "crypto":      crypto,
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
                self._handle_window(slug_prefix, start_ts)
            except Exception as e:
                log(f"⚠️ Unhandled error this window: {e}", crypto)
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
        sold    = [t for t in bought if t["sell_result"] == "sold"]
        forced  = [t for t in bought if t["sell_result"] in ("exited", "no_bids", "unmatched")]
        total_pnl = sum(float(t["pnl_usd"] or 0) for t in trades)

        log(f"  Buy fills: {len(bought)}/{len(trades)}")
        log(f"  Sold at/above entry+${PROFIT_MARGIN} margin: {len(sold)}")
        log(f"  Force-exited (never reached trigger): {len(forced)}")
        log(f"  Total PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}")
        log("-" * 70)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Up-Only Scalper Bot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Observe real order book data, place no real orders")
    mode.add_argument("--live", action="store_true", help="Place real orders with real funds")
    parser.add_argument("--amount", type=float, default=2.0, help="USDC stake per trade (default: $2)")
    args = parser.parse_args()

    bot = SpreadBot(dry_run=args.dry_run, amount=args.amount)
    bot.run()
