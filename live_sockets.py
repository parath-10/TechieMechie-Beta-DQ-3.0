"""NEXUS real-time layer: WebSockets instead of every browser polling the server.

How it works
- ONE background worker per server copy brings in new (simulated) orders every WS_TICK_SECONDS and pushes
  them straight to every open dashboard through a WebSocket. Ten open tabs cost the same database work as one.
- The worker does nothing while nobody is connected, so a forgotten tab on Railway no longer hammers
  the server or the database (and the browser closes its socket after the tab has been hidden for a minute).
- Each message is small: just the new orders. The page updates stock, counters and the order table at once.

Environment variables (all optional)
  WS_TICK_SECONDS   how often new orders are brought in while someone is watching (default 5)
  WS_MAX_CLIENTS    most open dashboards per server copy (default 200)
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import WebSocket
from starlette.concurrency import run_in_threadpool

import ads

log = logging.getLogger("nexus.realtime")

TICK = max(1.0, float(os.getenv("WS_TICK_SECONDS", "5") or 5))
MAX_CLIENTS = int(os.getenv("WS_MAX_CLIENTS", "200") or 200)
HEARTBEAT = 20.0  # seconds; keeps Railway's proxy from closing an idle socket
SEND_TIMEOUT = 5.0  # a slow or dead browser must never block the others
ERROR_WAIT = 10.0  # pause after a failed sync (for example a database hiccup)


class Hub:
    """The set of open dashboards, and a way to send one message to all of them."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> bool:
        if len(self.clients) >= MAX_CLIENTS:
            await ws.close(code=1013)  # "try again later"
            return False
        await ws.accept()
        self.clients.add(ws)
        return True

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self.clients:
            return
        text = json.dumps(message, default=str)

        async def _send(ws: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(ws.send_text(text), SEND_TIMEOUT)
                return None
            except Exception:
                return ws  # gone or too slow: drop it

        for dead in await asyncio.gather(*(_send(w) for w in list(self.clients))):
            if dead is not None:
                self.clients.discard(dead)


hub = Hub()


def order_event(result: dict[str, Any]) -> dict[str, Any] | None:
    """The WebSocket message for one sync result, or None when no order came in."""
    new_orders = result.get("new_orders") or []
    if not new_orders:
        return None
    return {
        "type": "orders",
        "new_orders": new_orders,
        "stock": result.get("stock", []),
        "orders": result.get("orders", len(new_orders)),
        "units": result.get("units", 0),
        "views": result.get("views", 0),
        "reviews": result.get("reviews", 0),
        "synced_at": result.get("synced_at", ""),
        "sent_at": time.time(),
    }


async def publish(result: dict[str, Any]) -> None:
    """Send a finished sync to every open dashboard (used by the worker and by the manual Sync button)."""
    event = order_event(result)
    if event:
        await hub.broadcast(event)


async def worker() -> None:
    """Runs for the life of the server. Sleeps while nobody is watching."""
    last_beat = time.monotonic()
    while True:
        try:
            if not hub.clients:
                await asyncio.sleep(1.0)
                continue
            result = await run_in_threadpool(ads.sync_orders)
            await publish(result)
            if time.monotonic() - last_beat >= HEARTBEAT:
                await hub.broadcast({"type": "hb", "t": time.time()})
                last_beat = time.monotonic()
            await asyncio.sleep(TICK)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # database hiccup etc.: log it, wait, carry on
            log.warning("Real-time sync failed: %s", exc)
            await asyncio.sleep(ERROR_WAIT)
