"""NEXUS database helpers: one shared Supabase client for the whole backend.

Environment variables (set them in Railway, or in .env for local runs):
  SUPABASE_URL               https://<project-ref>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY  the service_role key (Project Settings -> API). Backend only, never in the browser.
  APP_TIMEZONE               optional, default Asia/Kolkata (Railway servers run on UTC)
"""

import os
from datetime import date, datetime
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

PAGE_SIZE = 1000  # Supabase returns at most 1000 rows per request, so bigger reads are paged


class DatabaseNotConfigured(RuntimeError):
    pass


@lru_cache(maxsize=1)
def client() -> Client:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise DatabaseNotConfigured(
            "The database is not set up: add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to the environment.")
    return create_client(url, key)


def select_all(table: str, columns: str = "*", filters: list[tuple[str, str, Any]] | None = None,
               order: str | list[str] | None = None, desc: bool = False, limit: int | None = None) -> list[dict]:
    """Read rows, following pages so large tables are read completely.
    filters: [("eq", "status", "active"), ("in_", "sku", ["A", "B"]), ("lte", "ends_sim", 3.2)]"""
    orders = [order] if isinstance(order, str) else (order or [])
    out: list[dict] = []
    start = 0
    while True:
        q = client().table(table).select(columns)
        for op, col, val in filters or []:
            if op == "in_" and not val:
                return []  # "in ()" matches nothing
            q = getattr(q, op)(col, val)
        for col in orders:
            q = q.order(col, desc=desc)
        end = start + PAGE_SIZE - 1
        if limit is not None:
            end = min(end, limit - 1)
        rows = q.range(start, end).execute().data or []
        out.extend(rows)
        if len(rows) < end - start + 1 or (limit is not None and len(out) >= limit):
            return out
        start += PAGE_SIZE


def one(table: str, **eq: Any) -> dict | None:
    q = client().table(table).select("*")
    for col, val in eq.items():
        q = q.eq(col, val)
    rows = q.limit(1).execute().data or []
    return rows[0] if rows else None


def rpc(fn: str, params: dict | None = None) -> Any:
    """Call one of the nexus_* SQL functions from schema.sql."""
    return client().rpc(fn, params or {}).execute().data


# ---------- Local time (Railway runs on UTC) ----------
_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))


def now_local() -> datetime:
    return datetime.now(_TZ)


def today_local() -> date:
    return now_local().date()
