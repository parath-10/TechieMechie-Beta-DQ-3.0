"""NEXUS data layer: products, advertising, posts, orders and month-wise history, stored in Supabase.

Nothing is kept in Python memory between requests, so any number of server copies can run.
Order sync is written to the database in ONE transaction (SQL function nexus_apply_sync).

The numbers for the sample products are SIMULATED so the dashboard has something to show.
Once real platform APIs are connected, replace `generate_orders()` and `seed()` with real data.
"""

import hashlib
import math
import os
import random
import time
import uuid
from datetime import datetime
from typing import Any

import connectors
from db import client, now_local, one, rpc, select_all, today_local

AD_PLATFORMS = ["Instagram", "Facebook", "YouTube", "Google", "Amazon", "Flipkart", "TikTok"]
SALES_CHANNELS = AD_PLATFORMS + ["Website"]  # Website = direct sales, no ads

# Which ad formats each platform accepts
AD_FORMATS = {
    "image": {"label": "Image post", "hint": "Square or portrait picture"},
    "video": {"label": "Video", "hint": "Any shape, up to 10 minutes"},
    "reel": {"label": "Reel / Short", "hint": "Vertical 9:16 video, up to 90 seconds"},
}
FORMAT_SUPPORT = {
    "Instagram": ["image", "video", "reel"], "Facebook": ["image", "video", "reel"],
    "YouTube": ["video", "reel"], "TikTok": ["video", "reel"],
    "Google": ["image", "video"], "Amazon": ["image", "video"], "Flipkart": ["image"],
}
# Simulation: how each format changes views, click rate and orders compared to an image
FORMAT_EFFECT = {"image": (1.0, 1.0, 1.0), "video": (1.3, 0.9, 1.1), "reel": (1.8, 0.75, 1.2)}

# ---------- Order sync clock ----------
# Until real accounts are connected, orders are simulated. To make a demo watchable, time runs faster:
# ORDER_SIM_SPEED=60 means 1 real minute = 1 hour of orders. Set it to 1 for real-time speed.
SIM_SPEED = float(os.getenv("ORDER_SIM_SPEED", "60"))
_sim_start: float | None = None  # read once from the database; it never changes, so caching it is safe


def _start() -> float:
    global _sim_start
    if _sim_start is None:
        _sim_start = float(rpc("nexus_clock")["start"])
    return _sim_start


def sim_days() -> float:
    """Simulated days passed since the shared clock started (same on every server copy)."""
    return (time.time() - _start()) * SIM_SPEED / 86400


REVIEW_POOL = {
    5: ["Excellent quality, worth every rupee.", "Loved it, ordering another one.", "Perfect fit and fast delivery.",
        "Better than I expected from the ad.", "Fabric feels premium. Highly recommend."],
    4: ["Good product, slightly late delivery.", "Nice quality, colour a little different from the ad.",
        "Comfortable, good value for money.", "Happy with it, packaging could be better."],
    3: ["Okay for the price.", "Average quality, fits a bit small.", "Decent, but the photos looked better."],
    2: ["Colour faded after one wash.", "Took too long to arrive.", "Size chart was not accurate."],
    1: ["Received a damaged item.", "Not like the ad at all, returning it."],
}

# How many views ₹1 of daily budget buys, roughly, and typical click and buy rates (simulation only)
PLATFORM_RATES = {
    "Instagram": (38, 0.021, 0.028), "Facebook": (32, 0.017, 0.026), "YouTube": (55, 0.009, 0.020),
    "Google": (18, 0.042, 0.035), "Amazon": (14, 0.060, 0.090), "Flipkart": (15, 0.055, 0.080),
    "TikTok": (60, 0.012, 0.018),
}

METRICS = ("units", "revenue", "views", "clicks", "spend", "reviews", "rating_sum")
_ZERO = dict.fromkeys(METRICS, 0)


def _rng(*parts: Any) -> random.Random:
    seed_ = int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)
    return random.Random(seed_)


def month_keys(n: int = 6) -> list[str]:
    """Last n months as 'YYYY-MM', oldest first, ending with the current month."""
    today = today_local()
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out[::-1]


def month_label(key: str) -> str:
    return datetime.strptime(key, "%Y-%m").strftime("%b %Y")


def _pick_reviews(rng: random.Random, count: int, avg: float) -> list[tuple[int, str]]:
    out = []
    for _ in range(count):
        stars = max(1, min(5, round(rng.gauss(avg, 0.8))))
        out.append((stars, rng.choice(REVIEW_POOL[stars])))
    return out


# =====================================================================
# Products
# =====================================================================
def get_products() -> list[dict[str, Any]]:
    return select_all("products", order=["created_at", "sku"])


def get_product(sku: str) -> dict[str, Any] | None:
    return one("products", sku=sku)


def insert_product(row: dict[str, Any]) -> dict[str, Any]:
    return client().table("products").insert(row).execute().data[0]


def update_product(sku: str, changes: dict[str, Any]) -> dict[str, Any] | None:
    rows = client().table("products").update(changes).eq("sku", sku).execute().data or []
    return rows[0] if rows else None


def delete_product(sku: str) -> bool:
    """Deleting a product also deletes its ads, posts and month totals (ON DELETE CASCADE)."""
    return bool(client().table("products").delete().eq("sku", sku).execute().data)


def add_stock(sku: str, qty: int) -> dict[str, Any] | None:
    """Atomic: a sale syncing at the same moment is never lost."""
    return rpc("nexus_add_stock", {"p_sku": sku, "p_qty": qty})


def selling_rates(products: list[dict[str, Any]]) -> dict[str, float]:
    """Units sold per day for each product, worked out from order history (no manual entry). One query."""
    months = month_keys(2)
    skus = [p["sku"] for p in products]
    rows = select_all("sku_month_units", filters=[("in_", "sku", skus), ("in_", "month", months)], order=["sku", "month"])
    units = {(r["sku"], r["month"]): r["units"] or 0 for r in rows}
    sd = sim_days()
    day = today_local().day
    out = {}
    for p in products:
        u_prev, u_now = units.get((p["sku"], months[0]), 0), units.get((p["sku"], months[1]), 0)
        if p.get("added_sim") is None:  # sample product with history: last month + this month
            out[p["sku"]] = round((u_prev + u_now) / (30 + day + sd), 1)
        else:  # new product: since it was added
            out[p["sku"]] = round(u_now / max(sd - p["added_sim"], 0.5), 1)
    return out


def selling_rate(p: dict[str, Any]) -> float:
    return selling_rates([p])[p["sku"]]


# =====================================================================
# Things To Do actions
# =====================================================================
def get_actions() -> list[dict[str, Any]]:
    return [{k: v for k, v in a.items() if k != "sort"} for a in select_all("actions", order="sort")]


def mark_action_done(action_id: str | None, target: str | None) -> dict[str, Any] | None:
    q = client().table("actions").update({"status": "done"})
    q = q.eq("id", action_id) if action_id else q.eq("target", target or "")
    rows = q.execute().data or []
    return rows[0] if rows else None


# =====================================================================
# Reading ads, posts and totals
# =====================================================================
def _totals_map(skus: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    rows = select_all("channel_totals", filters=[("in_", "sku", skus)], order=["sku", "platform"])
    return {(r["sku"], r["platform"]): {k: r.get(k) or 0 for k in METRICS} for r in rows}


def _ads_rows(skus: list[str] | None = None, status: str | None = None) -> list[dict[str, Any]]:
    f: list[tuple[str, str, Any]] = []
    if skus is not None:
        f.append(("in_", "sku", skus))
    if status:
        f.append(("eq", "status", status))
    return select_all("ads", filters=f, order=["sku", "created_at", "platform"])


def _posts_by_id(ids: list[str]) -> dict[str, dict[str, Any]]:
    ids = sorted({i for i in ids if i})
    return {r["id"]: r for r in select_all("posts", filters=[("in_", "id", ids)], order="id")} if ids else {}


POST_HIDDEN = ("rating_sum", "ends_sim", "watch_rate", "created_at")


def post_view(post: dict[str, Any], ad: dict[str, Any] | None, sd: float | None = None) -> dict[str, Any]:
    sd = sim_days() if sd is None else sd
    status = post["status"]
    if status == "live" and ad and ad.get("post_id") == post["id"]:
        status = {"active": "live", "paused": "paused", "stopped": "stopped"}[ad["status"]]
    left = None if post["ends_sim"] is None else max(0.0, post["ends_sim"] - sd)
    return {**{k: v for k, v in post.items() if k not in POST_HIDDEN}, "status": status,
            "format_label": AD_FORMATS[post["format"]]["label"],
            "avg_watch_pct": round(post["watch_rate"] * 100 + 22) if post["format"] != "image" else None,
            "avg_rating": round(post["rating_sum"] / post["reviews"], 1) if post["reviews"] else None,
            "click_rate_pct": round(post["clicks"] / post["views"] * 100, 2) if post["views"] else 0,
            "days_left": None if left is None else round(left, 1)}


def _ad_view(ad: dict[str, Any], product: dict[str, Any], totals: dict, posts: dict, conn: set[str], sd: float) -> dict[str, Any]:
    t = totals.get((ad["sku"], ad["platform"]), _ZERO)
    profit = t["units"] * (product["price"] - product["cost"]) - t["spend"]
    post = posts.get(ad.get("post_id") or "")
    return {
        "platform": ad["platform"], "status": ad["status"], "daily_budget": ad["daily_budget"], "started": ad["started"],
        "views": int(t["views"]), "clicks": int(t["clicks"]), "orders": int(t["units"]), "sales": round(t["revenue"]),
        "spend": round(t["spend"]), "profit_after_ads": round(profit),
        "click_rate_pct": round(t["clicks"] / t["views"] * 100, 2) if t["views"] else 0,
        "review_count": int(t["reviews"]),
        "avg_rating": round(t["rating_sum"] / t["reviews"], 1) if t["reviews"] else None,
        "recent_reviews": (ad.get("recent_reviews") or [])[-5:][::-1],
        "connected": connectors.is_connected(ad["platform"], conn),
        "post": post_view(post, ad, sd) if post else None,
    }


def ads_for_products(products: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """{sku: [ad view, ...]} for many products with a handful of queries."""
    skus = [p["sku"] for p in products]
    rows = _ads_rows(skus)
    totals = _totals_map(skus)
    posts = _posts_by_id([a.get("post_id") for a in rows])
    conn, sd = connectors.connected_set(), sim_days()
    by_sku = {p["sku"]: p for p in products}
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in skus}
    for a in rows:
        out[a["sku"]].append(_ad_view(a, by_sku[a["sku"]], totals, posts, conn, sd))
    return out


def ads_for(product: dict[str, Any]) -> list[dict[str, Any]]:
    return ads_for_products([product])[product["sku"]]


def _one_ad_view(product: dict[str, Any], platform: str) -> dict[str, Any] | None:
    return next((a for a in ads_for(product) if a["platform"] == platform), None)


def summary_for_ai(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact version for Gemini (no review text, so the prompt stays small)."""
    views = ads_for_products(products)
    out = []
    for p in products:
        for a in views[p["sku"]]:
            out.append({"sku": p["sku"], "product": p["name"], "price": p["price"], "cost": p["cost"],
                        **{k: v for k, v in a.items() if k not in ("recent_reviews", "post")}})
    return out


def monthly(products: list[dict[str, Any]], sku: str = "all", platform: str = "all", n: int = 6) -> dict[str, Any]:
    months = month_keys(n)
    skus = [p["sku"] for p in products] if sku == "all" else [sku]
    f: list[tuple[str, str, Any]] = [("in_", "month", months), ("in_", "sku", skus)]
    if platform != "all":
        f.append(("eq", "platform", platform))
    rows = select_all("monthly_metrics", filters=f, order=["sku", "platform", "month"])
    sums = {mk: dict(_ZERO) for mk in months}
    for r in rows:
        for k in METRICS:
            sums[r["month"]][k] += r[k] or 0
    out = []
    for mk in months:
        t = sums[mk]
        out.append({"month": mk, "label": month_label(mk), "units": int(t["units"]), "sales": round(t["revenue"]),
                    "views": int(t["views"]), "clicks": int(t["clicks"]), "ad_spend": round(t["spend"]),
                    "reviews": int(t["reviews"]),
                    "avg_rating": round(t["rating_sum"] / t["reviews"], 1) if t["reviews"] else None})
    return {"sku": sku, "platform": platform, "months": out}


# =====================================================================
# Changing ads
# =====================================================================
def _get_ad(sku: str, platform: str) -> dict[str, Any] | None:
    return one("ads", sku=sku, platform=platform)


def _activate_ad(product: dict[str, Any], platform: str, budget: float) -> None:
    if _get_ad(product["sku"], platform):
        client().table("ads").update({"status": "active", "daily_budget": budget}) \
            .eq("sku", product["sku"]).eq("platform", platform).execute()
    else:
        client().table("ads").insert({"sku": product["sku"], "platform": platform, "status": "active",
                                      "daily_budget": budget, "started": today_local().isoformat(),
                                      "recent_reviews": []}).execute()


def start_ad(product: dict[str, Any], platform: str, budget: float) -> dict[str, Any]:
    _activate_ad(product, platform, budget)
    return _one_ad_view(product, platform)


def change_ad(product: dict[str, Any], platform: str, status: str | None, budget: float | None) -> dict[str, Any] | None:
    changes: dict[str, Any] = {}
    if status:
        changes["status"] = status
    if budget is not None:
        changes["daily_budget"] = budget
    if not changes:
        return _one_ad_view(product, platform)
    rows = client().table("ads").update(changes).eq("sku", product["sku"]).eq("platform", platform).execute().data
    if not rows:
        return None
    return _one_ad_view(product, platform)


def remove_product(sku: str) -> None:
    """Kept for compatibility: the database deletes ads, posts and month totals with the product."""
    return None


# =====================================================================
# Orders from platforms (simulated until real APIs are connected)
# =====================================================================
# How many orders a daily ad budget brings. More budget helps, but with diminishing returns: the audience
# is limited, so orders level off at CEILING times the usual number. Used by the order simulation,
# the "what if" slider and the predictions of the Things To Do page, so all three agree.
CEILING = 2.5
_S_FACTOR = -1 / math.log(1 - 1 / CEILING)  # makes the curve pass through (usual budget, usual orders)


def expected_units(p: dict[str, Any], platform: str, ad: dict[str, Any], budget: float | None = None) -> float:
    """Expected orders per day from one ad at a given daily budget (default: its current budget)."""
    b = float(ad["daily_budget"] if budget is None else budget)
    if b <= 0:
        return 0.0
    if ad.get("seed_rate") and ad.get("seed_budget"):
        u0, b0 = float(ad["seed_rate"]), float(ad["seed_budget"])
    else:
        vpr, ctr, cr = PLATFORM_RATES[platform]
        u0, b0 = 1000 * vpr * ctr * cr * 0.3, 1000.0
    return CEILING * u0 * (1 - math.exp(-b / (_S_FACTOR * b0)))


def _demand_per_day(p: dict[str, Any], platform: str, ad: dict[str, Any] | None) -> float:
    """Expected orders per day from one channel."""
    if platform == "Website":
        return p.get("website_rate") or 1.0
    return expected_units(p, platform, ad)


def _count(r: random.Random, expected: float) -> int:
    n = int(expected)
    return n + (1 if r.random() < expected - n else 0)


def generate_orders(days: float) -> dict[str, Any]:
    """Bring in `days` worth of orders, views and reviews from every connected channel.
    Everything is calculated here, then written in one database transaction."""
    out = {"orders": 0, "units": 0, "views": 0, "reviews": 0}
    if days <= 0:
        return out
    _end_finished_posts()
    products = [p for p in get_products() if p["status"] == "active"]
    if not products:
        return out
    ads_active = _ads_rows([p["sku"] for p in products], status="active")
    posts = _posts_by_id([a.get("post_id") for a in ads_active])
    conn = connectors.connected_set()
    mk = month_keys(1)[0]
    stamp = now_local().strftime("%d %b, %I:%M:%S %p")
    today_iso = today_local().isoformat()
    r = random.Random()
    payload: dict[str, list] = {"stock": [], "monthly": [], "posts": [], "orders": [], "reviews": []}
    synced: set[str] = set()

    for p in products:
        stock_left, sold = p["stock"], 0
        channels = [(a["platform"], a) for a in ads_active if a["sku"] == p["sku"]] + [("Website", None)]
        for plat, ad in channels:
            if not connectors.is_connected(plat, conn):
                continue  # nothing syncs from a platform that is not connected
            synced.add(plat)
            post = posts.get(ad.get("post_id") or "") if ad else None
            c = {"sku": p["sku"], "platform": plat, "month": mk, **_ZERO}
            pd = {"id": post["id"], "views": 0, "clicks": 0, "orders": 0, "sales": 0, "spend": 0,
                  "reviews": 0, "rating_sum": 0, "completed_views": 0} if post else None
            new_reviews: list[dict[str, Any]] = []
            vm, cm, om = FORMAT_EFFECT[post["format"]] if post else (1.0, 1.0, 1.0)
            if ad:  # running ads get views and cost money even when stock is out
                vpr, ctr, _cr = PLATFORM_RATES[plat]
                views = int(ad["daily_budget"] * vpr * days * vm * r.uniform(0.85, 1.15))
                clicks = int(views * ctr * cm * r.uniform(0.85, 1.15))
                spend = round(ad["daily_budget"] * days, 2)
                c["views"] += views
                c["clicks"] += clicks
                c["spend"] += spend
                out["views"] += views
                if pd:
                    pd["views"] += views
                    pd["clicks"] += clicks
                    pd["spend"] += spend
                    if post["format"] != "image":  # people who watched the whole video
                        pd["completed_views"] += int(views * post["watch_rate"])
            units = min(_count(r, _demand_per_day(p, plat, ad) * days * om), stock_left)
            while units > 0:
                qty = min(units, 2 if r.random() < 0.15 else 1)
                units -= qty
                stock_left -= qty
                sold += qty
                c["units"] += qty
                c["revenue"] += qty * p["price"]
                if pd:
                    pd["orders"] += qty
                    pd["sales"] += qty * p["price"]
                if not ad:
                    c["views"] += qty * r.randint(30, 50)
                payload["orders"].append({"id": "ORD-" + uuid.uuid4().hex[:6].upper(), "time": stamp, "sku": p["sku"],
                                          "product": p["name"], "platform": plat, "quantity": qty,
                                          "amount": qty * p["price"]})
                out["orders"] += 1
                out["units"] += qty
                if r.random() < 0.07:  # some buyers leave a review
                    stars, text = _pick_reviews(r, 1, 4.2)[0]
                    c["reviews"] += 1
                    c["rating_sum"] += stars
                    out["reviews"] += 1
                    if pd:
                        pd["reviews"] += 1
                        pd["rating_sum"] += stars
                    if ad:
                        new_reviews.append({"stars": stars, "text": text, "date": today_iso})
            if any(c[k] for k in METRICS):
                c["spend"] = round(c["spend"], 2)
                payload["monthly"].append(c)
            if pd and any(v for k, v in pd.items() if k != "id"):
                pd["spend"] = round(pd["spend"], 2)
                payload["posts"].append(pd)
            if new_reviews:
                payload["reviews"].append({"sku": p["sku"], "platform": plat, "items": new_reviews})
        if sold:
            payload["stock"].append({"sku": p["sku"], "qty": sold})

    if any(payload.values()):
        rpc("nexus_apply_sync", {"p": payload})
    connectors.mark_sync_many(synced)
    out["new_orders"] = payload["orders"]  # pushed to the dashboards over the WebSocket
    out["stock"] = payload["stock"]
    return out


def sync_orders(products: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pull every order placed since the last sync (called automatically by the dashboard).
    The database hands out each time slice once, so two servers never count the same orders twice."""
    global _sim_start
    label = now_local().strftime("%I:%M:%S %p")
    claim = rpc("nexus_claim_sync", {"p_label": label})
    _sim_start = float(claim["start"])
    days = min(max(float(claim["now"]) - float(claim["prev"]), 0.0) * SIM_SPEED / 86400, 2.0)
    return {**generate_orders(days), "synced_at": label}


def bonus_days() -> float:
    """Days fast-forwarded with the demo button (they count as time passed when measuring results)."""
    row = one("app_state", key="sim_bonus")
    return float(((row or {}).get("value") or {}).get("days", 0))


def simulate_day(products: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """DEMO button: bring in a whole day of orders at once."""
    res = generate_orders(1.0)
    client().table("app_state").upsert({"key": "sim_bonus", "value": {"days": bonus_days() + 1}}).execute()
    conn = connectors.connected_set()
    running = sum(1 for a in _ads_rows(status="active") if connectors.is_connected(a["platform"], conn))
    return {**res, "ads": running}


def orders_summary(products: list[dict[str, Any]]) -> dict[str, Any]:
    """Total orders per product per platform (all time), plus the latest orders."""
    skus = [p["sku"] for p in products]
    by_product: dict[str, dict[str, int]] = {s: {} for s in skus}
    for (s, pl), t in _totals_map(skus).items():
        if t["units"]:
            by_product[s][pl] = int(t["units"])
    recent = [{k: v for k, v in o.items() if k != "created_at"}
              for o in select_all("orders", order="created_at", desc=True, limit=25)]
    clock = rpc("nexus_clock") or {}
    return {"by_product": by_product, "recent": recent, "connected": connectors.connected_names(),
            "last_sync": clock.get("label", ""), "speed": SIM_SPEED}


# =====================================================================
# Posting ads
# =====================================================================
def _end_finished_posts() -> None:
    ended = select_all("posts", columns="id", filters=[("eq", "status", "live"), ("lte", "ends_sim", sim_days())], order="id")
    ids = [x["id"] for x in ended]
    if ids:
        client().table("posts").update({"status": "ended"}).in_("id", ids).execute()
        client().table("ads").update({"status": "stopped"}).in_("post_id", ids).execute()


def post_ad(product: dict[str, Any], platforms: list[str], creative: dict[str, Any], budget: float,
            days: int | None) -> list[dict[str, Any]]:
    """Publish one ad for a product on each chosen platform.
    Demo mode: an ID is made up and numbers are simulated. Live mode: TODO call connectors.post_ad()."""
    modes = {k: v.get("mode", "demo") for k, v in connectors.list_connections_map().items()}
    sd = sim_days()
    made = []
    for plat in platforms:
        ad = _get_ad(product["sku"], plat)
        if ad and ad.get("post_id"):  # the new ad takes over this product's slot on the platform
            client().table("posts").update({"status": "replaced"}).eq("id", ad["post_id"]).eq("status", "live").execute()
        _activate_ad(product, plat, budget)
        post = {"id": f"{plat[:2].upper()}-" + uuid.uuid4().hex[:8].upper(),
                "sku": product["sku"], "product": product["name"], "platform": plat,
                "headline": creative["headline"], "text": creative["text"], "cta": creative["cta"],
                "format": creative["format"], "media_url": creative.get("media_url") or "",
                "media_type": creative.get("media_type") or "", "media_seconds": creative.get("media_seconds"),
                "daily_budget": budget, "duration_days": days,
                "posted_at": now_local().strftime("%d %b %Y, %I:%M %p"),
                "ends_sim": None if not days else sd + days, "status": "live",
                "views": 0, "clicks": 0, "orders": 0, "sales": 0, "spend": 0, "reviews": 0, "rating_sum": 0,
                "completed_views": 0,
                "watch_rate": random.uniform(0.18, 0.35) if creative["format"] == "reel" else random.uniform(0.10, 0.25),
                "mode": modes.get(plat, "demo")}
        saved = client().table("posts").insert(post).execute().data[0]
        ad_now = client().table("ads").update({"post_id": saved["id"]}) \
            .eq("sku", product["sku"]).eq("platform", plat).execute().data[0]
        made.append(post_view(saved, ad_now, sd))
    return made


def posts_list(sku: str = "all") -> list[dict[str, Any]]:
    _end_finished_posts()
    f = [] if sku == "all" else [("eq", "sku", sku)]
    rows = select_all("posts", filters=f, order="created_at", desc=True)
    ads_map = {(a["sku"], a["platform"]): a for a in _ads_rows(sorted({x["sku"] for x in rows}))} if rows else {}
    sd = sim_days()
    return [post_view(x, ads_map.get((x["sku"], x["platform"])), sd) for x in rows]


def remove_posts_for(sku: str) -> None:
    client().table("posts").delete().eq("sku", sku).execute()


# =====================================================================
# Sample data: created once, the first time the app starts with an empty database
# =====================================================================
SAMPLE_PRODUCTS: list[dict[str, Any]] = [
    {"sku": "LM-204", "units_sold": 1840, "name": "Linen Shirt", "category": "Shirts", "price": 1999, "cost": 760, "stock": 56, "base_rate": 14, "status": "active", "ad_spend": 38000},
    {"sku": "LM-117", "units_sold": 2210, "name": "Chino Pants", "category": "Pants", "price": 2499, "cost": 1050, "stock": 380, "base_rate": 20, "status": "active", "ad_spend": 22500},
    {"sku": "LM-330", "units_sold": 640, "name": "Overshirt", "category": "Shirts", "price": 3299, "cost": 1400, "stock": 54, "base_rate": 9, "status": "active", "ad_spend": 15200},
    {"sku": "LM-051", "units_sold": 3920, "name": "Basic Tee", "category": "Tops", "price": 799, "cost": 290, "stock": 1260, "base_rate": 30, "status": "active", "ad_spend": 9800},
]

SAMPLE_ACTIONS: list[dict[str, Any]] = [
    {"id": "A-101", "type": "pause", "target": "Meta: Summer Drop – UGC v3", "change": "Stop ad", "impact": "Save about ₹18,400 a day", "confidence": 0.93, "status": "pending"},
    {"id": "A-102", "type": "realloc", "target": "TikTok: Founder Story", "change": "Add ₹45,000 a day", "impact": "More profit per ₹1 spent", "confidence": 0.88, "status": "pending"},
    {"id": "A-103", "type": "pause", "target": "Amazon: Sponsored ad, Linen Shirt", "change": "Stop until restock", "impact": "Avoids 3 days out of stock", "confidence": 0.96, "status": "pending"},
    {"id": "A-104", "type": "realloc", "target": "Meta: Retargeting to New customers", "change": "Move ₹20,000 a day", "impact": "About ₹9,100 more a day", "confidence": 0.74, "status": "pending"},
]


def seed() -> None:
    """Create believable history for the sample products (6 months, several platforms).
    Monthly units come from each product's selling rate, so all pages agree with each other."""
    months = month_keys(6)
    products = [dict(p) for p in SAMPLE_PRODUCTS]
    ads_rows: dict[tuple[str, str], dict[str, Any]] = {}
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in products:
        r = _rng("platforms", p["sku"])
        chosen = r.sample(AD_PLATFORMS, k=r.randint(3, 4))
        weights = [r.uniform(0.6, 1.4) for _ in chosen] + [0.35]  # last one is the website
        shares = [w / sum(weights) for w in weights]
        for i, plat in enumerate(chosen):
            status = "paused" if i == len(chosen) - 1 and r.random() < 0.4 else "active"
            ads_rows[(p["sku"], plat)] = {"sku": p["sku"], "platform": plat, "status": status,
                                          "daily_budget": r.choice([300, 500, 800, 1000, 1500]),
                                          "started": months[0] + "-01", "recent_reviews": []}
        for mi, mk in enumerate(months):
            days = today_local().day if mi == len(months) - 1 else 30
            month_units = p["base_rate"] * days * (0.8 + mi * 0.06)
            for plat, share in zip(chosen + ["Website"], shares):
                mr = _rng("month", p["sku"], plat, mk)
                units = int(month_units * share * mr.uniform(0.85, 1.15))
                reviews = int(units * mr.uniform(0.04, 0.09))
                avg = _rng("avg", p["sku"], plat).uniform(3.7, 4.7)
                c = {"sku": p["sku"], "platform": plat, "month": mk, **_ZERO,
                     "units": units, "revenue": units * p["price"], "reviews": reviews,
                     "rating_sum": round(reviews * avg * mr.uniform(0.96, 1.03), 1)}
                if plat == "Website":
                    c["views"] = units * mr.randint(30, 50)
                else:
                    _vpr, ctr, cr = PLATFORM_RATES[plat]
                    c["clicks"] = int(units / cr * mr.uniform(0.9, 1.1))
                    c["views"] = int(c["clicks"] / ctr * mr.uniform(0.9, 1.1))
                    c["spend"] = ads_rows[(p["sku"], plat)]["daily_budget"] * days
                cells[(p["sku"], plat, mk)] = c
        for plat in chosen:
            avg = _rng("avg", p["sku"], plat).uniform(3.7, 4.7)
            for stars, text in _pick_reviews(_rng("rev", p["sku"], plat), 6, avg):
                ads_rows[(p["sku"], plat)]["recent_reviews"].append({"stars": stars, "text": text, "date": months[-1]})
        # remember each channel's normal orders per day, so synced orders follow the same pattern
        for plat, share in zip(chosen, shares):
            ad = ads_rows[(p["sku"], plat)]
            ad["seed_rate"], ad["seed_budget"] = p["base_rate"] * share, ad["daily_budget"]
        p["website_rate"] = p["base_rate"] * shares[-1]
        # keep the product's "units sold" equal to the history
        p["units_sold"] = int(sum(c["units"] for (s_, _pl, _m), c in cells.items() if s_ == p["sku"]))

    sb = client()
    sb.table("products").upsert(products, on_conflict="sku").execute()  # products first (others refer to them)
    sb.table("ads").upsert(list(ads_rows.values()), on_conflict="sku,platform").execute()
    sb.table("monthly_metrics").upsert(list(cells.values()), on_conflict="sku,platform,month").execute()
    sb.table("actions").upsert([{**a, "sort": i} for i, a in enumerate(SAMPLE_ACTIONS)], on_conflict="id").execute()


def ensure_seeded() -> bool:
    """Create the sample data once, the very first time. Safe when several servers start together.
    Turn off with SEED_SAMPLE_DATA=0. To create it again: delete from app_state where key = 'seeded';"""
    if os.getenv("SEED_SAMPLE_DATA", "1").strip() != "1":
        return False
    if not rpc("nexus_try_lock", {"p_key": "seeded"}):
        return False
    seed()
    return True
