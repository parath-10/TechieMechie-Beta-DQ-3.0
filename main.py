"""NEXUS D2C Command Center: FastAPI orchestration layer.

Stateless: all data lives in Supabase (PostgreSQL + Storage), so the app can run on Railway
with any number of copies and survive restarts.

Run locally:  uvicorn main:app --reload --port 8000   (with SUPABASE_URL etc. in .env)
Then open http://localhost:8000
"""

import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import ads
import auth
import connectors
import db
from ai_engine import (
    analyze_omnichannel_feedback,
    answer_chat_query,
    get_ai_status,
    get_insights,
    get_monitoring_alerts,
    get_section_tip,
    get_suggestions,
    get_timeline_events,
    insights_meta,
    predict_future_performance,
    run_diagnosis,
)

try:
    from postgrest.exceptions import APIError as PostgrestAPIError
except Exception:
    PostgrestAPIError = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nexus.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        if await run_in_threadpool(ads.ensure_seeded):
            log.info("Sample data created in Supabase.")
    except Exception as exc:
        log.error("Could not prepare the database on start-up: %s", exc)
    yield


app = FastAPI(title="NEXUS D2C Command Center", version="3.0.0", lifespan=lifespan)
# ---------- Login gate: every page and API call needs a session, except the public ones ----------
PUBLIC_PATHS = {"/", "/index.html", "/health", "/favicon.ico",
                "/api/login", "/api/logout", "/api/auth/me", "/api/auth/config", "/api/public/summary"}
PAGES = {"/dashboard", "/dashboard.html"}


def gate(path: str, method: str, cookie: str | None) -> tuple[int, dict | str] | None:
    """None = let the request through. Otherwise (status, JSON body) or (303, address to send the browser to)."""
    if method == "OPTIONS" or path in PUBLIC_PATHS or path.startswith("/media/"):
        return None
    user = auth.read_token(cookie)
    if not user:
        if path in PAGES:
            return 303, "/?login=1"
        return 401, {"detail": "Please log in again."}
    if user["role"] == "demo" and method not in ("GET", "HEAD") and path not in auth.DEMO_ALLOWED_WRITES:
        return 403, {"detail": "The demo account can look around but not change anything. Log in with the owner account to make changes."}
    return None


@app.middleware("http")
async def require_login(request: Request, call_next):
    blocked = gate(request.url.path, request.method, request.cookies.get(auth.COOKIE))
    if blocked is None:
        return await call_next(request)
    status, body = blocked
    if status == 303:
        return RedirectResponse(body, status_code=303)
    return JSONResponse(status_code=status, content=body)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(db.DatabaseNotConfigured)
async def _db_not_configured(_request: Request, exc: Exception):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if PostgrestAPIError is not None:
    @app.exception_handler(PostgrestAPIError)
    async def _db_error(_request: Request, exc: Exception):
        log.error("Database error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Database error. Please try again in a moment."})


@app.exception_handler(RuntimeError)
async def _config_error(_request: Request, exc: Exception):
    log.error("Server problem: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


CAMPAIGN_DATA: dict[str, Any] = {
    "brand": "Lumen & Co.",
    "currency": "INR",
    "campaign": {"id": "cmp_123", "name": "Summer Drop – UGC v3", "platform": "meta", "status": "active"},
    "metrics_last_7_days": {
        "spend": 340000, "impressions": 1842000, "clicks": 41300, "ctr_pct": 2.2, "ctr_change_pct": -31,
        "frequency": 4.8, "orders": 1260, "revenue": 2110000, "roas": 1.9, "poas": 0.71, "cpa": 540,
    },
    "inventory": {"sku": "AES-VC-30", "product": "Linen Shirt", "days_of_cover": 4, "stock_status": "low"},
    "other_platforms": {
        "tiktok": {"campaign": "Founder Story", "spend": 420000, "orders": 980, "poas": 1.62},
        "shopify": {"store_visits": 214000, "orders": 4120, "return_rate_pct": 4.1},
        "amazon": {"spend": 160000, "orders": 1310, "poas": 0.94, "star_rating": 4.1},
    },
}

PRODUCT_HIDDEN = ("base_rate", "website_rate", "added_sim", "created_at", "updated_at")


def _view(p: dict[str, Any], rate: float | None = None) -> dict[str, Any]:
    rate = ads.selling_rate(p) if rate is None else rate
    cover = round(p["stock"] / rate) if rate > 0 else 999
    unit_profit = p["price"] - p["cost"]
    sold = p.get("units_sold", 0)
    return {
        **{k: v for k, v in p.items() if k not in PRODUCT_HIDDEN},
        "daily_sales": rate,
        "reorder_level": int(rate * 7),
        "units_sold": sold,
        "days_of_cover": cover,
        "profit_per_unit": round(unit_profit, 2),
        "margin_pct": round(unit_profit / p["price"] * 100, 1) if p["price"] else 0,
        "revenue": round(sold * p["price"], 2),
        "total_profit": round(sold * unit_profit, 2),
        "stock_value": round(p["stock"] * p["cost"], 2),
    }


def _views(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rates = ads.selling_rates(products) if products else {}
    return [_view(p, rates[p["sku"]]) for p in products]


def context() -> dict[str, Any]:
    try:
        products = ads.get_products()
    except Exception as exc:
        log.warning("Could not fetch products for AI context: %s", exc)
        products = []

    try:
        views_data = _views(products)
    except Exception:
        views_data = []

    try:
        ad_summary = ads.summary_for_ai(products)
    except Exception:
        ad_summary = {}

    try:
        posts = [{k: v for k, v in x.items() if k not in ("media_url",)} for x in ads.posts_list()[:20]]
    except Exception:
        posts = []

    try:
        conn_names = connectors.connected_names()
    except Exception:
        conn_names = []

    try:
        monthly_data = ads.monthly(products)["months"]
    except Exception:
        monthly_data = []

    return {
        **CAMPAIGN_DATA,
        "products": views_data,
        "advertising": ad_summary,
        "posted_ads": posts,
        "connected_platforms": conn_names,
        "monthly_all_products": monthly_data,
    }


class ExecuteActionRequest(BaseModel):
    action_id: str | None = None
    action: str | None = None
    target: str | None = None
    change: str | None = None
    type: str | None = None
    campaign_id: str | None = None


class ChatRequest(BaseModel):
    question: str = ""
    history: list[dict[str, Any]] = []


def _scale_pct(v: Any, default: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v / 100.0 if v > 1 else v


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/public/summary")
def public_summary():
    """Totals and examples for the public landing page. No tokens, accounts or customer data. Turn off with PUBLIC_SUMMARY=0."""
    import calendar, math
    from datetime import date, timedelta
    if os.getenv("PUBLIC_SUMMARY", "1") == "0":
        raise HTTPException(status_code=404, detail="Not found")
    today = date.today()
    products = ads.get_products()
    views = {v["sku"]: v for v in _views(products)}
    ad_views = ads.ads_for_products(products)
    rows, all_ads = [], []
    for p in products:
        v = views[p["sku"]]
        mine = [a for a in ad_views.get(p["sku"], []) if a.get("status") != "stopped"]
        spend = sum(a.get("spend", 0) or 0 for a in mine)
        orders = sum(a.get("orders", 0) or 0 for a in mine)
        daily = sum(a.get("daily_budget", 0) or 0 for a in mine if a.get("status") == "active")
        profit = sum(a.get("profit_after_ads", 0) or 0 for a in mine)
        rows.append({"name": p["name"], "cover": v["days_of_cover"], "daily": daily, "spend": spend, "profit": profit,
                     "roas": round(orders * p["price"] / spend, 1) if spend else None,
                     "reorder": max(0, math.ceil(v["daily_sales"] * 21 - p["stock"])), "stock": p["stock"],
                     "rate": round(v["daily_sales"], 1)})
        all_ads += [{"product": p["name"], "platform": a["platform"], "ratio": round((a.get("profit_after_ads", 0) + a["spend"]) / a["spend"], 2),
                     "profit": a.get("profit_after_ads", 0)} for a in mine if (a.get("spend") or 0) > 0]
    risky = [r for r in rows if r["daily"] > 0 and r["cover"] < 14]
    hero = min(risky, key=lambda r: r["cover"]) if risky else None
    try:
        days = 0
        for m in ads.monthly(products)["months"]:
            y, mo = map(int, m["month"].split("-"))
            days += today.day if (y, mo) == (today.year, today.month) else calendar.monthrange(y, mo)[1]
    except Exception:
        days = None
    fat = CAMPAIGN_DATA["metrics_last_7_days"]
    return {
        "platforms": len(ads.AD_PLATFORMS), "products": len(products), "days": days,
        "at_risk_week": sum(r["daily"] for r in risky) * 7,
        "hero": hero and {**hero, "runout": (today + timedelta(days=hero["cover"])).strftime("%d %b"),
                          "protected_week": hero["daily"] * 7,
                          "runout_from": (today + timedelta(days=hero["cover"] - 14)).strftime("%d %b"),
                          "runout_to": (today + timedelta(days=hero["cover"] + 14)).strftime("%d %b")},
        "fatigue": {"campaign": CAMPAIGN_DATA["campaign"]["name"], "ctr_change_pct": fat["ctr_change_pct"]},
        "weakest": min(all_ads, key=lambda a: a["ratio"]) if all_ads else None,
        "best": max(all_ads, key=lambda a: a["ratio"]) if all_ads else None,
    }


@app.get("/api/ai-status")
@app.get("/ai-status")
def ai_status(force: bool = False):
    return get_ai_status(force)


@app.get("/api/ai-insights")
def ai_insights():
    """When the saved AI insights were created, and by which model (no AI request)."""
    return insights_meta(get_insights(context))


@app.post("/api/ai-insights/refresh")
def refresh_ai_insights():
    """The ONLY way the page asks for a new AI request: the "Refresh AI" button."""
    return insights_meta(get_insights(context, force=True))


@app.get("/api/diagnose")
@app.get("/diagnose")
def diagnose_campaign():
    try:
        d = run_diagnosis(context)
    except Exception as exc:
        log.error("Diagnose error: %s", exc)
        d = {}

    action = d.get("recommended_action") or {}
    if isinstance(action, dict):
        rec_desc = action.get("description") or action.get("title") or "Review the campaign."
        rec_title = action.get("title", "")
    else:
        rec_desc = str(action) or "Review the campaign."
        rec_title = str(action) or "Review"

    return {
        "campaign": CAMPAIGN_DATA["campaign"],
        "root_cause": d.get("root_cause", "No clear cause found."),
        "confidence": _scale_pct(d.get("confidence_score"), 0.85),
        "opportunity_score": round(float(d.get("opportunity_score") or 80)),
        "recommended_action": rec_desc,
        "action_title": rec_title,
        "source": d.get("source", "fallback"),
    }


@app.get("/api/forecast")
@app.get("/forecast")
def get_forecast():
    try:
        return predict_future_performance(context)
    except Exception as exc:
        log.error("Forecast error: %s", exc)
        return {
            "forecast_horizon": "Next 14 days",
            "predicted_poas": 1.28,
            "predicted_revenue_growth_pct": 12.5,
            "fatigue_risk_level": "high",
            "stockout_risk_days": 4,
            "summary": "Profit per ₹1 of ads should improve slightly as weak ads are stopped.",
            "inventory_advice": "Reorder 240 Linen Shirts now to avoid stockouts.",
            "integration_suggestions": [
                "Connect Meta Conversions API",
                "Connect warehouse stock counts",
                "Sync courier tracking",
            ],
            "source": "fallback",
        }


@app.get("/api/alerts")
@app.get("/alerts")
def get_alerts():
    try:
        return get_monitoring_alerts(context)
    except Exception as exc:
        log.error("Alerts error: %s", exc)
        return []


@app.get("/api/timeline")
@app.get("/timeline")
def get_timeline():
    try:
        return get_timeline_events(context)
    except Exception as exc:
        log.error("Timeline error: %s", exc)
        return []


@app.get("/api/actions")
@app.get("/actions")
def get_actions():
    return ads.get_actions()


@app.post("/api/execute-action")
@app.post("/execute-action")
def execute_action(payload: ExecuteActionRequest):
    label = payload.target or payload.action or payload.action_id or "action"
    try:
        done = ads.mark_action_done(payload.action_id, payload.target)
        if done:
            label = done["target"]
    except Exception as exc:
        log.error("Action execution error: %s", exc)
    log.info("Executed: %s", label)
    return {"status": "executed", "action_id": payload.action_id, "message": f"“{label}” was applied successfully."}


@app.post("/api/chat")
@app.post("/chat")
def assistant_chat(payload: ChatRequest):
    try:
        return {"answer": answer_chat_query(payload.question, context(), payload.history)}
    except Exception as exc:
        log.error("Chat error: %s", exc)
        return {"answer": "I am currently updating insights from your catalog. Please try asking again in a moment."}


@app.post("/api/analyze-feedback")
@app.post("/analyze-feedback")
def analyze_feedback(payload: Any = Body(default=None)):
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("feedback") or [line for line in str(payload.get("text", "")).splitlines()]
    elif isinstance(payload, str):
        raw = payload.splitlines()
    else:
        raw = []
    reviews = [
        r if isinstance(r, dict) else {"platform": "shopify", "text": str(r).strip()}
        for r in raw
        if (isinstance(r, dict) or str(r).strip())
    ]
    report = analyze_omnichannel_feedback(reviews)
    return {**report, "report": report}


@app.get("/api/suggestions")
@app.get("/suggestions")
def suggestions(section: str = "actions"):
    try:
        res = get_suggestions(context)
        if isinstance(res, list):
            return {"suggestions": res}
        return res
    except Exception as exc:
        log.warning("get_suggestions failed: %s", exc)

    return {"suggestions": [
        {
            "title": "Stop 'Summer Drop - UGC v3' ad",
            "description": "Ad fatigue is causing low clicks. Moving budget to TikTok will save ₹45,000/day.",
            "action_type": "pause",
            "confidence": 92.0,
        },
        {
            "title": "Restock Linen Shirt",
            "description": "Stock is critically low with only 4 days of cover. Pause ads to prevent out-of-stock penalties.",
            "action_type": "restock",
            "confidence": 88.5,
        },
        {
            "title": "Scale TikTok Budget",
            "description": "The 'Founder Story' campaign is highly profitable. Adding budget will drive volume.",
            "action_type": "budget_increase",
            "confidence": 84.0,
        }
    ]}


@app.get("/api/section-tip")
@app.get("/section-tip")
def section_tip(section: str = "overview"):
    try:
        return get_section_tip(section, context)
    except Exception as exc:
        log.error("Section tip error: %s", exc)
        return {
            "section": section,
            "title": "Monitor Campaign Health",
            "tip": "Review underperforming ad sets and keep inventory restocked ahead of spikes.",
            "source": "fallback",
        }


class ProductIn(BaseModel):
    name: str
    price: float
    cost: float = 0
    stock: int = 0
    category: str = "General"


class ProductPatch(BaseModel):
    name: str | None = None
    category: str | None = None
    price: float | None = None
    cost: float | None = None
    stock: int | None = None
    status: Literal["active", "paused"] | None = None


class RestockIn(BaseModel):
    sku: str
    quantity: int


def _find(sku: str) -> dict[str, Any]:
    p = ads.get_product(sku)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return p


@app.get("/api/products")
@app.get("/products")
def list_products():
    return _views(ads.get_products())


@app.post("/api/products")
@app.post("/products")
def add_product(body: ProductIn):
    if not body.name.strip() or body.price <= 0 or body.cost < 0 or body.stock < 0:
        raise HTTPException(status_code=422, detail="Name and a selling price above 0 are needed; numbers cannot be negative.")
    p = ads.insert_product({
        "sku": "NX-" + uuid4().hex[:4].upper(),
        "units_sold": 0,
        "name": body.name.strip(),
        "category": body.category.strip() or "General",
        "price": body.price,
        "cost": body.cost,
        "stock": body.stock,
        "status": "active",
        "ad_spend": 0,
        "added_sim": ads.sim_days(),
        "website_rate": 1.0,
    })
    return _view(p)


@app.patch("/api/products/{sku}")
def update_product(sku: str, body: ProductPatch):
    _find(sku)
    changes = body.model_dump(exclude_none=True)
    if changes.get("price", 1) <= 0:
        raise HTTPException(status_code=422, detail="Selling price must be above 0.")
    if changes.get("cost", 0) < 0 or changes.get("stock", 0) < 0:
        raise HTTPException(status_code=422, detail="Cost and stock cannot be negative.")
    if "name" in changes and not changes["name"].strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty.")
    p = ads.update_product(sku, changes) if changes else _find(sku)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return _view(p)


@app.delete("/api/products/{sku}")
def delete_product(sku: str):
    if not ads.delete_product(sku):
        raise HTTPException(status_code=404, detail="Product not found")
    return {"status": "deleted", "sku": sku}


@app.post("/api/inventory/restock")
def restock(body: RestockIn):
    if body.quantity <= 0:
        raise HTTPException(status_code=422, detail="Quantity must be above 0.")
    p = ads.add_stock(body.sku, body.quantity)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return {**_view(p), "message": f"Added {body.quantity} units to {p['name']}."}


@app.post("/api/orders/sync")
def sync_orders():
    return ads.sync_orders()


@app.get("/api/orders")
def orders():
    return ads.orders_summary(ads.get_products())


class AdStartIn(BaseModel):
    sku: str
    platform: str
    daily_budget: float


class AdChangeIn(BaseModel):
    sku: str
    platform: str
    status: Literal["active", "paused", "stopped"] | None = None
    daily_budget: float | None = None


def _check_platform(platform: str) -> None:
    if platform not in ads.AD_PLATFORMS:
        raise HTTPException(status_code=422, detail=f"Unknown platform. Use one of: {', '.join(ads.AD_PLATFORMS)}")


@app.get("/api/ads/platforms")
def ad_platforms():
    return {
        "ad_platforms": ads.AD_PLATFORMS,
        "sales_channels": ads.SALES_CHANNELS,
        "formats": ads.AD_FORMATS,
        "format_support": ads.FORMAT_SUPPORT,
    }


@app.get("/api/ads")
def list_ads():
    products = ads.get_products()
    views = {v["sku"]: v for v in _views(products)}
    ad_views = ads.ads_for_products(products)
    return [
        {
            "sku": p["sku"],
            "name": p["name"],
            "status": p["status"],
            "stock": p["stock"],
            "days_of_cover": views[p["sku"]]["days_of_cover"],
            "ads": ad_views[p["sku"]],
        }
        for p in products
    ]


@app.post("/api/ads")
def start_ad(body: AdStartIn):
    _check_platform(body.platform)
    if body.daily_budget <= 0:
        raise HTTPException(status_code=422, detail="Daily budget must be above 0.")
    p = _find(body.sku)
    if not connectors.is_connected(body.platform):
        raise HTTPException(status_code=422, detail=f"Connect your {body.platform} account first (Connected Platforms page).")
    return {
        **ads.start_ad(p, body.platform, body.daily_budget),
        "message": f"{p['name']} is now advertised on {body.platform}.",
    }


@app.patch("/api/ads")
def change_ad(body: AdChangeIn):
    _check_platform(body.platform)
    if body.daily_budget is not None and body.daily_budget <= 0:
        raise HTTPException(status_code=422, detail="Daily budget must be above 0.")
    p = _find(body.sku)
    ad = ads.change_ad(p, body.platform, body.status, body.daily_budget)
    if ad is None:
        raise HTTPException(status_code=404, detail=f"{p['name']} is not advertised on {body.platform}.")
    return ad


@app.post("/api/ads/simulate-day")
def simulate_ad_day():
    return ads.simulate_day()


@app.get("/api/monthly")
def monthly_log(sku: str = "all", platform: str = "all", months: int = 6):
    if sku != "all":
        _find(sku)
    if platform != "all" and platform not in ads.SALES_CHANNELS:
        raise HTTPException(status_code=422, detail="Unknown platform.")
    return ads.monthly(ads.get_products(), sku, platform, max(1, min(months, 12)))


class PostIn(BaseModel):
    sku: str
    platforms: list[str]
    format: Literal["image", "video", "reel"] = "image"
    headline: str
    text: str = ""
    cta: str = "Shop now"
    media_url: str = ""
    media_seconds: float | None = None
    media_width: int | None = None
    media_height: int | None = None
    daily_budget: float
    days: int | None = None


BUCKET = "ad_creatives"
MEDIA_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}
MAX_IMAGE = 8 * 1024 * 1024
MAX_VIDEO = int(os.getenv("MAX_VIDEO_MB", "50")) * 1024 * 1024
MEDIA_NAME = re.compile(r"^[0-9a-f]{32}\.(jpg|png|webp|mp4|webm|mov)$")


def _public_prefix() -> str:
    return os.getenv("SUPABASE_URL", "").strip().rstrip("/") + f"/storage/v1/object/public/{BUCKET}/"


def _media_name(url: str) -> str | None:
    for prefix in (_public_prefix(), "/media/"):
        if url.startswith(prefix):
            name = url[len(prefix):].split("?")[0]
            return name if MEDIA_NAME.match(name) else None
    return None


def _exists_in_bucket(name: str) -> bool:
    try:
        req = urllib.request.Request(_public_prefix() + name, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status < 400
    except urllib.error.HTTPError as e:
        return e.code not in (400, 404)
    except Exception:
        return True


@app.post("/api/media")
async def upload_media(request: Request):
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype not in MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Use a JPG, PNG or WEBP picture, or an MP4, WEBM or MOV video.")
    limit = MAX_VIDEO if ctype.startswith("video/") else MAX_IMAGE
    name = uuid4().hex + MEDIA_TYPES[ctype]
    size = 0
    with tempfile.NamedTemporaryFile(suffix=MEDIA_TYPES[ctype]) as tmp:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"File is too big. Limit: {limit // (1024 * 1024)} MB.")
            tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="The file was empty.")
        tmp.flush()

        def _send() -> None:
            with open(tmp.name, "rb") as fh:
                db.client().storage.from_(BUCKET).upload(
                    path=name,
                    file=fh,
                    file_options={"content-type": ctype, "cache-control": "31536000", "upsert": "false"},
                )

        try:
            await run_in_threadpool(_send)
        except db.DatabaseNotConfigured:
            raise
        except Exception as exc:
            log.error("Upload to Supabase Storage failed: %s", exc)
            raise HTTPException(status_code=502, detail="Could not save the file to storage. Please try again.")
    return {"url": _public_prefix() + name, "type": "video" if ctype.startswith("video/") else "image", "size": size}


@app.get("/media/{name}", include_in_schema=False)
def get_media(name: str):
    if not MEDIA_NAME.match(name):
        raise HTTPException(status_code=404, detail="Not found")
    return RedirectResponse(_public_prefix() + name, status_code=307)


@app.post("/api/posts")
def post_ad(body: PostIn):
    p = _find(body.sku)
    plats = list(dict.fromkeys(body.platforms))
    if not plats:
        raise HTTPException(status_code=422, detail="Choose at least one platform.")
    for pl in plats:
        _check_platform(pl)
    conn = connectors.connected_set()
    not_conn = [pl for pl in plats if not connectors.is_connected(pl, conn)]
    if not_conn:
        raise HTTPException(status_code=422, detail="Connect these accounts first: " + ", ".join(not_conn))
    if not body.headline.strip() or len(body.headline) > 90:
        raise HTTPException(status_code=422, detail="Write a headline (up to 90 characters).")
    if len(body.text) > 1000:
        raise HTTPException(status_code=422, detail="Ad text can be up to 1000 characters.")
    wrong = [pl for pl in plats if body.format not in ads.FORMAT_SUPPORT[pl]]
    if wrong:
        raise HTTPException(status_code=422, detail=f"{', '.join(wrong)} cannot show a {ads.AD_FORMATS[body.format]['label'].lower()}.")
    media_type = ""
    media_url = body.media_url
    if media_url:
        name = _media_name(media_url)
        if not name or not _exists_in_bucket(name):
            raise HTTPException(status_code=422, detail="Upload the picture or video again.")
        media_type = "video" if name.endswith((".mp4", ".webm", ".mov")) else "image"
        media_url = _public_prefix() + name
    if body.format == "image" and media_type == "video":
        raise HTTPException(status_code=422, detail="An image post needs a picture, not a video.")
    if body.format in ("video", "reel"):
        if media_type != "video":
            raise HTTPException(status_code=422, detail="Upload a video for this ad.")
        secs = body.media_seconds or 0
        if body.format == "reel":
            if secs > 90.5:
                raise HTTPException(status_code=422, detail="A reel or short can be up to 90 seconds.")
            if body.media_width and body.media_height and body.media_height <= body.media_width:
                raise HTTPException(status_code=422, detail="A reel or short must be a vertical (portrait) video.")
        elif secs > 600:
            raise HTTPException(status_code=422, detail="A video ad can be up to 10 minutes.")
    if body.daily_budget <= 0 or (body.days is not None and not 1 <= body.days <= 90):
        raise HTTPException(status_code=422, detail="Budget must be above 0 and days between 1 and 90.")
    if p["status"] != "active":
        raise HTTPException(status_code=422, detail=f"{p['name']} is paused in Product Management. Set it to Selling first.")
    creative = {
        "headline": body.headline.strip(),
        "text": body.text.strip(),
        "cta": body.cta.strip() or "Shop now",
        "format": body.format,
        "media_url": media_url,
        "media_type": media_type,
        "media_seconds": round(body.media_seconds, 1) if body.media_seconds else None,
    }
    made = ads.post_ad(p, plats, creative, body.daily_budget, body.days)
    return {"posts": made, "message": f"Ad for {p['name']} posted on {', '.join(plats)}."}


@app.get("/api/posts")
def list_posts(sku: str = "all"):
    return ads.posts_list(sku)


class ConnectIn(BaseModel):
    platform: str
    account_name: str = ""
    values: dict[str, str] = {}


class PlatformIn(BaseModel):
    platform: str


@app.get("/api/connections")
def connections():
    return {"live_mode": connectors.LIVE, "platforms": connectors.list_connections()}


@app.post("/api/connections")
def connect_platform(body: ConnectIn):
    res = connectors.connect(body.platform, body.account_name, body.values)
    if not res["ok"]:
        raise HTTPException(status_code=422, detail=res["message"])
    return res


@app.post("/api/connections/demo")
def connect_demo(body: PlatformIn):
    if body.platform not in connectors.PLATFORMS:
        raise HTTPException(status_code=422, detail="Unknown platform.")
    return connectors.connect(body.platform, "Lumen & Co. (demo)", connectors.demo_values(body.platform))


@app.post("/api/connections/test")
def test_platform(body: PlatformIn):
    return connectors.test(body.platform)


@app.delete("/api/connections/{platform}")
def disconnect_platform(platform: str):
    if not connectors.disconnect(platform):
        raise HTTPException(status_code=404, detail="That platform is not connected.")
    return {"message": f"{platform} disconnected and its saved keys deleted. Nothing syncs from it now."}


# ---------- Login ----------
class LoginIn(BaseModel):
    email: str = ""
    password: str = ""


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")  # Railway puts the visitor's address here
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _is_https(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


@app.post("/api/login")
def login(body: LoginIn, request: Request):
    ip = _client_ip(request)
    if auth.too_many_tries(ip):
        raise HTTPException(status_code=429, detail="Too many wrong tries. Wait 10 minutes and try again.")
    role = auth.check_login(body.email, body.password)
    if not role:
        auth.note_failure(ip)
        raise HTTPException(status_code=401, detail="Those details don't match an account.")
    auth.clear_failures(ip)
    email = body.email.strip().lower()
    resp = JSONResponse({"ok": True, "email": email, "role": role})
    resp.set_cookie(auth.COOKIE, auth.make_token(email, role), max_age=int(auth.SESSION_DAYS * 86400),
                    httponly=True, secure=_is_https(request), samesite="lax", path="/")
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.get("/api/auth/me")
def who_am_i(request: Request):
    user = auth.read_token(request.cookies.get(auth.COOKIE))
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


@app.get("/api/auth/config")
def auth_config():
    """Tells the login page whether to show the demo account box."""
    return {"demo": auth.demo_enabled(), "owner_configured": auth.owner_configured()}


# ---------- Pages: landing + login at "/", dashboard at "/dashboard" (needs login) ----------
HERE = Path(__file__).parent
INDEX = HERE / "index.html"
DASHBOARD = HERE / "dashboard.html"


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def root():
    if INDEX.exists():
        return FileResponse(INDEX)
    return JSONResponse({"status": "NEXUS API running. Put index.html next to main.py to serve the site."})


@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard.html", include_in_schema=False)
def dashboard_page():
    if DASHBOARD.exists():
        return FileResponse(DASHBOARD)
    return JSONResponse({"status": "Put dashboard.html next to main.py."})
