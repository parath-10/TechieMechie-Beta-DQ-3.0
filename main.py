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
import connectors
import db
from ai_engine import (
    analyze_omnichannel_feedback,
    answer_chat_query,
    get_ai_status,
    get_monitoring_alerts,
    get_section_tip,
    get_suggestions,
    get_timeline_events,
    predict_future_performance,
    run_diagnosis,
)

try:  # error type raised by the Supabase client for database problems
    from postgrest.exceptions import APIError as PostgrestAPIError
except Exception:  # pragma: no cover
    PostgrestAPIError = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nexus.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """On start-up: create the sample data once (first run only). The app still starts if this fails."""
    try:
        if await run_in_threadpool(ads.ensure_seeded):
            log.info("Sample data created in Supabase.")
    except Exception as exc:
        log.error("Could not prepare the database on start-up: %s", exc)
    yield


app = FastAPI(title="NEXUS D2C Command Center", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Database errors become a clear message (same {"detail": ...} shape as other errors) ----------
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


# ---------- Fixed sample campaign figures used by the AI features (read-only, not state) ----------
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
    """Add the worked-out stats the dashboard shows for each product.
    daily_sales is calculated from synced platform orders, never typed in."""
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
    """Live data given to Gemini, so answers match what the dashboard shows."""
    products = ads.get_products()
    return {
        **CAMPAIGN_DATA,
        "products": _views(products),
        "advertising": ads.summary_for_ai(products),
        "posted_ads": [{k: v for k, v in x.items() if k not in ("media_url",)} for x in ads.posts_list()[:20]],
        "connected_platforms": connectors.connected_names(),
        "monthly_all_products": ads.monthly(products)["months"],
    }


# ---------- Request models ----------
class ExecuteActionRequest(BaseModel):
    action_id: str | None = None
    action: str | None = None
    target: str | None = None
    change: str | None = None
    type: str | None = None
    campaign_id: str | None = None


class ChatRequest(BaseModel):
    question: str = ""
    history: list[dict[str, Any]] = []  # earlier messages, so follow-up questions make sense


def _scale_pct(v: Any, default: float) -> float:
    """Gemini may return 0-1 or 0-100; always give the UI a 0-1 value."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v / 100.0 if v > 1 else v


# ---------- Endpoints ----------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/ai-status")
def ai_status(force: bool = False):
    """Tells the dashboard whether Gemini works and, if not, the exact reason."""
    return get_ai_status(force)


@app.get("/api/diagnose")
def diagnose_campaign():
    d = run_diagnosis(context())
    action = d.get("recommended_action") or {}
    return {
        "campaign": CAMPAIGN_DATA["campaign"],
        "root_cause": d.get("root_cause", "No clear cause found."),
        "confidence": _scale_pct(d.get("confidence_score"), 0.85),
        "opportunity_score": round(float(d.get("opportunity_score") or 80)),
        "recommended_action": action.get("description") or action.get("title") or "Review the campaign.",
        "action_title": action.get("title", ""),
        "source": d.get("source", "fallback"),
    }


@app.get("/api/forecast")
def get_forecast():
    return predict_future_performance(context())


@app.get("/api/alerts")
def get_alerts():
    return get_monitoring_alerts(context())


@app.get("/api/timeline")
def get_timeline():
    return get_timeline_events(context())


@app.get("/api/actions")
def get_actions():
    return ads.get_actions()


@app.post("/api/execute-action")
def execute_action(payload: ExecuteActionRequest):
    label = payload.target or payload.action or payload.action_id or "action"
    done = ads.mark_action_done(payload.action_id, payload.target)
    if done:
        label = done["target"]
    log.info("Executed: %s", label)
    return {"status": "executed", "action_id": payload.action_id, "message": f"“{label}” was applied successfully."}


@app.post("/api/chat")
def assistant_chat(payload: ChatRequest):
    return {"answer": answer_chat_query(payload.question, context(), payload.history)}


@app.post("/api/analyze-feedback")
def analyze_feedback(payload: Any = Body(default=None)):
    """Accepts a list of reviews, {"feedback": [...]}, or {"text": "one review per line"}."""
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("feedback") or [l for l in str(payload.get("text", "")).splitlines()]
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


# ---------- Product and inventory management ----------
class ProductIn(BaseModel):
    name: str
    price: float            # price at selling
    cost: float = 0         # cost to make one unit
    stock: int = 0          # opening stock
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
def list_products():
    return _views(ads.get_products())


@app.post("/api/products")
def add_product(body: ProductIn):
    if (not body.name.strip() or body.price <= 0 or body.cost < 0 or body.stock < 0
            ):
        raise HTTPException(status_code=422, detail="Name and a selling price above 0 are needed; other numbers cannot be negative.")
    p = ads.insert_product({
        "sku": "NX-" + uuid4().hex[:4].upper(), "units_sold": 0, "name": body.name.strip(),
        "category": body.category.strip() or "General", "price": body.price, "cost": body.cost, "stock": body.stock,
        "status": "active", "ad_spend": 0, "added_sim": ads.sim_days(), "website_rate": 1.0})
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
    if not ads.delete_product(sku):  # its ads, posts and month totals are deleted with it
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


# ---------- Orders synced from the connected platforms ----------
@app.post("/api/orders/sync")
def sync_orders():
    """Pull new orders from every connected platform; stock and stats update automatically.
    The dashboard calls this every 30 seconds."""
    return ads.sync_orders()


@app.get("/api/orders")
def orders():
    """Total orders per product per platform, the latest orders, and which channels are connected."""
    return ads.orders_summary(ads.get_products())


# ---------- Advertisement management ----------
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
    return {"ad_platforms": ads.AD_PLATFORMS, "sales_channels": ads.SALES_CHANNELS,
            "formats": ads.AD_FORMATS, "format_support": ads.FORMAT_SUPPORT}


@app.get("/api/ads")
def list_ads():
    """Every product with its ads on each platform (views, clicks, orders, spend, reviews)."""
    products = ads.get_products()
    views = {v["sku"]: v for v in _views(products)}
    ad_views = ads.ads_for_products(products)
    return [{"sku": p["sku"], "name": p["name"], "status": p["status"], "stock": p["stock"],
             "days_of_cover": views[p["sku"]]["days_of_cover"], "ads": ad_views[p["sku"]]} for p in products]


@app.post("/api/ads")
def start_ad(body: AdStartIn):
    _check_platform(body.platform)
    if body.daily_budget <= 0:
        raise HTTPException(status_code=422, detail="Daily budget must be above 0.")
    p = _find(body.sku)
    if not connectors.is_connected(body.platform):
        raise HTTPException(status_code=422, detail=f"Connect your {body.platform} account first (Connected Platforms page).")
    return {**ads.start_ad(p, body.platform, body.daily_budget),
            "message": f"{p['name']} is now advertised on {body.platform}."}


@app.patch("/api/ads")
def change_ad(body: AdChangeIn):
    """Pause, resume, stop, or change the daily budget of one ad."""
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
    """DEMO: add one day of results to every running ad (until real accounts are connected)."""
    return ads.simulate_day()


@app.get("/api/monthly")
def monthly_log(sku: str = "all", platform: str = "all", months: int = 6):
    """Month-wise sales, views and reviews, filtered by product and platform."""
    if sku != "all":
        _find(sku)
    if platform != "all" and platform not in ads.SALES_CHANNELS:
        raise HTTPException(status_code=422, detail="Unknown platform.")
    return ads.monthly(ads.get_products(), sku, platform, max(1, min(months, 12)))


# ---------- Posting ads to chosen platforms ----------
class PostIn(BaseModel):
    sku: str
    platforms: list[str]
    format: Literal["image", "video", "reel"] = "image"
    headline: str
    text: str = ""
    cta: str = "Shop now"
    media_url: str = ""          # from /api/media (optional for image posts)
    media_seconds: float | None = None
    media_width: int | None = None
    media_height: int | None = None
    daily_budget: float
    days: int | None = None      # how long it runs; empty = until stopped


# ---------- Uploading ad pictures and videos (Supabase Storage) ----------
BUCKET = "ad_creatives"
MEDIA_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov"}
MAX_IMAGE = 8 * 1024 * 1024
# Supabase Free plan allows files up to 50 MB; on Pro set MAX_VIDEO_MB=100 (and raise the bucket limit).
MAX_VIDEO = int(os.getenv("MAX_VIDEO_MB", "50")) * 1024 * 1024
MEDIA_NAME = re.compile(r"^[0-9a-f]{32}\.(jpg|png|webp|mp4|webm|mov)$")


def _public_prefix() -> str:
    return os.getenv("SUPABASE_URL", "").strip().rstrip("/") + f"/storage/v1/object/public/{BUCKET}/"


def _media_name(url: str) -> str | None:
    """File name from a public bucket URL (or an old /media/ address); None if it is not one of ours."""
    for prefix in (_public_prefix(), "/media/"):
        if url.startswith(prefix):
            name = url[len(prefix):].split("?")[0]
            return name if MEDIA_NAME.match(name) else None
    return None


def _exists_in_bucket(name: str) -> bool:
    """Quick check that an uploaded file is really there. Allows posting if the check itself cannot run."""
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
    """Upload one picture or video as the raw request body (Content-Type = the file's type).
    The bytes are streamed through a temporary spool (never kept on the server) into the
    Supabase `ad_creatives` bucket. Returns the file's public URL."""
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
            with open(tmp.name, "rb") as fh:  # read from the spool, never held fully in memory here
                db.client().storage.from_(BUCKET).upload(
                    path=name, file=fh,
                    file_options={"content-type": ctype, "cache-control": "31536000", "upsert": "false"})

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
    """Old /media/ addresses keep working: they now redirect to the file in Supabase Storage."""
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
        media_url = _public_prefix() + name  # always store the full public address
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
    creative = {"headline": body.headline.strip(), "text": body.text.strip(), "cta": body.cta.strip() or "Shop now",
                "format": body.format, "media_url": media_url, "media_type": media_type,
                "media_seconds": round(body.media_seconds, 1) if body.media_seconds else None}
    made = ads.post_ad(p, plats, creative, body.daily_budget, body.days)
    return {"posts": made, "message": f"Ad for {p['name']} posted on {', '.join(plats)}."}


@app.get("/api/posts")
def list_posts(sku: str = "all"):
    return ads.posts_list(sku)


# ---------- Connected platform accounts ----------
class ConnectIn(BaseModel):
    platform: str
    account_name: str = ""
    values: dict[str, str] = {}


class PlatformIn(BaseModel):
    platform: str


@app.get("/api/connections")
def connections():
    """Every platform with its connection state. Secrets are never included, only the last 4 characters."""
    return {"live_mode": connectors.LIVE, "platforms": connectors.list_connections()}


@app.post("/api/connections")
def connect_platform(body: ConnectIn):
    res = connectors.connect(body.platform, body.account_name, body.values)
    if not res["ok"]:
        raise HTTPException(status_code=422, detail=res["message"])
    return res


@app.post("/api/connections/demo")
def connect_demo(body: PlatformIn):
    """Connect a platform with a ready-made demo account (handy for testing and presentations)."""
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


@app.get("/api/suggestions")
def suggestions(section: str = "actions"):
    """AI suggestions for Things To Do (actions), Problems to Watch (causal) and Profit Check (poas)."""
    return get_suggestions(section, context())


@app.get("/api/section-tip")
def section_tip(section: str = "overview"):
    return get_section_tip(section, context())


# ---------- Serve the dashboard from the same address (avoids CORS problems) ----------
INDEX = Path(__file__).parent / "index.html"


@app.get("/", include_in_schema=False)
def root():
    if INDEX.exists():
        return FileResponse(INDEX)
    return JSONResponse({"status": "NEXUS API running. Put index.html next to main.py to serve the dashboard."})
