"""NEXUS AI engine: Gemini 2.5 Flash with strict JSON schemas and safe fallbacks.

Every public function returns usable data even if the API key is missing,
the SDK is not installed, the model is throttled, or the reply is invalid.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:  # SDK missing: every function falls back to sample data
    genai = None
    types = None

# Load .env from the same folder as this file, so it works no matter where uvicorn is started from.
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()
log = logging.getLogger("nexus.ai")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
# Tried in order only if the main model name is not found or has been retired.
BACKUP_MODELS = [m for m in ("gemini-flash-latest", "gemini-2.0-flash") if m != PRIMARY_MODEL]
CACHE_TTL = int(os.getenv("AI_CACHE_SECONDS", "60"))
_cache: dict[str, tuple[float, dict]] = {}
_working_model: str | None = None
_last_error: str = ""

SYSTEM = (
    "You are NEXUS, an advertising analyst for a direct-to-consumer brand. "
    "Use simple, plain English a shop owner understands. Currency is Indian rupees. "
    "Use only the data given; never invent numbers that cannot be worked out from it."
)


# ---------- Strict response schemas ----------
class RecommendedAction(BaseModel):
    action_type: str
    title: str
    description: str
    target: str


class DiagnosisResult(BaseModel):
    root_cause: str
    confidence_score: float = Field(description="0 to 100")
    opportunity_score: float = Field(description="0 to 100")
    recommended_action: RecommendedAction


class ForecastResult(BaseModel):
    forecast_horizon: str
    predicted_poas: float
    predicted_revenue_growth_pct: float
    fatigue_risk_level: str = Field(description="low, medium or high")
    stockout_risk_days: int
    integration_suggestions: list[str] = Field(description="exactly 3 items")
    summary: str = Field(description="2 plain-English sentences on what will happen next")
    inventory_advice: str = Field(description="one sentence on what to restock and when")


class SectionTip(BaseModel):
    title: str = Field(description="at most 6 words")
    tip: str = Field(description="at most 2 short sentences in plain English")


class AlertItem(BaseModel):
    id: str
    severity: str = Field(description="'critical', 'warning', or 'success'")
    title: str
    message: str
    timestamp: str


class AlertList(BaseModel):
    alerts: list[AlertItem]


class TimelineEvent(BaseModel):
    time: str
    title: str
    desc: str
    type: str = Field(description="'info', 'warning', or 'success'")


class TimelineList(BaseModel):
    events: list[TimelineEvent]


class Sentiment(BaseModel):
    positive: float
    neutral: float
    negative: float


class FeedbackReport(BaseModel):
    sentiment: Sentiment = Field(description="three shares that add up to 1")
    summary: str
    themes: list[str]
    creative_feedback: list[str]


# ---------- Helpers ----------
def _api_key() -> str:
    """Read the key and remove stray quotes or spaces that often sneak into .env files."""
    raw = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return raw.strip().strip('"').strip("'").strip()


def _get_client():
    if genai is None:
        raise RuntimeError("SDK_MISSING")
    api_key = _api_key()
    if not api_key:
        raise ValueError("KEY_MISSING")
    return genai.Client(api_key=api_key)


def _explain(exc: Exception) -> str:
    """Turn an SDK or network error into a plain-English reason the dashboard can show."""
    text = str(exc)
    low = text.lower()
    if genai is None or "SDK_MISSING" in text:
        return "The google-genai package is not installed. Run: pip install google-genai"
    if "KEY_MISSING" in text:
        return ("No GEMINI_API_KEY found. Create a file named .env (not .env.txt) next to ai_engine.py "
                "containing GEMINI_API_KEY=your_key, then restart the server.")
    if "api_key_invalid" in low or "api key not valid" in low or "invalid api key" in low:
        return "Gemini rejected the API key. Copy a fresh key from aistudio.google.com/apikey into .env."
    if "429" in text or "resource_exhausted" in low or "quota" in low:
        return "Gemini free-tier limit reached (too many requests). Wait a minute or check your quota in AI Studio."
    if "permission" in low or "403" in text:
        return "This API key is not allowed to use Gemini. Enable the Generative Language API for its project."
    if "404" in text or "not found" in low:
        return f"Model '{PRIMARY_MODEL}' was not found. Set GEMINI_MODEL to a current model name in .env."
    if any(w in low for w in ("connect", "timed out", "timeout", "getaddrinfo", "network", "ssl")):
        return "The server could not reach Google (no internet, firewall or proxy)."
    return "Gemini error: " + text[:200]


def _call_model(**kwargs):
    """Call Gemini; if the model name is not found, try the backup names once."""
    global _working_model, _last_error
    client = _get_client()
    models = [_working_model] if _working_model else [PRIMARY_MODEL, *BACKUP_MODELS]
    last_exc: Exception | None = None
    for model in models:
        try:
            response = client.models.generate_content(model=model, **kwargs)
            _working_model, _last_error = model, ""
            return response
        except Exception as exc:
            last_exc = exc
            low = str(exc).lower()
            if "404" not in low and "not found" not in low:
                break  # a bad key, quota or network problem will not be fixed by another model name
    raise last_exc  # type: ignore[misc]


def _note_error(name: str, exc: Exception) -> None:
    global _last_error
    _last_error = _explain(exc)
    log.warning("%s fell back to sample data: %s", name, _last_error)


_status_cache: dict[str, Any] = {"at": 0.0, "data": None}


def get_ai_status(force: bool = False) -> dict[str, Any]:
    """Make a tiny test call so the dashboard can show exactly why live AI is on or off."""
    global _last_error
    if not force and _status_cache["data"] and time.time() - _status_cache["at"] < 120:
        return _status_cache["data"]
    try:
        _call_model(contents="Reply with the single word OK.")
        data = {"live": True, "model": _working_model, "reason": f"Connected to {_working_model}."}
    except Exception as exc:
        _last_error = _explain(exc)
        data = {"live": False, "model": PRIMARY_MODEL, "reason": _last_error}
    _status_cache.update(at=time.time(), data=data)
    return data


def _stamp(minutes_ago: int = 0) -> str:
    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%d %b, %I:%M %p")


def _generate(name: str, prompt: str, schema: type[BaseModel], temperature: float) -> dict | None:
    """Call Gemini, validate against the schema, cache for a short time. None on any failure."""
    key = name + hashlib.sha1(prompt.encode()).hexdigest()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        response = _call_model(
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        data = schema.model_validate_json(response.text).model_dump()
        _cache[key] = (time.time(), data)
        return data
    except Exception as exc:  # throttling, bad key, network, invalid JSON, anything
        _note_error(name, exc)
        return None


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


# ---------- Fallback data ----------
def _fallback_diagnosis() -> dict:
    return {
        "root_cause": "The ad “Summer Drop – UGC v3” has been shown too many times to the same people, so clicks dropped about 31%. Each sale now costs more, while your best-selling linen shirt has only 4 days of stock left.",
        "confidence_score": 91.0,
        "opportunity_score": 84.0,
        "recommended_action": {
            "action_type": "pause_and_reallocate",
            "title": "Stop the tired ad and move its budget",
            "description": "Stop “Summer Drop – UGC v3”, move ₹45,000 a day to the “Founder Story” TikTok video, and limit ads on the linen shirt until new stock arrives.",
            "target": "Meta: Summer Drop – UGC v3",
        },
    }


def _fallback_forecast() -> dict:
    return {
        "forecast_horizon": "Next 14 days",
        "predicted_poas": 1.28,
        "predicted_revenue_growth_pct": 12.5,
        "fatigue_risk_level": "high",
        "stockout_risk_days": 4,
        "summary": "If nothing changes, profit per ₹1 of ads should improve slightly to about 1.28 as weak ads are stopped. The linen shirt will run out in about 4 days and ad tiredness on Meta will keep rising.",
        "inventory_advice": "Reorder about 240 Linen Shirts and 135 Overshirts now so you have 3 weeks of stock.",
        "integration_suggestions": [
            "Connect Meta Conversions API (server-side tracking) to recover sales that iPhone privacy settings hide.",
            "Connect your warehouse stock count so ads pause automatically before a product sells out.",
            "Connect your courier tracking to show real delivery times and cut complaints about late parcels.",
        ],
    }


def _fallback_alerts() -> list[dict]:
    return [
        {"id": "AL-1", "severity": "critical", "title": "Linen shirt almost out of stock", "message": "Only 4 days of stock left at the current sales speed. Ads on this product should be limited today.", "timestamp": _stamp(6)},
        {"id": "AL-2", "severity": "warning", "title": "Ad fatigue on Meta", "message": "“Summer Drop – UGC v3” is seen 4.8 times per person and clicks are down 31% this week.", "timestamp": _stamp(38)},
        {"id": "AL-3", "severity": "success", "title": "TikTok video is doing well", "message": "“Founder Story” earns ₹1.62 profit for every ₹1 spent. It can take more budget.", "timestamp": _stamp(95)},
    ]


def _fallback_timeline() -> list[dict]:
    return [
        {"time": _stamp(180), "title": "Data synced", "desc": "Meta, TikTok, Shopify and Amazon data was updated.", "type": "info"},
        {"time": _stamp(120), "title": "Orders matched", "desc": "99.8% of orders were matched with real sales and product costs.", "type": "success"},
        {"time": _stamp(45), "title": "Tired ad detected", "desc": "Clicks on “Summer Drop – UGC v3” fell sharply. A fix was suggested.", "type": "warning"},
        {"time": _stamp(10), "title": "Low stock guard started", "desc": "Ads on the linen shirt were flagged to be limited until restock.", "type": "info"},
    ]


def _fallback_feedback(reviews: list) -> dict:
    return {
        "sentiment": {"positive": 0.58, "neutral": 0.17, "negative": 0.25},
        "summary": "Customers like the fabric and fit. Most complaints are about the colour looking different from the ad and slow delivery outside big cities.",
        "themes": ["Fabric quality", "Colour in ad vs real", "Slow delivery", "Size runs small"],
        "creative_feedback": [
            "Re-shoot the main photo in daylight so the colour matches the product.",
            "Show fabric and fit close-ups first, since people praise these most.",
            "Mention delivery time on pages for smaller cities.",
        ],
    }


def _rs(n: Any) -> str:
    try:
        return "₹" + f"{round(float(n)):,}"
    except (TypeError, ValueError):
        return "₹0"


def _local_answer(question: str, ctx: dict) -> str:
    """Backup answer when Gemini is off: reads the question and answers from the live data, never refuses."""
    q = (question or "").lower()
    has = lambda *w: any(x in q for x in w)
    prods = ctx.get("products", [])
    m = ctx.get("metrics_last_7_days", {})
    camp = ctx.get("campaign", {})
    plats = ctx.get("other_platforms", {})
    lines: list[str] = []

    def stats(p: dict) -> str:
        cover = p.get("days_of_cover", 999)
        return (f"{p['name']}: sells at {_rs(p['price'])}, costs {_rs(p['cost'])} to make "
                f"({p.get('margin_pct', 0)}% margin), {p.get('units_sold', 0):,} sold, "
                f"{_rs(p.get('total_profit', 0))} profit, {p['stock']:,} in stock"
                + ("" if cover >= 999 else f" (lasts {cover} days)"))

    named = [p for p in prods if p["name"].lower() in q]
    for p in named:
        lines.append(stats(p))

    if prods and not named:
        if has("best sell", "most sold", "top", "popular", "sells most", "selling most", "selling the most", "best product", "best item", "sells the most"):
            p = max(prods, key=lambda x: x.get("units_sold", 0))
            lines.append(f"Best seller: {p['name']} with {p.get('units_sold', 0):,} units sold ({_rs(p.get('revenue', 0))} in sales).")
        if has("profit", "earn", "margin", "money"):
            p = max(prods, key=lambda x: x.get("total_profit", 0))
            low = min(prods, key=lambda x: x.get("margin_pct", 0))
            lines.append(f"Most profitable product: {p['name']} ({_rs(p.get('total_profit', 0))} total profit). "
                         f"Lowest margin: {low['name']} at {low.get('margin_pct', 0)}%.")
        if has("stock", "inventory", "run out", "reorder", "restock", "low"):
            low = sorted(prods, key=lambda x: x.get("days_of_cover", 999))
            urgent = [p for p in low if p.get("days_of_cover", 999) < 7]
            if urgent:
                lines.append("Reorder soon: " + ", ".join(f"{p['name']} ({p['days_of_cover']} days left)" for p in urgent) + ".")
            else:
                lines.append(f"No product runs out within a week. First to run out: {low[0]['name']} ({low[0].get('days_of_cover')} days).")
        if has("price", "cost", "all product", "list", "every product"):
            lines.extend(stats(p) for p in prods)

    for name, d in plats.items():
        if name in q:
            lines.append(f"{name.title()}: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in d.items()) + ".")
    if has("meta campaign", "summer drop", "campaign", "fatigue", "tired"):
        lines.append(f"Meta campaign “{camp.get('name', '')}” (last 7 days): spend {_rs(m.get('spend'))}, "
                     f"{m.get('impressions', 0):,} views, {m.get('clicks', 0):,} clicks (down {abs(m.get('ctr_change_pct', 0))}%), "
                     f"{m.get('orders', 0):,} orders, profit per ₹1 of ads {m.get('poas', 0)}, seen {m.get('frequency', 0)} times per person.")

    ad_rows = ctx.get("advertising", [])
    if ad_rows and has("advert", "ads", "review", "rating", "views", "platform", "where"):
        rows = [a for a in ad_rows if not named or a["product"] in {p["name"] for p in named}]
        for a in sorted(rows, key=lambda x: -x["views"])[:6]:
            lines.append(f"{a['product']} on {a['platform']} ({a['status']}): {a['views']:,} views, {a['orders']:,} orders, "
                         f"profit after ads {_rs(a['profit_after_ads'])}"
                         + (f", rated {a['avg_rating']} from {a['review_count']} reviews" if a.get("avg_rating") else ""))

    if not lines:  # greeting or general question: give a short business overview instead of refusing
        if prods:
            best = max(prods, key=lambda x: x.get("units_sold", 0))
            rich = max(prods, key=lambda x: x.get("total_profit", 0))
            first = min(prods, key=lambda x: x.get("days_of_cover", 999))
            lines.append(f"Quick overview: {len(prods)} products. Best seller is {best['name']}, most profitable is {rich['name']}, "
                         f"and {first['name']} runs out first ({first.get('days_of_cover')} days).")
        lines.append(f"Ads: profit per ₹1 on Meta is {m.get('poas', 0)}; TikTok is {plats.get('tiktok', {}).get('poas', '-')}.")
        lines.append("You can ask about a product by name, best sellers, profit, stock, prices, or any platform.")

    note = f"\n\n(Live AI is off, so this is a quick summary from your numbers. Reason: {_last_error or 'unknown, click the AI badge at the top'})"
    return "\n".join(lines) + note


# ---------- Public functions ----------
def run_diagnosis(campaign_data: dict) -> dict[str, Any]:
    prompt = (
        "Find the main reason this campaign is underperforming and the best single fix. "
        "Scores are 0 to 100.\n" + _json(campaign_data)
    )
    data = _generate("diagnosis", prompt, DiagnosisResult, 0.2)
    if data:
        data["source"] = "gemini"
        return data
    fb = _fallback_diagnosis()
    fb["source"] = "fallback"
    return fb


def predict_future_performance(campaign_data: dict) -> dict[str, Any]:
    prompt = (
        "Forecast the next 14 days of performance (POAS = profit per rupee of ad spend) and suggest "
        "exactly 3 practical tool integrations the brand should add. Use the product stock and daily "
        "sales to predict stock-outs, and give one short restocking tip.\n" + _json(campaign_data)
    )
    data = _generate("forecast", prompt, ForecastResult, 0.4)
    if data and len(data.get("integration_suggestions", [])) >= 1:
        data["integration_suggestions"] = data["integration_suggestions"][:3]
        data["source"] = "gemini"
        return data
    fb = _fallback_forecast()
    fb["source"] = "fallback"
    return fb


def get_monitoring_alerts(campaign_data: dict | None = None) -> list[dict[str, Any]]:
    prompt = (
        "Write exactly 3 realistic 24/7 monitoring alerts (for example low stock, ad fatigue, a win) "
        "from this live data. Use ids AL-1, AL-2, AL-3.\n" + _json(campaign_data or {})
    )
    data = _generate("alerts", prompt, AlertList, 0.5)
    alerts = (data or {}).get("alerts", [])[:3]
    if not alerts:
        return _fallback_alerts()
    for a in alerts:
        if a["severity"] not in ("critical", "warning", "success"):
            a["severity"] = "warning"
    return alerts


def get_timeline_events(campaign_data: dict | None = None) -> list[dict[str, Any]]:
    prompt = (
        "Write exactly 4 timeline events, oldest first, showing recent system actions and data syncs "
        "based on this data.\n" + _json(campaign_data or {})
    )
    data = _generate("timeline", prompt, TimelineList, 0.5)
    events = (data or {}).get("events", [])[:4]
    if not events:
        return _fallback_timeline()
    for e in events:
        if e["type"] not in ("info", "warning", "success"):
            e["type"] = "info"
    return events


def analyze_omnichannel_feedback(feedback_list: list) -> dict[str, Any]:
    if not feedback_list:
        fb = _fallback_feedback([])
        fb["summary"] = "No reviews were given, so this is an example report."
        fb["source"] = "fallback"
        return fb
    prompt = (
        "Read these customer reviews. Give the share of positive, neutral and negative (adding to 1), "
        "a short summary, the main topics, and advice for improving the ads.\n" + _json(feedback_list)
    )
    data = _generate("feedback", prompt, FeedbackReport, 0.2)
    if data:
        data["source"] = "gemini"
        return data
    fb = _fallback_feedback(feedback_list)
    fb["source"] = "fallback"
    return fb


def answer_chat_query(question: str, context_data: dict, history: list | None = None) -> str:
    question = (question or "").strip()
    if not question:
        return "Please type a question."
    past = ""
    for h in (history or [])[-6:]:
        who = "Owner" if h.get("me") else "Assistant"
        past += f"{who}: {str(h.get('t', ''))[:500]}\n"
    prompt = (
        "You are the assistant inside this brand's dashboard. Answer the owner's question in 2 to 5 short, "
        "friendly sentences using the data below (products with cost, price, units sold, profit, stock and "
        "days of cover; ad campaigns; platforms). Work out totals, rankings and comparisons when asked. "
        "For greetings, reply briefly and suggest 2 example questions. If the question has nothing to do with "
        "this business (for example homework or general trivia), say politely in one sentence that you can "
        "only help with this dashboard's data, and suggest a relevant question."
        f"\n\nData:\n{_json(context_data)}"
        + (f"\n\nEarlier in this chat:\n{past}" if past else "")
        + f"\n\nQuestion: {question}"
    )
    try:
        response = _call_model(
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3),
        )
        text = (response.text or "").strip()
        return text or _local_answer(question, context_data)
    except Exception as exc:
        _note_error("chat", exc)
        return _local_answer(question, context_data)


# ---------- AI suggestions (Things To Do, Problems to Watch, Profit Check) ----------
class Suggestion(BaseModel):
    title: str = Field(description="at most 8 words")
    why: str = Field(description="one plain sentence with the real numbers")
    action: str = Field(description="one plain sentence saying exactly what to do")
    priority: str = Field(description="'high', 'medium' or 'low'")
    kind: str = Field(description="'restock', 'pause_ad', 'raise_budget', 'lower_budget', 'fix_reviews' or 'other'")
    sku: str = Field(description="product sku this is about, or empty")
    platform: str = Field(description="platform name this is about, or empty")


class SuggestionList(BaseModel):
    suggestions: list[Suggestion]


SUGGEST_FOCUS = {
    "causal": "problems that are hurting the business right now: stock running out, ads on products that cannot sell, ads losing money, low ratings",
    "poas": "profit from ads: which product-platform ads earn the most and least profit per rupee of ad spend, and where to move budget",
    "actions": "the most useful things the owner should do today, across stock, ads and reviews",
}
KINDS = {"restock", "pause_ad", "raise_budget", "lower_budget", "fix_reviews", "other"}


def _profit_per_rupee(a: dict) -> float | None:
    if not a.get("spend"):
        return None
    return round(a["orders"] * (a["price"] - a["cost"]) / a["spend"], 2)


def _rule_suggestions(section: str, ctx: dict) -> list[dict]:
    """Suggestions worked out from the live numbers (used when Gemini is off)."""
    if section == "actions":  # a mix: the biggest problems plus the best profit moves
        problems = [x for x in _rule_suggestions("causal", ctx) if x["title"] != "Nothing urgent"]
        profit = [x for x in _rule_suggestions("poas", ctx) if x["kind"] != "other"]
        mix = problems[:3] + profit[:2] + problems[3:]
        return mix[:5] or _rule_suggestions("causal", ctx)
    prods = {p["sku"]: p for p in ctx.get("products", [])}
    ads_ = [a for a in ctx.get("advertising", []) if a.get("status") == "active"]
    out: list[dict] = []
    add = lambda **k: out.append({"sku": "", "platform": "", **k})
    if section in ("causal", "actions"):
        for p in prods.values():
            if p["status"] != "active":
                continue
            running = [a for a in ads_ if a["sku"] == p["sku"]]
            if p["stock"] <= 0:
                add(title=f"Restock {p['name']} now", priority="high", kind="restock", sku=p["sku"],
                    why=f"It is out of stock, so orders have stopped" + (f" while {len(running)} ads still cost money." if running else "."),
                    action=f"Add stock in Inventory" + (" and pause its ads until it arrives." if running else "."))
            elif p["days_of_cover"] < 7:
                add(title=f"Reorder {p['name']}", priority="high" if p["days_of_cover"] < 4 else "medium", kind="restock", sku=p["sku"],
                    why=f"Only {p['stock']} left, about {p['days_of_cover']} days at {p['daily_sales']} a day.",
                    action=f"Reorder about {max(0, round(p['daily_sales'] * 21 - p['stock']))} units to cover 3 weeks.")
        for a in ads_:
            p = prods.get(a["sku"])
            if p and (p["status"] != "active" or p["stock"] <= 0):
                add(title=f"Pause {a['product']} ad on {a['platform']}", priority="high", kind="pause_ad", sku=a["sku"], platform=a["platform"],
                    why=f"The ad spends ₹{a['daily_budget']:,.0f} a day but the product cannot be sold right now.",
                    action="Pause this ad until the product is back on sale.")
            elif a.get("avg_rating") and a["avg_rating"] < 3.8 and a.get("review_count", 0) >= 5:
                add(title=f"Fix reviews for {a['product']} on {a['platform']}", priority="medium", kind="fix_reviews", sku=a["sku"], platform=a["platform"],
                    why=f"It is rated {a['avg_rating']} stars from {a['review_count']} reviews, which puts buyers off.",
                    action="Read the reviews, fix the main complaint, and update the ad or listing.")
            if not a.get("connected"):
                add(title=f"Connect {a['platform']}", priority="low", kind="other", platform=a["platform"],
                    why=f"{a['product']} has an ad on {a['platform']} but the account is not connected, so nothing syncs.",
                    action="Connect the account on the Connected Platforms page.")
    if section in ("poas", "actions"):
        scored = [(r, a) for a in ads_ if a.get("connected") and (r := _profit_per_rupee(a)) is not None]
        scored.sort(key=lambda x: x[0])
        if scored:
            worst_r, worst = scored[0]
            if worst_r < 1:
                add(title=f"Cut the {worst['product']} ad on {worst['platform']}", priority="high", kind="pause_ad" if worst_r < 0.6 else "lower_budget",
                    sku=worst["sku"], platform=worst["platform"],
                    why=f"It makes only ₹{worst_r} of profit for every ₹1 of ads, so it loses money.",
                    action="Pause it." if worst_r < 0.6 else "Lower its daily budget by a quarter and try a new picture or video.")
            best_r, best = scored[-1]
            bp = prods.get(best["sku"])
            if best_r > 1.5 and bp and bp["days_of_cover"] >= 14:
                add(title=f"Spend more on {best['product']} on {best['platform']}", priority="medium", kind="raise_budget",
                    sku=best["sku"], platform=best["platform"],
                    why=f"It earns ₹{best_r} profit for every ₹1 of ads and has {bp['days_of_cover']} days of stock.",
                    action=f"Raise its daily budget from ₹{best['daily_budget']:,.0f} by about a quarter.")
            elif best_r > 1.5 and bp:
                add(title=f"Restock before boosting {best['product']}", priority="medium", kind="restock", sku=best["sku"],
                    why=f"Its {best['platform']} ad earns ₹{best_r} per ₹1, but only {bp['days_of_cover']} days of stock are left.",
                    action="Reorder first, then raise the ad budget.")
        elif section == "poas":
            add(title="No ad results yet", priority="low", kind="other",
                why="There are no running ads on connected accounts with spend yet.",
                action="Connect your accounts and post an ad to start measuring profit.")
    if not out:
        add(title="Nothing urgent", priority="low", kind="other", why="Stock, ads and ratings all look healthy right now.",
            action="Check back later or post a new ad to grow sales.")
    order = {"high": 0, "medium": 1, "low": 2}
    seen, uniq = set(), []
    for x in sorted(out, key=lambda x: order.get(x["priority"], 2)):
        k = (x["kind"], x["sku"], x["platform"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq[:5]


def get_suggestions(section: str, ctx: dict) -> dict[str, Any]:
    section = section if section in SUGGEST_FOCUS else "actions"
    slim = {"products": ctx.get("products", []), "advertising": ctx.get("advertising", []),
            "connected_platforms": ctx.get("connected_platforms", [])}
    prompt = (
        f"Give 3 to 5 specific suggestions about {SUGGEST_FOCUS[section]}. Use the real numbers. "
        "Profit per rupee of ads = orders x (price - cost) / spend; below 1 loses money. "
        "Only suggest pause_ad, raise_budget or lower_budget for ads that exist with that sku and platform. "
        "Use empty strings for sku or platform when not needed.\n" + _json(slim)
    )
    data = _generate("suggest-" + section, prompt, SuggestionList, 0.3)
    items = (data or {}).get("suggestions") or []
    if items:
        valid_sku = {p["sku"] for p in slim["products"]}
        valid_ad = {(a["sku"], a["platform"]) for a in slim["advertising"]}
        for it in items:
            it["priority"] = it.get("priority") if it.get("priority") in ("high", "medium", "low") else "medium"
            if it.get("kind") not in KINDS or it.get("sku") not in valid_sku:
                it["kind"] = "other" if it.get("sku") not in valid_sku else it.get("kind", "other")
            if it["kind"] in ("pause_ad", "raise_budget", "lower_budget") and (it.get("sku"), it.get("platform")) not in valid_ad:
                it["kind"] = "other"
        return {"section": section, "suggestions": items[:5], "source": "gemini"}
    return {"section": section, "suggestions": _rule_suggestions(section, ctx), "source": "rules"}


# ---------- Page-aware tips (pop-up that changes with the sidebar section) ----------
SECTION_FOCUS = {
    "overview": "the overall health of the business: profit on ads, the biggest problem, and the one thing to do today",
    "platforms": "which platform (Meta, TikTok, Shopify, Amazon) performs best and worst, and what to change on the weakest",
    "products": "product prices and profit margins: which product to raise the price of, discount, or pause",
    "inventory": "stock levels: which products run out first, how much to reorder, and which ads to limit until restock",
    "ingestion": "data quality: whether any platform's data is slow or missing and what that could hide",
    "causal": "tired ads and low-stock risks, and which one to fix first",
    "poas": "which campaigns earn or lose money per ₹1 of ads and where to move budget",
    "actions": "which suggested action to approve first and why",
    "timeline": "the month-wise trend in units sold, views and reviews, and what changed most recently",
    "ads": "which product-platform ads earn or lose money after ad spend, which to pause or stop, and where to add budget",
}

_TIPS = {
    "overview": ("Fix the tired ad first", "Profit is held back by one tired Meta ad. Approve the suggested stop in Things To Do, then give TikTok more budget."),
    "platforms": ("TikTok is your best earner", "TikTok earns the most profit per ₹1. Amazon earns the least, so check its listing photos and ad spend."),
    "products": ("Protect your margins", "Linen Shirt sells fastest, so avoid discounts until it is back in stock. Check products with a margin under 50%."),
    "inventory": ("Reorder two products now", "Linen Shirt and Overshirt will run out within a week. Add stock soon and limit their ads until it arrives."),
    "ingestion": ("Amazon data is running slow", "Amazon updated 14 minutes ago while others updated within 3. Recent Amazon sales may be missing from totals."),
    "causal": ("Two problems need action", "One ad is tired and two products are almost out of stock. Fixing the tired ad saves the most money."),
    "poas": ("Stop money-losing campaigns", "Campaigns below 1.00 lose money on every ₹1 spent. Move that budget to the campaign with the highest number."),
    "actions": ("Start with the 96% sure item", "Stopping the Amazon ad on the linen shirt is the safest change because the product is almost out of stock."),
    "timeline": ("Compare this month to last", "Use the product and platform filters to see which channel is growing. The current month only counts days so far."),
    "ads": ("Move budget to what sells", "Check profit after ads on each platform. Pause ads where it is negative and stop ads on products almost out of stock."),
}


def get_section_tip(section: str, context_data: dict) -> dict[str, Any]:
    focus = SECTION_FOCUS.get(section, SECTION_FOCUS["overview"])
    prompt = (
        f"The owner just opened the '{section}' page. Give one useful, specific tip about {focus}. "
        "Use the real numbers below.\n" + _json(context_data)
    )
    data = _generate("tip-" + section, prompt, SectionTip, 0.4)
    if data:
        return {"section": section, "title": data["title"], "tip": data["tip"], "source": "gemini"}
    title, tip = _TIPS.get(section, _TIPS["overview"])
    return {"section": section, "title": title, "tip": tip, "source": "fallback"}
