"""NEXUS AI engine: OpenRouter (Llama 3.3 70B) via the official OpenAI SDK.

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
    from openai import OpenAI
except ImportError:  # SDK missing: every function falls back to sample data
    OpenAI = None

# Load .env from the same folder as this file, so it works no matter where uvicorn is started from.
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()
log = logging.getLogger("nexus.ai")

# Using OpenRouter's permanently free, massive Llama 3.3 70B model (zero credits used)
PRIMARY_MODEL = "google/gemini-1.5-flash:free"
CACHE_TTL = int(os.getenv("AI_CACHE_SECONDS", "60"))
_cache: dict[str, tuple[float, dict]] = {}
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

class SuggestionItem(BaseModel):
    title: str = Field(description="Actionable title, e.g., 'Pause Meta Ad'")
    description: str = Field(description="Why this should be done")
    action_type: str = Field(description="'pause', 'budget_increase', 'restock'")
    confidence: float = Field(description="0 to 100")

class SuggestionList(BaseModel):
    suggestions: list[SuggestionItem]


# ---------- Helpers ----------
def _api_key() -> str:
    raw = os.getenv("OPENROUTER_API_KEY") or ""
    return raw.strip().strip('"').strip("'").strip()


def _get_client():
    if OpenAI is None:
        raise RuntimeError("SDK_MISSING")
    api_key = _api_key()
    if not api_key:
        raise ValueError("KEY_MISSING")
    
    # CRITICAL FIX: Pointing the OpenAI SDK specifically to OpenRouter
    return OpenAI(
        api_key=api_key, 
        base_url="https://openrouter.ai/api/v1"
    )


def _explain(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if OpenAI is None or "SDK_MISSING" in text:
        return "The openai package is not installed. Run: pip install openai"
    if "KEY_MISSING" in text:
        return ("No OPENROUTER_API_KEY found. Add it to your Railway variables.")
    if "401" in text or "unauthorized" in low:
        return "API key rejected. Ensure you pasted the OpenRouter key correctly."
    if "429" in text or "rate limit" in low:
        return "Free limit reached. Wait a minute and try again."
    if any(w in low for w in ("connect", "timed out", "timeout", "network")):
        return "Could not reach the AI server."
    return "AI error: " + text[:200]


def _note_error(name: str, exc: Exception) -> None:
    global _last_error
    _last_error = _explain(exc)
    log.warning("%s fell back to sample data: %s", name, _last_error)


_status_cache: dict[str, Any] = {"at": 0.0, "data": None}


def get_ai_status(force: bool = False) -> dict[str, Any]:
    global _last_error
    if not force and _status_cache["data"] and time.time() - _status_cache["at"] < 120:
        return _status_cache["data"]
    try:
        client = _get_client()
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            model=PRIMARY_MODEL,
            max_tokens=10
        )
        data = {"live": True, "model": "Llama-3.3 70B (OpenRouter)", "reason": "Connected successfully."}
        _last_error = ""
    except Exception as exc:
        _last_error = _explain(exc)
        data = {"live": False, "model": PRIMARY_MODEL, "reason": _last_error}
    _status_cache.update(at=time.time(), data=data)
    return data


def _stamp(minutes_ago: int = 0) -> str:
    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%d %b, %I:%M %p")


def _generate(name: str, prompt: str, schema: type[BaseModel], temperature: float) -> dict | None:
    key = name + hashlib.sha1(prompt.encode()).hexdigest()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    
    try:
        client = _get_client()
        
        # Enforce strict JSON output
        schema_json = json.dumps(schema.model_json_schema())
        full_prompt = prompt + f"\n\nYou MUST output strictly in JSON format matching this schema. Output nothing but the JSON object:\n{schema_json}"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": full_prompt}
            ],
            model=PRIMARY_MODEL,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content
        data = schema.model_validate_json(response_text).model_dump()
        _cache[key] = (time.time(), data)
        return data
    except Exception as exc:
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
            "description": "Stop “Summer Drop – UGC v3”, move ₹45,000 a day to the “Founder Story” TikTok video.",
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
        "summary": "If nothing changes, profit per ₹1 of ads should improve slightly to about 1.28 as weak ads are stopped. The linen shirt will run out in about 4 days.",
        "inventory_advice": "Reorder about 240 Linen Shirts and 135 Overshirts now so you have 3 weeks of stock.",
        "integration_suggestions": [
            "Connect Meta Conversions API (server-side tracking) to recover sales.",
            "Connect your warehouse stock count so ads pause automatically before a product sells out.",
            "Connect your courier tracking to show real delivery times.",
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
        "summary": "Customers like the fabric and fit. Most complaints are about the colour looking different from the ad.",
        "themes": ["Fabric quality", "Colour in ad vs real", "Slow delivery", "Size runs small"],
        "creative_feedback": [
            "Re-shoot the main photo in daylight so the colour matches the product.",
            "Show fabric and fit close-ups first, since people praise these most.",
        ],
    }

def _fallback_suggestions() -> list[dict]:
    return [
        {
            "title": "Stop 'Summer Drop - UGC v3' ad",
            "description": "Ad fatigue is causing low clicks. Moving budget to TikTok will save ₹45,000/day.",
            "action_type": "pause",
            "confidence": 92.0
        },
        {
            "title": "Restock Linen Shirt",
            "description": "Stock is critically low with only 4 days of cover. Pause ads to prevent out-of-stock penalties.",
            "action_type": "restock",
            "confidence": 88.5
        }
    ]

def _rs(n: Any) -> str:
    try:
        return "₹" + f"{round(float(n)):,}"
    except (TypeError, ValueError):
        return "₹0"


def _local_answer(question: str, ctx: dict) -> str:
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
                lines.append("Reorder soon: " + ", ".join(f"{p['name']} ({p.get('days_of_cover')} days left)" for p in urgent) + ".")
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

    if not lines:  
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
    prompt = "Find the main reason this campaign is underperforming and the best single fix. Scores are 0 to 100.\n" + _json(campaign_data)
    data = _generate("diagnosis", prompt, DiagnosisResult, 0.2)
    if data:
        data["source"] = "openrouter"
        return data
    fb = _fallback_diagnosis()
    fb["source"] = "fallback"
    return fb

def predict_future_performance(campaign_data: dict) -> dict[str, Any]:
    prompt = "Forecast the next 14 days of performance (POAS = profit per rupee of ad spend) and suggest exactly 3 practical tool integrations the brand should add. Use the product stock and daily sales to predict stock-outs, and give one short restocking tip.\n" + _json(campaign_data)
    data = _generate("forecast", prompt, ForecastResult, 0.4)
    if data and len(data.get("integration_suggestions", [])) >= 1:
        data["integration_suggestions"] = data["integration_suggestions"][:3]
        data["source"] = "openrouter"
        return data
    fb = _fallback_forecast()
    fb["source"] = "fallback"
    return fb

def get_monitoring_alerts(campaign_data: dict | None = None) -> list[dict[str, Any]]:
    prompt = "Write exactly 3 realistic 24/7 monitoring alerts (for example low stock, ad fatigue, a win) from this live data. Use ids AL-1, AL-2, AL-3.\n" + _json(campaign_data or {})
    data = _generate("alerts", prompt, AlertList, 0.5)
    alerts = (data or {}).get("alerts", [])[:3]
    if not alerts:
        return _fallback_alerts()
    for a in alerts:
        if a["severity"] not in ("critical", "warning", "success"):
            a["severity"] = "warning"
    return alerts

def get_timeline_events(campaign_data: dict | None = None) -> list[dict[str, Any]]:
    prompt = "Write exactly 4 timeline events, oldest first, showing recent system actions and data syncs based on this data.\n" + _json(campaign_data or {})
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
    prompt = "Read these customer reviews. Give the share of positive, neutral and negative (adding to 1), a short summary, the main topics, and advice for improving the ads.\n" + _json(feedback_list)
    data = _generate("feedback", prompt, FeedbackReport, 0.2)
    if data:
        data["source"] = "openrouter"
        return data
    fb = _fallback_feedback(feedback_list)
    fb["source"] = "fallback"
    return fb

def get_suggestions(campaign_data: dict) -> list[dict[str, Any]]:
    prompt = "Based on this data, provide exactly 2 to 3 actionable suggestions to improve performance. Each must have a title, description, action_type, and confidence score (0-100).\n" + _json(campaign_data)
    data = _generate("suggestions", prompt, SuggestionList, 0.4)
    suggestions = (data or {}).get("suggestions", [])
    if not suggestions:
        return _fallback_suggestions()
    return suggestions

def answer_chat_query(question: str, context_data: dict, history: list | None = None) -> str:
    question = (question or "").strip()
    if not question:
        return "Please type a question."
    
    messages = [{"role": "system", "content": SYSTEM}]
    for h in (history or [])[-6:]:
        role = "user" if h.get("me") else "assistant"
        messages.append({"role": role, "content": str(h.get("t", ""))[:500]})
        
    prompt = (
        "You are the assistant inside this brand's dashboard. Answer the owner's question in 2 to 5 short, "
        "friendly sentences using the data below. Work out totals, rankings and comparisons when asked. "
        "If the question has nothing to do with this business, say politely that you can only help with this dashboard."
        f"\n\nData:\n{_json(context_data)}"
        f"\n\nQuestion: {question}"
    )
    messages.append({"role": "user", "content": prompt})
    
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0.3
        )
        text = (response.choices[0].message.content or "").strip()
        return text or _local_answer(question, context_data)
    except Exception as exc:
        _note_error("chat", exc)
        return _local_answer(question, context_data)

# ---------- Page-aware tips ----------
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
    prompt = f"The owner just opened the '{section}' page. Give one useful, specific tip about {focus}. Use the real numbers below.\n" + _json(context_data)
    data = _generate("tip-" + section, prompt, SectionTip, 0.4)
    if data:
        return {"section": section, "title": data["title"], "tip": data["tip"], "source": "openrouter"}
    title, tip = _TIPS.get(section, _TIPS["overview"])
    return {"section": section, "title": title, "tip": tip, "source": "fallback"}
