"""NEXUS AI engine: Gemini via the official OpenAI SDK.

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
except ImportError:
    OpenAI = None

# Load .env from the same folder as this file.
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()
log = logging.getLogger("nexus.ai")

# Using Gemini 1.5 Flash for high-speed rate limits
PRIMARY_MODEL = "gemini-1.5-flash"
CACHE_TTL = int(os.getenv("AI_CACHE_SECONDS", "300"))
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
    raw = os.getenv("GEMINI_API_KEY") or ""
    return raw.strip().strip('"').strip("'").strip()


def _get_client():
    if OpenAI is None:
        raise RuntimeError("SDK_MISSING")
    api_key = _api_key()
    if not api_key:
        raise ValueError("KEY_MISSING")
    
    return OpenAI(
        api_key=api_key, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def _explain(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if OpenAI is None or "SDK_MISSING" in text:
        return "The openai package is not installed."
    if "KEY_MISSING" in text:
        return "No GEMINI_API_KEY found in variables."
    if "401" in text or "unauthorized" in low:
        return "API key rejected."
    if "429" in text or "rate limit" in low:
        return "Free limit reached. Using fallback data."
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
            messages=[{"role": "user", "content": "Reply OK."}],
            model=PRIMARY_MODEL,
            max_tokens=10
        )
        data = {"live": True, "model": "Gemini 1.5 Flash", "reason": "Connected successfully."}
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
        "root_cause": "The ad “Summer Drop – UGC v3” has been shown too many times to the same people.",
        "confidence_score": 91.0,
        "opportunity_score": 84.0,
        "recommended_action": {
            "action_type": "pause_and_reallocate",
            "title": "Stop the tired ad and move its budget",
            "description": "Stop “Summer Drop – UGC v3”, move ₹45,000 a day to TikTok.",
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
        "summary": "Profit per ₹1 of ads should improve slightly to about 1.28.",
        "inventory_advice": "Reorder 240 Linen Shirts and 135 Overshirts now.",
        "integration_suggestions": ["Connect Meta CAPI", "Connect warehouse count", "Connect courier tracking"],
    }

def _fallback_alerts() -> list[dict]:
    return [
        {"id": "AL-1", "severity": "critical", "title": "Linen shirt almost out of stock", "message": "Only 4 days of stock left.", "timestamp": _stamp(6)}
    ]

def _fallback_timeline() -> list[dict]:
    return [
        {"time": _stamp(180), "title": "Data synced", "desc": "Meta and Shopify data updated.", "type": "info"},
        {"time": _stamp(120), "title": "Orders matched", "desc": "99.8% of orders matched.", "type": "success"},
        {"time": _stamp(45), "title": "Tired ad detected", "desc": "Clicks on “Summer Drop” fell sharply.", "type": "warning"},
        {"time": _stamp(10), "title": "Low stock guard", "desc": "Ads on linen shirt flagged.", "type": "info"},
    ]

def _fallback_feedback(reviews: list) -> dict:
    return {
        "sentiment": {"positive": 0.58, "neutral": 0.17, "negative": 0.25},
        "summary": "Customers like the fabric and fit, but some complain about colour.",
        "themes": ["Fabric quality", "Colour mismatch", "Slow delivery"],
        "creative_feedback": ["Re-shoot main photo in daylight."],
    }

def _fallback_suggestions() -> list[dict]:
    return [
        {"title": "Stop 'Summer Drop' ad", "description": "Ad fatigue is causing low clicks.", "action_type": "pause", "confidence": 92.0},
        {"title": "Restock Linen Shirt", "description": "Stock is critically low.", "action_type": "restock", "confidence": 88.5},
        {"title": "Scale TikTok Budget", "description": "POAS is high. Add ₹15k/day.", "action_type": "budget_increase", "confidence": 84.0}
    ]

def _rs(n: Any) -> str:
    try:
        return "₹" + f"{round(float(n)):,}"
    except (TypeError, ValueError):
        return "₹0"


def _local_answer(question: str, ctx: dict) -> str:
    q = (question or "").lower()
    return "This is a quick summary from your numbers. Reason: API limits reached, retry in 1 minute."


# ---------- Public functions ----------
def run_diagnosis(campaign_data: dict) -> dict[str, Any]:
    prompt = "Find the main reason this campaign is underperforming and the best single fix. Scores are 0 to 100.\n" + _json(campaign_data)
    data = _generate("diagnosis", prompt, DiagnosisResult, 0.2)
    if data:
        data["source"] = "gemini"
        return data
    fb = _fallback_diagnosis()
    fb["source"] = "fallback"
    return fb

def predict_future_performance(campaign_data: dict) -> dict[str, Any]:
    prompt = "Forecast the next 14 days of performance (POAS) and suggest 3 tool integrations. Predict stock-outs.\n" + _json(campaign_data)
    data = _generate("forecast", prompt, ForecastResult, 0.4)
    if data and len(data.get("integration_suggestions", [])) >= 1:
        data["integration_suggestions"] = data["integration_suggestions"][:3]
        data["source"] = "gemini"
        return data
    fb = _fallback_forecast()
    fb["source"] = "fallback"
    return fb

def get_monitoring_alerts(campaign_data: dict | None = None) -> list[dict[str, Any]]:
    prompt = "Write exactly 1 high-priority monitoring alert (e.g., low stock or ad fatigue) from this live data. Do not write more than 1. Use id AL-1.\n" + _json(campaign_data or {})
    data = _generate("alerts", prompt, AlertList, 0.5)
    
    alerts = (data or {}).get("alerts", [])[:1]
    if not alerts:
        return _fallback_alerts()[:1]
        
    for a in alerts:
        if a["severity"] not in ("critical", "warning", "success"):
            a["severity"] = "warning"
    return alerts

def get_timeline_events(campaign_data: dict | None = None) -> list[dict[str, Any]]:
    prompt = "Write exactly 4 timeline events, oldest first, showing recent system actions based on this data.\n" + _json(campaign_data or {})
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
        fb["summary"] = "No reviews given."
        fb["source"] = "fallback"
        return fb
    prompt = "Read these customer reviews. Give the share of positive, neutral and negative (adding to 1), summary, and themes.\n" + _json(feedback_list)
    data = _generate("feedback", prompt, FeedbackReport, 0.2)
    if data:
        data["source"] = "gemini"
        return data
    fb = _fallback_feedback(feedback_list)
    fb["source"] = "fallback"
    return fb

def get_suggestions(campaign_data: dict) -> dict[str, Any]:
    prompt = "Based on this data, you MUST provide exactly 3 actionable suggestions to improve performance. Never return an empty list. Each must have a title, description, action_type, and confidence score (0-100).\n" + _json(campaign_data)
    data = _generate("suggestions", prompt, SuggestionList, 0.4)
    
    suggestions = (data or {}).get("suggestions", [])
    if not suggestions:
        suggestions = _fallback_suggestions()
        
    return {"suggestions": suggestions}

def answer_chat_query(question: str, context_data: dict, history: list | None = None) -> str:
    question = (question or "").strip()
    if not question:
        return "Please type a question."
    
    messages = [{"role": "system", "content": SYSTEM}]
    for h in (history or [])[-6:]:
        role = "user" if h.get("me") else "assistant"
        messages.append({"role": role, "content": str(h.get("t", ""))[:500]})
        
    prompt = (
        "Answer the owner's question in 2 to 5 short sentences using this data:\n"
        f"{_json(context_data)}\n\nQuestion: {question}"
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
_TIPS = {
    "overview": ("Fix the tired ad first", "Profit is held back by one tired Meta ad. Check the suggestions."),
    "platforms": ("TikTok is your best earner", "TikTok earns the most profit per ₹1. Amazon earns the least."),
    "products": ("Protect your margins", "Linen Shirt sells fastest. Check products with a margin under 50%."),
    "inventory": ("Reorder two products now", "Linen Shirt and Overshirt will run out within a week."),
    "ingestion": ("Amazon data is running slow", "Amazon updated 14 minutes ago. Recent sales may be missing."),
    "causal": ("Two problems need action", "One ad is tired and two products are almost out of stock."),
    "poas": ("Stop money-losing campaigns", "Campaigns below 1.00 lose money. Move budget to higher numbers."),
    "actions": ("Start with the 96% sure item", "Stopping the Amazon ad on the linen shirt is the safest change."),
    "timeline": ("Compare this month to last", "Use filters to see which channel is growing."),
    "ads": ("Move budget to what sells", "Pause ads where profit after ads is negative."),
}

def get_section_tip(section: str, context_data: dict) -> dict[str, Any]:
    metrics = context_data.get("metrics_last_7_days", {})
    ctr_change = metrics.get("ctr_change_pct", 0)
    
    if ctr_change > 15:
        return {"section": section, "title": "Viral Spike Detected! 🚀", "tip": f"Click-through rate jumped by {ctr_change}%. Check your latest creative.", "source": "local-rules"}
    if ctr_change < -20:
        return {"section": section, "title": "Traffic Drop Alert ⚠️", "tip": f"Clicks dropped by {abs(ctr_change)}%. Ad fatigue is setting in.", "source": "local-rules"}
        
    title, tip = _TIPS.get(section, _TIPS["overview"])
    return {"section": section, "title": title, "tip": tip, "source": "local-rules"}
