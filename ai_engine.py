"""NEXUS AI engine: Groq (or Gemini) through the OpenAI SDK, using as few requests as possible.

How AI requests are saved
- ONE request creates every insight at once: the opening tip, suggestions, forecast, alert,
  timeline and diagnosis. The result is stored in Supabase (app_state key 'ai_insights') and
  reused by every page, every browser tab and every server restart.
- A new request happens only when you press "Refresh AI" (or after AI_REFRESH_HOURS, if you set it).
- If the AI is unavailable (no key, free limit reached), insights are worked out from your live
  numbers instead, and the AI is tried again after AI_RETRY_MINUTES.
- The chat and the review analyser still use the AI, but only when you ask, with a small data summary.

Environment variables
  GROQ_API_KEY       your Groq key (console.groq.com/keys). Used first when present.
  GEMINI_API_KEY     optional: Google Gemini key, used when there is no Groq key.
  AI_MODEL           optional: model name (default openai/gpt-oss-120b on Groq, gemini-2.5-flash on Gemini)
  AI_REFRESH_HOURS   optional: refresh AI insights automatically after this many hours (default 0 = only on "Refresh AI")
  AI_RETRY_MINUTES   optional: after a failed AI request, wait this long before trying again (default 30)
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # the dashboard still works without the AI package
    OpenAI = None

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()
log = logging.getLogger("nexus.ai")

INSIGHTS_KEY = "ai_insights"
REFRESH_HOURS = float(os.getenv("AI_REFRESH_HOURS", "0") or 0)
RETRY_MINUTES = float(os.getenv("AI_RETRY_MINUTES", "30") or 30)

SYSTEM = (
    "You are NEXUS, an advertising and stock analyst for a small Indian direct-to-consumer brand. "
    "Write in simple, plain English a shop owner understands. Money is Indian rupees (₹). "
    "Use only the data given and never invent numbers that cannot be worked out from it."
)

_last_error = ""
_gen_lock = threading.Lock()


# =====================================================================
# Talking to the AI provider
# =====================================================================
def _clean(v: str | None) -> str:
    return (v or "").strip().strip('"').strip("'").strip()


def _provider() -> dict[str, str] | None:
    """Groq first (your .env has GROQ_API_KEY), Gemini as a backup. None if there is no key."""
    groq, gemini = _clean(os.getenv("GROQ_API_KEY")), _clean(os.getenv("GEMINI_API_KEY"))
    model = _clean(os.getenv("AI_MODEL"))
    if groq:
        return {"name": "groq", "key": groq, "url": "https://api.groq.com/openai/v1",
                "model": model or "openai/gpt-oss-120b"}
    if gemini:
        return {"name": "gemini", "key": gemini, "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model": model or "gemini-2.5-flash"}
    return None


def _explain(exc: Exception) -> str:
    """A plain-English reason the dashboard can show."""
    text, name = str(exc), type(exc).__name__
    low = text.lower()
    if "SDK_MISSING" in text:
        return "The openai package is not installed. Run: pip install openai"
    if "KEY_MISSING" in text:
        return "No AI key found. Add GROQ_API_KEY in Railway Variables (or .env)."
    if name == "RateLimitError" or "429" in text or "rate limit" in low:
        return f"The free AI limit was reached. Insights come from your numbers for now; the AI is tried again in {int(RETRY_MINUTES)} minutes."
    if name in ("AuthenticationError", "PermissionDeniedError") or "401" in text or "invalid api key" in low:
        return "The AI key was rejected. Create a new key at console.groq.com/keys and update GROQ_API_KEY."
    if name == "NotFoundError" or "model" in low and ("not found" in low or "decommissioned" in low or "does not exist" in low):
        return "The AI model name is not available. Set AI_MODEL to a current model from console.groq.com/docs/models."
    if name in ("APIConnectionError", "APITimeoutError") or any(w in low for w in ("connect", "timed out", "timeout")):
        return "Could not reach the AI service. Check the internet connection."
    return "AI error: " + text[:180]


def _complete(messages: list[dict], max_tokens: int, json_mode: bool, temperature: float = 0.3) -> str:
    """One request to the AI. No automatic retries, so a limit error never costs extra requests."""
    global _last_error
    if OpenAI is None:
        raise RuntimeError("SDK_MISSING")
    prov = _provider()
    if not prov:
        raise ValueError("KEY_MISSING")
    client = OpenAI(api_key=prov["key"], base_url=prov["url"], timeout=40, max_retries=0)
    kw: dict[str, Any] = {"model": prov["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    if "gpt-oss" in prov["model"]:
        kw["extra_body"] = {"reasoning_effort": "low"}  # less hidden "thinking" = fewer tokens used
    try:
        resp = client.chat.completions.create(**kw)
    except Exception as exc:
        # some models refuse JSON mode or the reasoning option: try once more without them
        if type(exc).__name__ == "BadRequestError" and ("response_format" in str(exc) or "reasoning" in str(exc)):
            kw.pop("response_format", None)
            kw.pop("extra_body", None)
            resp = client.chat.completions.create(**kw)
        else:
            raise
    _last_error = ""
    return (resp.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start >= 0 and end > start else {}


# =====================================================================
# A small summary of the live data (keeps every AI request cheap)
# =====================================================================
def _num(v: Any, d: int = 0) -> float:
    try:
        return round(float(v), d) if d else round(float(v))
    except (TypeError, ValueError):
        return 0


def compact(ctx: dict) -> dict:
    prods = [{"name": p.get("name"), "price": _num(p.get("price")), "cost": _num(p.get("cost")),
              "stock": p.get("stock"), "sells_per_day": p.get("daily_sales"), "days_left": p.get("days_of_cover"),
              "margin_pct": p.get("margin_pct"), "units_sold": p.get("units_sold"), "status": p.get("status")}
             for p in (ctx.get("products") or [])][:25]
    ads = sorted(ctx.get("advertising") or [], key=lambda a: -(a.get("spend") or 0))
    ad_rows = [{"product": a.get("product"), "platform": a.get("platform"), "status": a.get("status"),
                "budget_per_day": _num(a.get("daily_budget")), "views": a.get("views"), "orders": a.get("orders"),
                "spend": a.get("spend"), "profit_after_ads": a.get("profit_after_ads"), "rating": a.get("avg_rating")}
               for a in ads if a.get("status") in ("active", "paused")][:30]
    months = [{k: m.get(k) for k in ("label", "units", "sales", "views", "ad_spend", "reviews", "avg_rating")}
              for m in (ctx.get("monthly_all_products") or [])[-3:]]
    return {"products": prods, "ads": ad_rows, "last_3_months": months,
            "meta_campaign_last_7_days": ctx.get("metrics_last_7_days", {}),
            "connected_platforms": ctx.get("connected_platforms", [])}


def _rs(n: Any) -> str:
    try:
        return "₹" + f"{round(float(n)):,}"
    except (TypeError, ValueError):
        return "₹0"


# =====================================================================
# Insights worked out from the live numbers (used when the AI is not available)
# =====================================================================
def _stamp(minutes_ago: int = 0) -> str:
    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%d %b, %I:%M %p")


def _issues(ctx: dict) -> list[dict]:
    """The real problems and opportunities in the data, most important first."""
    prods = [p for p in (ctx.get("products") or []) if p.get("status") == "active"]
    by_name = {p["name"]: p for p in prods}
    ads = [a for a in (ctx.get("advertising") or []) if a.get("status") == "active" and a.get("connected", True)]
    out = []
    for p in sorted(prods, key=lambda x: x.get("days_of_cover", 999)):
        running = sum(1 for a in ads if a.get("product") == p["name"])
        if p.get("stock", 0) <= 0:
            out.append({"kind": "restock", "sev": "critical", "score": 100, "title": f"Restock {p['name']} now",
                        "why": f"{p['name']} is out of stock" + (f" while {running} ads still cost money." if running else "."),
                        "fix": "Add stock in Inventory and pause its ads until it arrives."})
        elif p.get("days_of_cover", 999) < 7:
            need = max(0, round((p.get("daily_sales") or 0) * 21 - p["stock"]))
            out.append({"kind": "restock", "sev": "critical" if p["days_of_cover"] < 4 else "warning",
                        "score": 90 - p["days_of_cover"], "title": f"Reorder {p['name']}",
                        "why": f"Only {p['stock']} left, about {p['days_of_cover']} days at {p.get('daily_sales')} a day.",
                        "fix": f"Reorder about {need} units to cover 3 weeks."})
    for a in ads:
        spend = a.get("spend") or 0
        p = by_name.get(a.get("product"))
        if not spend or not p:
            continue
        ppr = (a.get("orders") or 0) * (p["price"] - p["cost"]) / spend
        a["_ppr"] = ppr
        if ppr < 1:
            out.append({"kind": "pause", "sev": "warning", "score": 70 - ppr * 10,
                        "title": f"Pause {a['product']} ad on {a['platform']}",
                        "why": f"It earns only ₹{ppr:.2f} of profit per ₹1 of ads, so it loses money.",
                        "fix": f"Pause it or lower its budget from {_rs(a.get('daily_budget'))} a day."})
        if a.get("avg_rating") and a["avg_rating"] < 3.8 and (a.get("review_count") or 0) >= 5:
            out.append({"kind": "reviews", "sev": "warning", "score": 55,
                        "title": f"Fix reviews for {a['product']} on {a['platform']}",
                        "why": f"Rated {a['avg_rating']} stars from {a['review_count']} reviews.",
                        "fix": "Read the reviews and fix the main complaint."})
    good = sorted([a for a in ads if a.get("_ppr", 0) > 1.5], key=lambda a: -a["_ppr"])
    if good:
        a = good[0]
        p = by_name[a["product"]]
        if p.get("days_of_cover", 999) >= 14:
            out.append({"kind": "budget_increase", "sev": "success", "score": 50,
                        "title": f"Spend more on {a['product']} on {a['platform']}",
                        "why": f"It earns ₹{a['_ppr']:.2f} profit per ₹1 of ads and has {p['days_of_cover']} days of stock.",
                        "fix": f"Raise its budget from {_rs(a.get('daily_budget'))} by about a quarter."})
    return sorted(out, key=lambda x: -x["score"])


def rules_insights(ctx: dict) -> dict:
    issues = _issues(ctx)
    prods = ctx.get("products") or []
    months = ctx.get("monthly_all_products") or []
    ads = [a for a in (ctx.get("advertising") or []) if a.get("status") == "active" and a.get("spend")]
    by_name = {p["name"]: p for p in prods}
    gross = sum((a.get("orders") or 0) * (by_name[a["product"]]["price"] - by_name[a["product"]]["cost"])
                for a in ads if a.get("product") in by_name)
    spend = sum(a.get("spend") or 0 for a in ads)
    poas = round(gross / spend, 2) if spend else 0.0
    growth = 0.0
    if len(months) >= 2 and months[-2].get("sales"):
        day = max(datetime.now().day, 1)
        growth = round(((months[-1].get("sales") or 0) / day) / (months[-2]["sales"] / 30) * 100 - 100, 1)
    first = min(prods, key=lambda p: p.get("days_of_cover", 999)) if prods else None
    ctr = (ctx.get("metrics_last_7_days") or {}).get("ctr_change_pct", 0)
    top = issues[0] if issues else None
    low = [p for p in prods if p.get("days_of_cover", 999) < 7]

    suggestions = [{"title": i["title"], "description": f"{i['why']} {i['fix']}",
                    "action_type": "restock" if i["kind"] == "restock" else "pause" if i["kind"] == "pause"
                    else "budget_increase" if i["kind"] == "budget_increase" else "other",
                    "confidence": 90.0 if i["sev"] == "critical" else 80.0} for i in issues[:3]]
    if not suggestions:
        suggestions = [{"title": "Nothing urgent", "description": "Stock, ads and ratings look healthy right now.",
                        "action_type": "other", "confidence": 70.0}]
    return {
        "tip": {"title": top["title"] if top else "Everything looks healthy",
                "tip": f"{top['why']} {top['fix']}" if top else "No urgent problems. Keep an eye on stock and ad results."},
        "suggestions": suggestions,
        "forecast": {
            "forecast_horizon": "Next 14 days",
            "predicted_poas": poas,
            "predicted_revenue_growth_pct": growth,
            "fatigue_risk_level": "high" if ctr <= -25 else "medium" if ctr < -10 else "low",
            "stockout_risk_days": int(first.get("days_of_cover", 999)) if first else 999,
            "summary": (f"Ads currently earn ₹{poas} of profit for every ₹1 spent. "
                        f"Sales this month are running {abs(growth)}% {'faster' if growth >= 0 else 'slower'} than last month."),
            "inventory_advice": ("Reorder " + " and ".join(f"{p['name']} ({p['days_of_cover']} days left)" for p in low) + " now."
                                 if low else "No product runs out in the next week."),
            "integration_suggestions": ["Connect your real Shopify orders", "Connect Meta Conversions API",
                                        "Connect courier tracking"],
        },
        "alerts": [{"id": "AL-1", "severity": top["sev"] if top and top["sev"] in ("critical", "warning", "success") else "success",
                    "title": top["title"] if top else "All good", "message": f"{top['why']} {top['fix']}" if top else
                    "No urgent problems right now.", "timestamp": _stamp(1)}],
        "timeline": [
            {"time": _stamp(90), "title": "Orders synced", "desc": "Orders from connected platforms were added and stock updated.", "type": "info"},
            {"time": _stamp(60), "title": "Ad results updated", "desc": f"{len(ads)} running ads checked for profit.", "type": "success"},
            {"time": _stamp(30), "title": "Stock checked", "desc": (f"{len(low)} product(s) run out within a week." if low else "No product runs out within a week."),
             "type": "warning" if low else "success"},
            {"time": _stamp(5), "title": "Insights ready", "desc": "Suggestions were worked out from your latest numbers.", "type": "info"},
        ],
        "diagnosis": {
            "root_cause": top["why"] if top else "No major problem found in the current numbers.",
            "confidence_score": 85.0 if top else 60.0,
            "opportunity_score": 80.0 if top else 40.0,
            "recommended_action": {"action_type": top["kind"] if top else "none", "title": top["title"] if top else "Keep going",
                                   "description": top["fix"] if top else "Keep monitoring stock and ads.",
                                   "target": top["title"] if top else ""},
        },
    }


# =====================================================================
# The ONE combined AI request, saved in Supabase
# =====================================================================
_SKELETON = """{
 "tip": {"title": "at most 6 words", "tip": "the single most important thing to do, max 2 short sentences"},
 "suggestions": [{"title": "short action", "description": "why, with real numbers, and what to do",
                  "action_type": "restock | pause | budget_increase | other", "confidence": 0-100}]  (exactly 3),
 "forecast": {"forecast_horizon": "Next 14 days", "predicted_poas": number, "predicted_revenue_growth_pct": number,
              "fatigue_risk_level": "low | medium | high", "stockout_risk_days": integer,
              "summary": "2 sentences", "inventory_advice": "1 sentence", "integration_suggestions": ["3 short items"]},
 "alerts": [{"id": "AL-1", "severity": "critical | warning | success", "title": "short", "message": "1 sentence", "timestamp": "now"}] (exactly 1),
 "timeline": [{"time": "e.g. 2 hours ago", "title": "short", "desc": "1 sentence", "type": "info | warning | success"}] (exactly 4, oldest first),
 "diagnosis": {"root_cause": "1-2 sentences", "confidence_score": 0-100, "opportunity_score": 0-100,
               "recommended_action": {"action_type": "short", "title": "short", "description": "1 sentence", "target": "what it applies to"}}
}"""


def _merge(ai: dict, base: dict) -> dict:
    """Use the AI's answer where it is complete; fill anything missing or broken from the live-number version."""
    out = dict(base)
    if isinstance(ai.get("tip"), dict) and ai["tip"].get("title") and ai["tip"].get("tip"):
        out["tip"] = {"title": str(ai["tip"]["title"]), "tip": str(ai["tip"]["tip"])}
    sug = [s for s in (ai.get("suggestions") or []) if isinstance(s, dict) and s.get("title") and s.get("description")]
    if sug:
        out["suggestions"] = [{"title": str(s["title"]), "description": str(s["description"]),
                               "action_type": str(s.get("action_type") or "other"),
                               "confidence": float(_num(s.get("confidence"), 1) or 75)} for s in sug[:3]]
    f = ai.get("forecast")
    if isinstance(f, dict) and f.get("summary"):
        fc = dict(base["forecast"])
        for k in fc:
            if f.get(k) not in (None, ""):
                fc[k] = f[k]
        try:
            fc["predicted_poas"] = float(fc["predicted_poas"])
            fc["predicted_revenue_growth_pct"] = float(fc["predicted_revenue_growth_pct"])
            fc["stockout_risk_days"] = int(float(fc["stockout_risk_days"]))
        except (TypeError, ValueError):
            fc = dict(base["forecast"])
        if fc.get("fatigue_risk_level") not in ("low", "medium", "high"):
            fc["fatigue_risk_level"] = base["forecast"]["fatigue_risk_level"]
        fc["integration_suggestions"] = [str(x) for x in (fc.get("integration_suggestions") or [])][:3] or base["forecast"]["integration_suggestions"]
        out["forecast"] = fc
    al = [a for a in (ai.get("alerts") or []) if isinstance(a, dict) and a.get("title") and a.get("message")][:1]
    if al:
        a = al[0]
        out["alerts"] = [{"id": "AL-1", "severity": a.get("severity") if a.get("severity") in ("critical", "warning", "success") else "warning",
                          "title": str(a["title"]), "message": str(a["message"]), "timestamp": _stamp(0)}]
    tl = [e for e in (ai.get("timeline") or []) if isinstance(e, dict) and e.get("title")][:4]
    if len(tl) >= 2:
        out["timeline"] = [{"time": str(e.get("time") or ""), "title": str(e["title"]), "desc": str(e.get("desc") or ""),
                            "type": e.get("type") if e.get("type") in ("info", "warning", "success") else "info"} for e in tl]
    d = ai.get("diagnosis")
    if isinstance(d, dict) and d.get("root_cause"):
        ra = d.get("recommended_action") if isinstance(d.get("recommended_action"), dict) else {}
        out["diagnosis"] = {"root_cause": str(d["root_cause"]),
                            "confidence_score": float(_num(d.get("confidence_score"), 1) or 80),
                            "opportunity_score": float(_num(d.get("opportunity_score"), 1) or 75),
                            "recommended_action": {k: str(ra.get(k) or base["diagnosis"]["recommended_action"][k])
                                                   for k in ("action_type", "title", "description", "target")}}
    return out


def _generate_bundle(ctx: dict) -> dict:
    global _last_error
    base = rules_insights(ctx)
    now = datetime.now()
    meta = {"generated_at": time.time(), "generated_label": now.strftime("%d %b, %I:%M %p")}
    prov = _provider()
    if not prov or OpenAI is None:
        _last_error = _explain(ValueError("KEY_MISSING") if OpenAI else RuntimeError("SDK_MISSING"))
        return {**base, **meta, "source": "rules", "model": None, "reason": _last_error}
    prompt = ("Study this shop data and reply with ONE JSON object in exactly this shape:\n" + _SKELETON +
              "\n\nData:\n" + json.dumps(compact(ctx), ensure_ascii=False, separators=(",", ":"), default=str))
    try:
        text = _complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                         max_tokens=1800, json_mode=True, temperature=0.3)
        ai = _parse_json(text)
        if not ai:
            raise ValueError("The AI reply was not valid JSON")
        return {**_merge(ai, base), **meta, "source": prov["name"], "model": prov["model"], "reason": ""}
    except Exception as exc:
        _last_error = _explain(exc)
        log.warning("AI insights request failed, using live-number insights: %s", _last_error)
        return {**base, **meta, "source": "rules", "model": prov["model"], "reason": _last_error}


def _load() -> dict | None:
    try:
        import db
        row = db.one("app_state", key=INSIGHTS_KEY)
        return (row or {}).get("value") or None
    except Exception as exc:
        log.warning("Could not read saved AI insights: %s", exc)
        return None


def _save(bundle: dict) -> None:
    try:
        import db
        db.client().table("app_state").upsert({"key": INSIGHTS_KEY, "value": bundle}, on_conflict="key").execute()
    except Exception as exc:
        log.warning("Could not save AI insights: %s", exc)


def _is_stale(b: dict) -> bool:
    now = time.time()
    if b.get("retry_at") and now < float(b["retry_at"]):
        return False  # a recent AI request failed: wait before trying again
    age = now - float(b.get("generated_at") or 0)
    if b.get("source") == "rules":
        return age > RETRY_MINUTES * 60  # no AI result yet: try the AI again after a pause
    return REFRESH_HOURS > 0 and age > REFRESH_HOURS * 3600


def get_insights(context_fn: Callable[[], dict], force: bool = False) -> dict:
    """Saved insights, creating them (with ONE AI request) only if there are none yet,
    if you pressed Refresh AI, or if an automatic refresh/retry is due."""
    b = _load()
    if b and not force and not _is_stale(b):
        return b
    with _gen_lock:  # requests arriving together wait for the same single AI request
        b = _load()
        age = time.time() - float((b or {}).get("generated_at") or 0)
        if b and ((not force and not _is_stale(b)) or (force and age < 30)):
            return b  # created a moment ago (also stops double-clicks on Refresh)
        new = _generate_bundle(context_fn())
        if new["source"] == "rules" and b and b.get("source") != "rules":
            # the AI failed but earlier AI insights exist: keep showing those, and wait before retrying
            b["retry_at"] = time.time() + RETRY_MINUTES * 60
            b["reason"] = new["reason"]
            _save(b)
            return b
        _save(new)
        return new


def insights_meta(b: dict) -> dict:
    return {"source": b.get("source"), "model": b.get("model"), "generated_label": b.get("generated_label"),
            "generated_at": b.get("generated_at"), "reason": b.get("reason", ""),
            "auto_refresh_hours": REFRESH_HOURS}


# =====================================================================
# Public functions used by main.py (same return shapes as before)
# =====================================================================
def _src(b: dict) -> str:
    return "gemini" if b.get("source") not in ("rules", None) else "fallback"  # the page treats any AI source the same


def run_diagnosis(context_fn: Callable[[], dict]) -> dict[str, Any]:
    b = get_insights(context_fn)
    return {**b["diagnosis"], "source": _src(b)}


def predict_future_performance(context_fn: Callable[[], dict]) -> dict[str, Any]:
    b = get_insights(context_fn)
    return {**b["forecast"], "source": _src(b)}


def get_monitoring_alerts(context_fn: Callable[[], dict]) -> list[dict[str, Any]]:
    return get_insights(context_fn)["alerts"]


def get_timeline_events(context_fn: Callable[[], dict]) -> list[dict[str, Any]]:
    return get_insights(context_fn)["timeline"]


def get_suggestions(context_fn: Callable[[], dict]) -> dict[str, Any]:
    b = get_insights(context_fn)
    return {"suggestions": b["suggestions"], "source": _src(b)}


def get_section_tip(section: str, context_fn: Callable[[], dict]) -> dict[str, Any]:
    b = get_insights(context_fn)
    return {"section": section, "title": b["tip"]["title"], "tip": b["tip"]["tip"],
            "source": _src(b), "generated_at": b.get("generated_at")}


def get_ai_status(force: bool = False) -> dict[str, Any]:
    """No AI request unless you click the AI badge (force=True); otherwise uses the last known result."""
    global _last_error
    prov = _provider()
    if OpenAI is None:
        return {"live": False, "model": None, "reason": _explain(RuntimeError("SDK_MISSING"))}
    if not prov:
        return {"live": False, "model": None, "reason": _explain(ValueError("KEY_MISSING"))}
    if force:
        try:
            _complete([{"role": "user", "content": "Reply with OK."}], max_tokens=5, json_mode=False, temperature=0)
            return {"live": True, "model": prov["model"], "reason": f"Connected to {prov['name'].title()} ({prov['model']})."}
        except Exception as exc:
            _last_error = _explain(exc)
            return {"live": False, "model": prov["model"], "reason": _last_error}
    b = _load() or {}
    if b.get("source") == "rules" and b.get("reason"):
        return {"live": False, "model": prov["model"], "reason": b["reason"]}
    if _last_error:
        return {"live": False, "model": prov["model"], "reason": _last_error}
    return {"live": True, "model": prov["model"], "reason": f"Using {prov['name'].title()} ({prov['model']})."}


# ---------- Chat (only when you ask) ----------
def _local_answer(question: str, ctx: dict, reason: str) -> str:
    q = (question or "").lower()
    prods = ctx.get("products") or []
    lines: list[str] = []
    for p in prods:
        if p["name"].lower() in q:
            lines.append(f"{p['name']}: sells at {_rs(p['price'])}, costs {_rs(p['cost'])} to make ({p.get('margin_pct')}% margin), "
                         f"{p.get('units_sold', 0):,} sold, {p['stock']:,} in stock (about {p.get('days_of_cover')} days).")
    if prods and not lines:
        if any(w in q for w in ("best", "top", "most", "popular")):
            b = max(prods, key=lambda p: p.get("units_sold", 0))
            lines.append(f"Best seller: {b['name']} with {b.get('units_sold', 0):,} units sold.")
        if any(w in q for w in ("stock", "inventory", "run out", "reorder", "low")):
            low = [p for p in prods if p.get("days_of_cover", 999) < 7]
            lines.append("Reorder soon: " + ", ".join(f"{p['name']} ({p['days_of_cover']} days left)" for p in low) + "."
                         if low else "No product runs out within a week.")
        if any(w in q for w in ("profit", "margin", "earn", "money")):
            r = max(prods, key=lambda p: p.get("total_profit", 0))
            lines.append(f"Most profitable product: {r['name']} ({_rs(r.get('total_profit', 0))} profit so far).")
    if not lines:
        issues = _issues(ctx)
        lines.append(f"Most important right now: {issues[0]['title']}. {issues[0]['why']}" if issues
                     else "Stock, ads and ratings look healthy right now.")
    return "\n".join(lines) + f"\n\n(Quick answer from your numbers. {reason})"


def answer_chat_query(question: str, context_data: dict, history: list | None = None) -> str:
    question = (question or "").strip()
    if not question:
        return "Please type a question."
    messages = [{"role": "system", "content": SYSTEM + " Answer in 2 to 5 short sentences."}]
    for h in (history or [])[-4:]:
        messages.append({"role": "user" if h.get("me") else "assistant", "content": str(h.get("t", ""))[:400]})
    messages.append({"role": "user", "content": "Shop data:\n" + json.dumps(compact(context_data), ensure_ascii=False,
                     separators=(",", ":"), default=str) + f"\n\nQuestion: {question}"})
    try:
        return _complete(messages, max_tokens=350, json_mode=False) or _local_answer(question, context_data, "")
    except Exception as exc:
        reason = _explain(exc)
        log.warning("chat used a local answer: %s", reason)
        return _local_answer(question, context_data, reason)


# ---------- Review analyser (only when you press Analyse) ----------
_POS = ("good", "great", "love", "excellent", "perfect", "nice", "comfortable", "happy", "premium", "best", "fast")
_NEG = ("bad", "poor", "faded", "late", "damaged", "slow", "wrong", "small", "tight", "return", "worst", "broken", "not like")


def _local_feedback(reviews: list) -> dict:
    texts = [str(r.get("text", "")).lower() for r in reviews]
    pos = sum(1 for t in texts if any(w in t for w in _POS) and not any(w in t for w in _NEG))
    neg = sum(1 for t in texts if any(w in t for w in _NEG))
    n = max(len(texts), 1)
    neu = max(n - pos - neg, 0)
    themes = [w for w in ("quality", "fit", "size", "colour", "color", "delivery", "price", "fabric") if any(w in t for t in texts)][:4]
    return {"sentiment": {"positive": round(pos / n, 2), "neutral": round(neu / n, 2), "negative": round(neg / n, 2)},
            "summary": f"{pos} of {n} reviews sound positive and {neg} mention a problem.",
            "themes": [t.title() for t in themes] or ["General feedback"],
            "creative_feedback": ["Show the real colour and fit clearly in the ad."] if neg else ["Keep the current ad style."]}


def analyze_omnichannel_feedback(feedback_list: list) -> dict[str, Any]:
    if not feedback_list:
        return {**_local_feedback([]), "summary": "No reviews given.", "source": "fallback"}
    reviews = [{"platform": r.get("platform", ""), "text": str(r.get("text", ""))[:300]} for r in feedback_list[:40]]
    prompt = ('Read these customer reviews and reply with ONE JSON object: {"sentiment":{"positive":0-1,"neutral":0-1,'
              '"negative":0-1 (adding to 1)},"summary":"2 sentences","themes":["up to 4"],"creative_feedback":["up to 3 ad ideas"]}\n'
              + json.dumps(reviews, ensure_ascii=False))
    try:
        data = _parse_json(_complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                     max_tokens=500, json_mode=True, temperature=0.2))
        s = data.get("sentiment") or {}
        tot = sum(float(s.get(k) or 0) for k in ("positive", "neutral", "negative")) or 1
        return {"sentiment": {k: round(float(s.get(k) or 0) / tot, 2) for k in ("positive", "neutral", "negative")},
                "summary": str(data.get("summary") or ""), "themes": [str(x) for x in data.get("themes") or []][:4],
                "creative_feedback": [str(x) for x in data.get("creative_feedback") or []][:3], "source": "gemini"}
    except Exception as exc:
        log.warning("review analysis used local counting: %s", _explain(exc))
        return {**_local_feedback(reviews), "source": "fallback"}
