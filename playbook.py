"""NEXUS playbook: suggestions, approve-and-apply, learning from results, and the "what if" curve.

The loop
1. SUGGEST  Suggestions are worked out from the live numbers (stock cover, each ad's budget and profit),
            using the same demand curve as the order simulation (ads.expected_units).
2. APPROVE  Approving really changes the ad (pause or new daily budget) or adds stock.
            The state before the change is saved with Nexus' prediction.
3. MEASURE  As orders sync, Nexus compares what happened since the change with what it predicted
            (a "day" is measured in simulated days, so the demo button counts too).
4. LEARN    Verified results set a calibration number. When past predictions were off, the
            "How sure" figure on new suggestions drops, and when they were right it stays high.

Everything is stored in the existing app_state table (keys "act:..."), so no database change is needed.
"""

import logging
import math
import os
import time
from typing import Any

import ads
import connectors
from db import client, now_local

log = logging.getLogger("nexus.playbook")

LEARN_DAYS = max(0.05, float(os.getenv("LEARN_WINDOW_DAYS", "1") or 1))  # simulated days to watch before judging
MIN_GAIN = 150  # ignore changes worth less than this many rupees a day
RESTOCK_COVER_DAYS, RESTOCK_TARGET_DAYS, STOCKOUT_DAYS = 14, 21, 3


def rs(n: float) -> str:
    return "₹" + format(int(round(abs(n))), ",")


def _signed(n: float) -> str:
    return ("+" if n >= 0 else "−") + rs(n)


def profit_per_day(p: dict[str, Any], platform: str, ad: dict[str, Any], budget: float) -> float:
    """Expected profit per day from one ad at a daily budget: orders x profit per unit, minus the budget."""
    return ads.expected_units(p, platform, ad, budget) * (p["price"] - p["cost"]) - budget


# =====================================================================
# 1. Suggestions
# =====================================================================
def _calibration() -> float:
    """0..1: how accurate past predictions were (1 = no history yet or all spot on)."""
    vals = [r["final"]["accuracy"] for r in _records(10) if (r.get("final") or {}).get("accuracy") is not None]
    return sum(vals) / len(vals) if vals else 1.0


def build_suggestions(products: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    products = ads.get_products() if products is None else products
    if not products:
        return []
    rates = ads.selling_rates(products)
    by_sku = {p["sku"]: p for p in products}
    rows = ads._ads_rows([p["sku"] for p in products], status="active")
    conn = connectors.connected_set()
    cal = 0.85 + 0.15 * _calibration()  # past accuracy nudges every "How sure"
    out: list[dict[str, Any]] = []

    def cover(p: dict[str, Any]) -> float:
        r = rates.get(p["sku"], 0)
        return p["stock"] / r if r > 0 else 999.0

    for p in products:  # --- running low: reorder
        r, c = rates.get(p["sku"], 0), cover(p)
        if p["status"] == "active" and r > 0 and c <= RESTOCK_COVER_DAYS:
            qty = math.ceil(r * RESTOCK_TARGET_DAYS - p["stock"])
            if qty > 0:
                at_risk = r * (p["price"] - p["cost"])
                out.append({
                    "id": f"restock:{p['sku']}", "type": "restock", "sku": p["sku"], "platform": None,
                    "target": f"Stock: {p['name']}", "change": f"Reorder {qty} units",
                    "impact": f"Lasts {round(c)} days now, about {RESTOCK_TARGET_DAYS} days after. Protects {rs(at_risk)} a day of profit",
                    "confidence": round(min(0.97, 0.9 * cal + 0.07), 2), "gain": at_risk * 2,
                    "params": {"qty": qty, "rate": r, "cover_before": round(c, 1), "unit_margin": p["price"] - p["cost"]}})

    for a in rows:  # --- each running ad
        p = by_sku.get(a["sku"])
        if not p or p["status"] != "active" or not connectors.is_connected(a["platform"], conn):
            continue
        plat, b0, c = a["platform"], float(a["daily_budget"]), cover(p)
        base = profit_per_day(p, plat, a, b0)
        target = f"{plat}: {p['name']}"
        seeded = bool(a.get("seed_rate"))
        if c <= STOCKOUT_DAYS:  # about to run out: stop paying for views nobody can buy
            out.append({
                "id": f"stockout:{p['sku']}:{plat}", "type": "pause", "sku": p["sku"], "platform": plat,
                "target": target, "change": f"Pause until restocked ({rs(b0)} a day)",
                "impact": f"Only {round(c, 1)} days of stock left. Saves {rs(b0)} a day of ads on an empty shelf",
                "confidence": round(min(0.97, 0.9 * cal + 0.06), 2), "gain": b0 * 1.5,
                "params": {"mode": "stockout", "budget_before": b0, "baseline": -b0, "predicted": b0,
                           "unit_margin": p["price"] - p["cost"]}})
            continue
        factors = [0.0] + [0.5 + 0.05 * i for i in range(21 if c >= 7 else 11)]  # no pushing low-stock products
        best_b, best = b0, base
        for f in factors:
            b = 0.0 if f == 0 else max(50.0, round(b0 * f / 50) * 50)
            v = profit_per_day(p, plat, a, b) if b > 0 else 0.0
            if v > best + 1e-6:
                best_b, best = b, v
        gain = best - base
        if best_b == b0 or gain < max(MIN_GAIN, 0.05 * abs(base)):
            continue
        conf = min(0.95, 0.7 + 0.22 * min(1, gain / 3000)) * cal - (0 if seeded else 0.1)
        common = {"sku": p["sku"], "platform": plat, "target": target, "confidence": round(max(0.5, conf), 2),
                  "gain": gain}
        params = {"budget_before": b0, "baseline": base, "predicted": gain, "unit_margin": p["price"] - p["cost"]}
        if best_b == 0:
            out.append({**common, "id": f"pause:{p['sku']}:{plat}", "type": "pause", "change": f"Stop ad ({rs(b0)} a day)",
                        "impact": f"Loses money at this budget. Saves about {rs(gain)} a day",
                        "params": {**params, "mode": "loss"}})
        else:
            up = best_b > b0
            out.append({**common, "id": f"budget:{p['sku']}:{plat}", "type": "realloc",
                        "change": f"{'Raise' if up else 'Lower'} budget {rs(b0)} → {rs(best_b)} a day",
                        "impact": f"About {rs(gain)} more profit a day after ad costs",
                        "params": {**params, "mode": "budget", "budget_after": best_b}})
    out.sort(key=lambda s: -s["gain"])
    return out[:8]


def _public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in s.items() if k not in ("params", "gain")} | {"status": "pending"}


# =====================================================================
# 2. Approve and apply
# =====================================================================
class ActionError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _ad_totals(sku: str, platform: str) -> dict[str, float]:
    return ads._totals_map([sku]).get((sku, platform), dict(ads._ZERO))


def approve(action_id: str, role: str = "owner") -> dict[str, Any]:
    sug = next((s for s in build_suggestions() if s["id"] == action_id), None)
    if not sug:
        raise ActionError(404, "That suggestion is out of date or was already applied. Press “Get new suggestions”.")
    p = ads.get_product(sug["sku"])
    if not p:
        raise ActionError(404, "Product not found.")
    prm, plat = sug["params"], sug["platform"]
    rec = {k: sug[k] for k in ("id", "type", "target", "change", "impact", "confidence", "sku", "platform")}
    rec.update({"params": prm, "by": role, "applied_at": now_local().strftime("%d %b, %I:%M %p"),
                "applied_sim": ads.sim_days(), "applied_bonus": ads.bonus_days(), "stock_before": p["stock"]})
    if sug["type"] == "restock":
        new = ads.add_stock(p["sku"], int(prm["qty"]))
        if not new:
            raise ActionError(404, "Product not found.")
        message = f"Added {prm['qty']} units of {p['name']}. Nexus will check that stock now covers about {RESTOCK_TARGET_DAYS} days."
    else:
        rec["base_totals"] = {k: float(_ad_totals(p["sku"], plat).get(k, 0)) for k in ("units", "spend", "revenue")}
        if prm["mode"] == "budget":
            ad = ads.change_ad(p, plat, "active", float(prm["budget_after"]))
            message = f"{plat} budget for {p['name']} is now {rs(prm['budget_after'])} a day."
        else:
            ad = ads.change_ad(p, plat, "paused", None)
            message = f"{plat} ad for {p['name']} is paused."
        if ad is None:
            raise ActionError(404, f"{p['name']} is not advertised on {plat}.")
    if sug["type"] == "restock":
        message += " Nexus will check the result."
    else:
        message += f" Nexus predicts {_signed(prm['predicted'])} a day and will report the real result here."
    client().table("app_state").upsert({"key": f"act:{int(time.time() * 1000)}:{action_id}", "value": rec}).execute()
    return {"status": "executed", "action_id": action_id, "message": message}


# =====================================================================
# 3. Measure and 4. Learn
# =====================================================================
def _records(limit: int = 8) -> list[dict[str, Any]]:
    rows = client().table("app_state").select("key,value,updated_at").like("key", "act:%") \
        .order("updated_at", desc=True).limit(limit).execute().data or []
    return [{**r["value"], "_key": r["key"]} for r in rows]


def _accuracy(predicted: float, actual: float) -> float:
    return round(max(0.0, 1 - abs(actual - predicted) / max(abs(predicted), 1.0)), 2)


def measure(rec: dict[str, Any], products: dict[str, dict[str, Any]], rates: dict[str, float],
            totals: dict[tuple[str, str], dict[str, float]], now_sim: float, now_bonus: float) -> dict[str, Any]:
    prm = rec["params"]
    if rec["type"] == "restock":
        p = products.get(rec["sku"])
        r = rates.get(rec["sku"], 0)
        cover_now = round(p["stock"] / r, 1) if p and r > 0 else None
        if cover_now is None:
            return {"status": "verified", "summary": "Stock added.", "accuracy": None}
        acc = _accuracy(RESTOCK_TARGET_DAYS, cover_now)
        return {"status": "verified", "accuracy": acc, "predicted": RESTOCK_TARGET_DAYS, "actual": cover_now,
                "summary": f"Stock now lasts {cover_now:g} days (was {prm['cover_before']:g}, aimed for {RESTOCK_TARGET_DAYS}). Accuracy {round(acc * 100)}%."}
    elapsed = (now_sim - rec["applied_sim"]) + (now_bonus - rec["applied_bonus"])
    progress = max(0.0, min(1.0, elapsed / LEARN_DAYS))
    pred = float(prm["predicted"])
    if elapsed < 0.02:
        return {"status": "learning", "progress": round(progress, 2), "predicted": round(pred),
                "summary": f"Predicted {_signed(pred)} a day. Measuring… (needs about {LEARN_DAYS:g} day of orders; the “Simulate 1 day” button speeds this up)."}
    now_t, base_t = totals.get((rec["sku"], rec["platform"]), {}), rec.get("base_totals", {})
    d_units = float(now_t.get("units", 0)) - base_t.get("units", 0)
    d_spend = float(now_t.get("spend", 0)) - base_t.get("spend", 0)
    realized = (d_units * prm["unit_margin"] - d_spend) / elapsed
    actual = realized - prm["baseline"]
    out = {"predicted": round(pred), "actual": round(actual), "orders": int(d_units), "days": round(elapsed, 2),
           "progress": round(progress, 2)}
    if progress < 1:
        out.update(status="learning", summary=f"Predicted {_signed(pred)} a day. So far {_signed(actual)} a day from {int(d_units)} orders "
                                              f"({round(progress * 100)}% of the check window).")
    else:
        acc = _accuracy(pred, actual)
        out.update(status="verified", accuracy=acc,
                   summary=f"Predicted {_signed(pred)} a day, actual {_signed(actual)} a day over {elapsed:.1f} days "
                           f"({int(d_units)} orders). Accuracy {round(acc * 100)}%.")
    return out


def history(limit: int = 6) -> list[dict[str, Any]]:
    recs = _records(limit)
    if not recs:
        return []
    skus = sorted({r["sku"] for r in recs})
    products = {p["sku"]: p for p in ads.get_products() if p["sku"] in skus}
    rates = ads.selling_rates(list(products.values())) if products else {}
    totals = ads._totals_map(skus)
    sim, bonus = ads.sim_days(), ads.bonus_days()
    out = []
    for r in recs:
        res = r.get("final") or measure(r, products, rates, totals, sim, bonus)
        if not r.get("final") and res.get("status") == "verified":  # remember it, so Nexus learns from it
            try:
                client().table("app_state").update({"value": {**{k: v for k, v in r.items() if k != "_key"}, "final": res}}) \
                    .eq("key", r["_key"]).execute()
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not save a verified result: %s", exc)
        out.append({"id": "done:" + r["_key"], "type": r["type"], "target": r["target"], "change": r["change"],
                    "impact": r["impact"], "confidence": r["confidence"], "status": "done",
                    "applied_at": r.get("applied_at", ""), "by": r.get("by", ""), "result": res})
    return out


def actions_list() -> list[dict[str, Any]]:
    """Pending suggestions first, then what was already approved with how it turned out.
    One half failing must not hide the other, so each is read on its own and the problem is logged."""
    try:
        todo = [_public(s) for s in build_suggestions()]
    except Exception:  # noqa: BLE001
        log.exception("Could not work out new suggestions")
        todo = []
    try:
        done = history()
    except Exception:  # noqa: BLE001
        log.exception("Could not read past decisions")
        done = []
    return todo + done


# =====================================================================
# What if: profit at any daily budget for one ad
# =====================================================================
def whatif(sku: str, platform: str) -> dict[str, Any] | None:
    p, ad = ads.get_product(sku), ads._get_ad(sku, platform)
    if not p or not ad:
        return None
    b0, margin = float(ad["daily_budget"]), p["price"] - p["cost"]
    top = max(b0 * 4, 500.0)
    curve = []
    for i in range(41):
        b = round(top * i / 40)
        u = ads.expected_units(p, platform, ad, b)
        curve.append({"b": b, "units": round(u, 3), "profit": round(u * margin - b)})
    best = max(((top * i / 400, profit_per_day(p, platform, ad, top * i / 400)) for i in range(401)), key=lambda t: t[1])
    rate = ads.selling_rates([p]).get(sku, 0)
    return {"sku": sku, "platform": platform, "name": p["name"], "status": ad["status"], "budget": b0,
            "price": p["price"], "cost": p["cost"], "unit_margin": margin, "stock": p["stock"],
            "units_now": round(ads.expected_units(p, platform, ad), 3), "product_rate": rate,
            "max_budget": round(top), "best_budget": round(best[0] / 50) * 50, "best_profit": round(best[1]),
            "curve": curve}
