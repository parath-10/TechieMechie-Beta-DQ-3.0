"""NEXUS exports: CSV files and the one-page report for the brand owner.

CSV files open in Excel and Google Sheets. Cells that start with = + - @ get a leading quote, so a product
name like "=HYPERLINK(...)" can never run as a formula when the file is opened.
"""

import csv
import io
from typing import Any

# kind -> [(column heading, key in the row)]
COLUMNS: dict[str, list[tuple[str, str]]] = {
    "products": [("SKU", "sku"), ("Product", "name"), ("Category", "category"), ("Status", "status"),
                 ("Selling price (INR)", "price"), ("Cost (INR)", "cost"), ("Profit per unit (INR)", "profit_per_unit"),
                 ("Margin %", "margin_pct"), ("In stock", "stock"), ("Units sold", "units_sold"),
                 ("Sales per day", "daily_sales"), ("Days of stock left", "days_of_cover"),
                 ("Sales (INR)", "revenue"), ("Profit (INR)", "total_profit"), ("Stock value at cost (INR)", "stock_value")],
    "ads": [("Product", "product"), ("Platform", "platform"), ("Status", "status"), ("Daily budget (INR)", "daily_budget"),
            ("Views", "views"), ("Clicks", "clicks"), ("Click rate %", "click_rate_pct"), ("Orders", "orders"),
            ("Sales (INR)", "sales"), ("Ad spend (INR)", "spend"), ("Profit after ads (INR)", "profit_after_ads"),
            ("Sales per INR 1 of ads", "roas"), ("Avg rating", "avg_rating"), ("Reviews", "review_count")],
    "orders": [("Time", "time"), ("Order ID", "id"), ("SKU", "sku"), ("Product", "product"), ("Platform", "platform"),
               ("Quantity", "quantity"), ("Amount (INR)", "amount")],
    "decisions": [("Applied", "applied_at"), ("By", "by"), ("Where", "target"), ("Change", "change"),
                  ("Expected", "impact"), ("How sure", "confidence"), ("Result", "result_text"), ("Status", "result_status")],
}
KINDS = tuple(COLUMNS)


def _safe(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return "" if v is None else v


def make_csv(kind: str, rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for h, _ in COLUMNS[kind]])
    for r in rows:
        w.writerow([_safe(r.get(k)) for _, k in COLUMNS[kind]])
    return ("﻿" + buf.getvalue()).encode("utf-8")  # BOM so Excel shows ₹ and accents correctly


def decision_rows(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only approved changes, flattened for a spreadsheet."""
    out = []
    for a in actions:
        if a.get("status") != "done":
            continue
        res = a.get("result") or {}
        out.append({**a, "result_text": res.get("summary", ""), "result_status": res.get("status", "")})
    return out


def ad_rows(products: list[dict[str, Any]], ads_by_sku: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for p in products:
        for a in ads_by_sku.get(p["sku"], []):
            out.append({**{k: v for k, v in a.items() if k not in ("recent_reviews", "post")}, "product": p["name"],
                        "roas": round(a["sales"] / a["spend"], 2) if a.get("spend") else ""})
    return out


def build_report(brand: str, generated: str, products: list[dict[str, Any]], ads: list[dict[str, Any]],
                 orders: list[dict[str, Any]], actions: list[dict[str, Any]], simulated: bool) -> dict[str, Any]:
    """Everything the printable report shows, in one JSON object."""
    revenue = sum(p.get("revenue", 0) for p in products)
    profit = sum(p.get("total_profit", 0) for p in products)
    spend = sum(a.get("spend", 0) or 0 for a in ads)
    at_risk = [p for p in products if p.get("status") == "active" and p.get("days_of_cover", 999) < 14]
    return {
        "brand": brand, "generated": generated, "simulated": simulated,
        "kpis": {"revenue": round(revenue), "profit": round(profit), "ad_spend": round(spend),
                 "profit_after_ads": round(profit - spend), "units": sum(p.get("units_sold", 0) for p in products),
                 "products": len(products), "at_risk": len(at_risk)},
        "products": products, "ads": ads, "orders": orders[:25],
        "todo": [a for a in actions if a.get("status") == "pending"][:6],
        "decisions": [a for a in actions if a.get("status") == "done"],
    }
