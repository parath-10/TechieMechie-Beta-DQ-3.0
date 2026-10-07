"""NEXUS platform connections, stored in Supabase.

Stores the login details each platform's API needs (IDs, access tokens, API keys) in the
`connections` table.
- Secrets are ENCRYPTED with Fernet (AES) before they leave this server; the database only
  ever holds ciphertext. The key comes from the NEXUS_SECRET_KEY environment variable.
- Secrets are never sent back to the browser; only the last 4 characters are shown.

Modes
- demo (default): details are checked for the right shape and saved; ad numbers and orders are simulated.
- live: set LIVE_PLATFORM_APIS=1. "Test connection" then makes a real read-only API call
  for Facebook, Instagram, YouTube and Shopify. Posting ads and pulling orders through the real APIs
  still need approved developer apps on each platform; the hooks for that are marked TODO below.
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from db import client, now_local, one, select_all

TABLE = "connections"
LIVE = os.getenv("LIVE_PLATFORM_APIS", "0").strip() == "1"

# What each platform's API needs. secret=True fields are encrypted and never shown again.
PLATFORMS: dict[str, dict[str, Any]] = {
    "Instagram": {"kind": "Ads + posts", "api": "Meta Graph / Marketing API", "fields": [
        {"key": "account_id", "label": "Instagram business account ID", "pattern": r"^\d{6,}$", "demo": "17841400000000001"},
        {"key": "ad_account_id", "label": "Meta ad account ID (act_...)", "pattern": r"^act_\d+$", "demo": "act_1234567890"},
        {"key": "access_token", "label": "Meta access token", "secret": True, "demo": "EAAdemo0000instagram0000token"}]},
    "Facebook": {"kind": "Ads + page posts", "api": "Meta Graph / Marketing API", "fields": [
        {"key": "page_id", "label": "Facebook page ID", "pattern": r"^\d{6,}$", "demo": "102030405060708"},
        {"key": "ad_account_id", "label": "Meta ad account ID (act_...)", "pattern": r"^act_\d+$", "demo": "act_1234567890"},
        {"key": "access_token", "label": "Page access token", "secret": True, "demo": "EAAdemo0000facebook0000token"}]},
    "YouTube": {"kind": "Video ads", "api": "YouTube Data API + Google Ads API", "fields": [
        {"key": "channel_id", "label": "YouTube channel ID (UC...)", "pattern": r"^UC[\w-]{10,}$", "demo": "UCdemo1234567890abcdef"},
        {"key": "customer_id", "label": "Google Ads customer ID", "pattern": r"^\d{3}-?\d{3}-?\d{4}$", "demo": "123-456-7890"},
        {"key": "access_token", "label": "Google OAuth access token", "secret": True, "demo": "ya29.demo0000youtube0000token"}]},
    "Google": {"kind": "Search + shopping ads", "api": "Google Ads API", "fields": [
        {"key": "customer_id", "label": "Google Ads customer ID", "pattern": r"^\d{3}-?\d{3}-?\d{4}$", "demo": "123-456-7890"},
        {"key": "developer_token", "label": "Developer token", "secret": True, "demo": "demo0000devtoken"},
        {"key": "refresh_token", "label": "OAuth refresh token", "secret": True, "demo": "1//demo0000refresh"}]},
    "TikTok": {"kind": "Video ads", "api": "TikTok Marketing API", "fields": [
        {"key": "advertiser_id", "label": "Advertiser ID", "pattern": r"^\d{6,}$", "demo": "7000000000000000001"},
        {"key": "access_token", "label": "Access token", "secret": True, "demo": "demo0000tiktok0000token"}]},
    "Shopify": {"kind": "Your website store (orders)", "api": "Shopify Admin API", "channel": "Website", "fields": [
        {"key": "shop_domain", "label": "Store address (yourstore.myshopify.com)", "pattern": r"^[\w-]+\.myshopify\.com$", "demo": "lumen-co-demo.myshopify.com"},
        {"key": "access_token", "label": "Admin API access token (shpat_...)", "secret": True, "demo": "shpat_demo0000000000000000"}]},
    "Amazon": {"kind": "Marketplace orders + sponsored ads", "api": "Selling Partner API + Amazon Ads API", "fields": [
        {"key": "seller_id", "label": "Seller ID", "pattern": r"^[A-Z0-9]{8,20}$", "demo": "A1DEMOSELLER01"},
        {"key": "marketplace_id", "label": "Marketplace ID (India: A21TJRUUN4KGV)", "pattern": r"^[A-Z0-9]{10,16}$", "demo": "A21TJRUUN4KGV"},
        {"key": "client_id", "label": "LWA client ID", "demo": "amzn1.application-oa2-client.demo"},
        {"key": "client_secret", "label": "LWA client secret", "secret": True, "demo": "demo0000amazon0000secret"},
        {"key": "refresh_token", "label": "Refresh token", "secret": True, "demo": "Atzr|demo0000refresh"}]},
    "Flipkart": {"kind": "Marketplace orders + ads", "api": "Flipkart Seller API", "fields": [
        {"key": "app_id", "label": "Application ID", "demo": "demo-app-1234"},
        {"key": "app_secret", "label": "Application secret", "secret": True, "demo": "demo0000flipkart0000secret"}]},
}

# Which connection a sales channel / ad platform depends on
CHANNEL_TO_PLATFORM = {"Website": "Shopify"}


# ---------- Encryption ----------
def _fernet() -> Fernet:
    key = os.getenv("NEXUS_SECRET_KEY", "").strip()
    if not key:
        raise RuntimeError("NEXUS_SECRET_KEY is not set. Make one with: "
                           "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        raise RuntimeError("NEXUS_SECRET_KEY is not a valid Fernet key (it must be 44 characters ending in '=').") from exc


# ---------- Reading rows ----------
def _rows() -> dict[str, dict[str, Any]]:
    return {r["platform"]: r for r in select_all(TABLE, order="platform")}


def _row(platform: str) -> dict[str, Any] | None:
    return one(TABLE, platform=platform)


def _secret(platform: str, field: str) -> str:
    """Decrypt one secret, for use by the platform API code only (never returned to the browser)."""
    row = _row(platform)
    enc = ((row or {}).get("secrets_enc") or {}).get(field)
    if not enc:
        return ""
    try:
        return _fernet().decrypt(enc.encode()).decode()
    except InvalidToken:
        return ""


# ---------- Public API ----------
def connected_set() -> set[str]:
    """Names of connected platforms. Read once per request and pass it around to avoid repeat queries."""
    return {r["platform"] for r in select_all(TABLE, columns="platform", order="platform")}


def is_connected(name: str, conn: set[str] | None = None) -> bool:
    """Accepts an ad platform ('Instagram') or a sales channel ('Website' -> Shopify)."""
    conn = connected_set() if conn is None else conn
    return CHANNEL_TO_PLATFORM.get(name, name) in conn


def connected_names(conn: set[str] | None = None) -> list[str]:
    conn = connected_set() if conn is None else conn
    return [p for p in PLATFORMS if p in conn]


def list_connections() -> list[dict[str, Any]]:
    data = _rows()
    out = []
    for name, spec in PLATFORMS.items():
        c = data.get(name)
        out.append({
            "platform": name, "kind": spec["kind"], "api": spec["api"],
            "fields": [{k: f[k] for k in ("key", "label") if k in f} | {"secret": bool(f.get("secret"))} for f in spec["fields"]],
            "connected": bool(c),
            "account_name": c.get("account_name") if c else None,
            "mode": c.get("mode") if c else None,
            "connected_at": c.get("connected_at") if c else None,
            "public": (c.get("public_ids") or {}) if c else {},
            "masked": {k: "••••" + v[-4:] for k, v in (c.get("masked") or {}).items()} if c else {},
            "last_sync": c.get("last_sync") if c else None,
            "last_test": c.get("last_test") if c else None,
        })
    return out


def list_connections_map() -> dict[str, dict[str, Any]]:
    return {x["platform"]: x for x in list_connections() if x["connected"]}


def connect(platform: str, account_name: str, values: dict[str, str]) -> dict[str, Any]:
    """Check and save one platform's details. Returns {"ok": bool, "message": str}."""
    spec = PLATFORMS.get(platform)
    if not spec:
        return {"ok": False, "message": "Unknown platform."}
    values = {k: (v or "").strip() for k, v in values.items()}
    missing = [f["label"] for f in spec["fields"] if not values.get(f["key"])]
    if missing:
        return {"ok": False, "message": "Please fill in: " + ", ".join(missing)}
    for f in spec["fields"]:
        if f.get("pattern") and not re.match(f["pattern"], values[f["key"]]):
            return {"ok": False, "message": f"“{f['label']}” does not look right. Check it and try again."}
    fer = _fernet()
    is_demo = all(values[f["key"]] == f.get("demo") for f in spec["fields"] if f.get("secret"))
    mode = "live" if LIVE and not is_demo else "demo"
    row = {
        "platform": platform,
        "account_name": (account_name or "").strip() or platform + " account",
        "public_ids": {f["key"]: values[f["key"]] for f in spec["fields"] if not f.get("secret")},
        "secrets_enc": {f["key"]: fer.encrypt(values[f["key"]].encode()).decode() for f in spec["fields"] if f.get("secret")},
        "masked": {f["key"]: values[f["key"]][-4:] for f in spec["fields"] if f.get("secret")},
        "mode": mode,
        "connected_at": now_local().strftime("%d %b %Y, %I:%M %p"),
        "last_sync": None,
        "last_test": None,
    }
    client().table(TABLE).upsert(row, on_conflict="platform").execute()
    result = test(platform)
    if LIVE and not is_demo and not result["ok"]:
        client().table(TABLE).delete().eq("platform", platform).execute()
        return {"ok": False, "message": "Saved details were rejected by " + platform + ": " + result["message"]}
    return {"ok": True, "message": f"{platform} connected" + (" (live)." if mode == "live" else
                                                            " in demo mode. Ad numbers and orders are simulated.")}


def disconnect(platform: str) -> bool:
    """Delete the row, which deletes the encrypted keys with it."""
    deleted = client().table(TABLE).delete().eq("platform", platform).execute().data or []
    return bool(deleted)


def mark_sync(platform: str) -> None:
    mark_sync_many({platform})


def mark_sync_many(platforms: set[str]) -> None:
    names = sorted({CHANNEL_TO_PLATFORM.get(p, p) for p in platforms})
    if names:
        stamp = now_local().strftime("%d %b, %I:%M:%S %p")
        client().table(TABLE).update({"last_sync": stamp}).in_("platform", names).execute()


def _http_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def test(platform: str) -> dict[str, Any]:
    """Check that a saved connection works. Live mode makes a real read-only call where supported."""
    c = _row(platform)
    if not c:
        return {"ok": False, "message": "Not connected."}
    if c["mode"] != "live":
        res = {"ok": True, "message": "Demo connection is working. Numbers are simulated."}
    else:
        try:
            if platform in ("Facebook", "Instagram"):
                me = _http_get("https://graph.facebook.com/v19.0/me?fields=id,name&access_token=" + _secret(platform, "access_token"))
                res = {"ok": True, "message": f"Connected to Meta as {me.get('name', me.get('id'))}."}
            elif platform == "YouTube":
                ch = _http_get("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                               {"Authorization": "Bearer " + _secret(platform, "access_token")})
                title = (ch.get("items") or [{}])[0].get("snippet", {}).get("title", "your channel")
                res = {"ok": True, "message": f"Connected to YouTube channel {title}."}
            elif platform == "Shopify":
                shop = _http_get(f"https://{c['public_ids']['shop_domain']}/admin/api/2024-07/shop.json",
                                 {"X-Shopify-Access-Token": _secret(platform, "access_token")})
                res = {"ok": True, "message": f"Connected to Shopify store {shop.get('shop', {}).get('name', '')}."}
            else:
                # TODO: Google Ads, TikTok, Amazon SP-API and Flipkart need signed requests from an approved app.
                res = {"ok": True, "message": "Details saved. A live check for this platform is not built yet."}
        except urllib.error.HTTPError as e:
            res = {"ok": False, "message": f"The platform said no (HTTP {e.code}). Check the token and its permissions."}
        except Exception as e:  # network, timeout, bad JSON
            res = {"ok": False, "message": f"Could not reach the platform ({type(e).__name__})."}
    client().table(TABLE).update({"last_test": ("OK: " if res["ok"] else "Failed: ") + res["message"]}) \
        .eq("platform", platform).execute()
    return res


def demo_values(platform: str) -> dict[str, str]:
    return {f["key"]: f["demo"] for f in PLATFORMS[platform]["fields"]}


# TODO (live mode): when each platform's developer app is approved, add here
#   post_ad(platform, creative, budget) -> real ad/post ID
#   fetch_ad_stats(platform, post_id)   -> views, clicks, spend
#   fetch_orders(platform, since)       -> new orders
# and call them from ads.py instead of the simulated numbers.
