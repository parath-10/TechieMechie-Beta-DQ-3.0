"""NEXUS login: checked on the server, remembered with a signed cookie.

- Passwords are compared on the server (never shipped to the browser).
- The session is a signed cookie (HMAC-SHA256), so it works on any number of server copies
  and survives restarts without storing anything in the database. It cannot be read by
  page scripts (HttpOnly) and cannot be forged without the secret key.
- Two kinds of account:
    owner : full access. Set NEXUS_LOGIN_EMAIL and NEXUS_LOGIN_PASSWORD.
    demo  : look-around access for judges and visitors (demo@nexus.local / Nexus123!).
            It can view everything but cannot change products, ads, accounts or settings.
            Turn it off with DEMO_LOGIN=0.

Environment variables
  NEXUS_LOGIN_EMAIL     owner's email
  NEXUS_LOGIN_PASSWORD  owner's password (choose a long one)
  DEMO_LOGIN            1 = demo account allowed (default), 0 = owner only
  SESSION_SECRET        optional: key that signs sessions (defaults to NEXUS_SECRET_KEY)
  SESSION_DAYS          optional: how long a login lasts, default 7
"""

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from typing import Any

COOKIE = "nexus_session"
DEMO_EMAIL = "demo@nexus.local"
DEMO_PASSWORD = "Nexus123!"
SESSION_DAYS = float(os.getenv("SESSION_DAYS", "7") or 7)

# Requests the read-only demo account may still make (they change nothing important)
DEMO_ALLOWED_WRITES = {"/api/orders/sync", "/api/chat", "/api/analyze-feedback", "/api/logout"}


def demo_enabled() -> bool:
    return os.getenv("DEMO_LOGIN", "1").strip() != "0"


def owner_configured() -> bool:
    return bool(os.getenv("NEXUS_LOGIN_EMAIL", "").strip() and os.getenv("NEXUS_LOGIN_PASSWORD", ""))


def _secret() -> bytes:
    key = os.getenv("SESSION_SECRET", "").strip() or os.getenv("NEXUS_SECRET_KEY", "").strip()
    if not key:
        raise RuntimeError("Set SESSION_SECRET or NEXUS_SECRET_KEY so logins can be signed.")
    return hashlib.sha256(("nexus-session:" + key).encode()).digest()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(email: str, role: str) -> str:
    payload = _b64(json.dumps({"e": email, "r": role, "x": int(time.time() + SESSION_DAYS * 86400)},
                              separators=(",", ":")).encode())
    sig = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return payload + "." + sig


def read_token(token: str | None) -> dict[str, Any] | None:
    """The logged-in user, or None if the cookie is missing, changed or expired."""
    if not token or token.count(".") != 1:
        return None
    payload, sig = token.split(".")
    try:
        good = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(_unb64(payload))
    except Exception:
        return None
    if data.get("x", 0) < time.time():
        return None
    if data.get("r") == "demo" and not demo_enabled():
        return None  # demo switched off: old demo sessions stop working too
    return {"email": data.get("e"), "role": data.get("r")}


def check_login(email: str, password: str) -> str | None:
    """'owner' or 'demo' if the details are right, otherwise None. Comparisons take constant time."""
    email = (email or "").strip().lower()
    password = password or ""
    owner_email = os.getenv("NEXUS_LOGIN_EMAIL", "").strip().lower()
    owner_pass = os.getenv("NEXUS_LOGIN_PASSWORD", "")
    if owner_email and owner_pass:
        ok_e = hmac.compare_digest(email.encode(), owner_email.encode())
        ok_p = hmac.compare_digest(password.encode(), owner_pass.encode())
        if ok_e and ok_p:
            return "owner"
    if demo_enabled():
        ok_e = hmac.compare_digest(email.encode(), DEMO_EMAIL.encode())
        ok_p = hmac.compare_digest(password.encode(), DEMO_PASSWORD.encode())
        if ok_e and ok_p:
            return "demo"
    return None


# ---------- Slow down password guessing (per server copy; resets on restart) ----------
_attempts: dict[str, list[float]] = {}
_lock = threading.Lock()
MAX_TRIES, WINDOW = 8, 600  # 8 wrong tries per 10 minutes per address


def too_many_tries(ip: str) -> bool:
    now = time.time()
    with _lock:
        tries = [t for t in _attempts.get(ip, []) if now - t < WINDOW]
        _attempts[ip] = tries
        return len(tries) >= MAX_TRIES


def note_failure(ip: str) -> None:
    with _lock:
        _attempts.setdefault(ip, []).append(time.time())


def clear_failures(ip: str) -> None:
    with _lock:
        _attempts.pop(ip, None)
