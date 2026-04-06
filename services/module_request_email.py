"""
services/module_request_email.py
--------------------------------
SMTP email notifications + signed action links for module approval.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Tuple
from urllib.parse import urlencode


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _get_token_secret() -> str:
    return (
        os.getenv("MODULE_APPROVAL_TOKEN_SECRET")
        or os.getenv("SUPERADMIN_SECRET")
        or "changeme-superadmin"
    )


def make_action_token(
    request_id: str,
    tenant_id: str,
    module_name: str,
    requested_state: bool,
    action: str,
) -> str:
    ttl_minutes = int(os.getenv("MODULE_APPROVAL_TOKEN_TTL_MINUTES", "60"))
    payload = {
        "rid": request_id,
        "tid": tenant_id,
        "mod": module_name,
        "state": bool(requested_state),
        "action": action,  # approved | rejected
        "exp": int(time.time()) + (ttl_minutes * 60),
        "nonce": secrets.token_urlsafe(12),
    }
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    sig = hmac.new(
        _get_token_secret().encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{payload_b64}.{sig_b64}"


def verify_action_token(token: str) -> Dict:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("Malformed token") from exc

    expected_sig = hmac.new(
        _get_token_secret().encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    got_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, got_sig):
        raise ValueError("Invalid token signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid token payload") from exc

    if payload.get("action") not in ("approved", "rejected"):
        raise ValueError("Invalid token action")
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Token expired")
    return payload


def _smtp_settings() -> Tuple[str, int, str, str, bool, str]:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() in ("1", "true", "yes")
    from_email = (os.getenv("SMTP_FROM_EMAIL") or username).strip()
    return host, port, username, password, use_tls, from_email


def _require_mail_config() -> Tuple[str, int, str, str, bool, str, str, str]:
    host, port, username, password, use_tls, from_email = _smtp_settings()
    superadmin_email = os.getenv("SUPERADMIN_EMAIL", "").strip()
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not host or not username or not password or not from_email:
        raise ValueError("SMTP config missing (host/username/password/from_email)")
    if not superadmin_email:
        raise ValueError("SUPERADMIN_EMAIL missing")
    if not public_base_url:
        raise ValueError("PUBLIC_BASE_URL missing")
    return host, port, username, password, use_tls, from_email, superadmin_email, public_base_url


def send_module_request_email(
    request_id: str,
    tenant_id: str,
    business_name: str,
    admin_email: str,
    module_name: str,
    requested_state: bool,
    note: str = "",
) -> Dict[str, str]:
    (
        host,
        port,
        username,
        password,
        use_tls,
        from_email,
        superadmin_email,
        public_base_url,
    ) = _require_mail_config()

    approve_token = make_action_token(
        request_id=request_id,
        tenant_id=tenant_id,
        module_name=module_name,
        requested_state=requested_state,
        action="approved",
    )
    reject_token = make_action_token(
        request_id=request_id,
        tenant_id=tenant_id,
        module_name=module_name,
        requested_state=requested_state,
        action="rejected",
    )

    approve_url = f"{public_base_url}/email/module-requests/decision?{urlencode({'token': approve_token})}"
    reject_url = f"{public_base_url}/email/module-requests/decision?{urlencode({'token': reject_token})}"
    requested_action = "ENABLE" if requested_state else "DISABLE"
    note_html = (
        f"<p><b>Admin Note:</b> {note}</p>" if note else "<p><b>Admin Note:</b> (none)</p>"
    )

    subject = f"[SamaySetu] Module Request: {requested_action} {module_name}"
    html = f"""
    <html>
      <body style="font-family:Arial,sans-serif;line-height:1.5;">
        <h3>Module Access Request</h3>
        <p><b>Business:</b> {business_name or tenant_id}</p>
        <p><b>Tenant ID:</b> {tenant_id}</p>
        <p><b>Admin:</b> {admin_email}</p>
        <p><b>Module:</b> {module_name}</p>
        <p><b>Requested Action:</b> {requested_action}</p>
        {note_html}
        <p style="margin-top:20px;">
          <a href="{approve_url}" style="background:#16a34a;color:#fff;padding:10px 16px;text-decoration:none;border-radius:8px;margin-right:8px;">Approve</a>
          <a href="{reject_url}" style="background:#dc2626;color:#fff;padding:10px 16px;text-decoration:none;border-radius:8px;">Reject</a>
        </p>
        <p style="color:#666;font-size:12px;">These links expire automatically for security.</p>
      </body>
    </html>
    """
    plain = (
        f"Module request\n"
        f"Business: {business_name or tenant_id}\n"
        f"Tenant ID: {tenant_id}\n"
        f"Admin: {admin_email}\n"
        f"Module: {module_name}\n"
        f"Requested Action: {requested_action}\n"
        f"Note: {note or '(none)'}\n\n"
        f"Approve: {approve_url}\n"
        f"Reject: {reject_url}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = superadmin_email
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(host, port, timeout=20) as server:
        if use_tls:
            server.starttls()
        server.login(username, password)
        server.sendmail(from_email, [superadmin_email], msg.as_string())

    return {
        "to": superadmin_email,
        "approve_url": approve_url,
        "reject_url": reject_url,
    }
