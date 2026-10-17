"""Pre-filtering parser for raw APM, SIEM, and Audit log events.

Transforms raw payloads into standardized SecurityEventFrame objects categorized by Domain Event.
"""

from collections.abc import Mapping
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .types import EventType, SecurityEventFrame

# Precompiled regex patterns mapping incident signatures to core Domain Events
_PATTERNS: List[Tuple[EventType, re.Pattern, float, str]] = [
    # Credential Stuffing & Password Spraying
    (
        EventType.CREDENTIAL_STUFFING,
        re.compile(
            r"(credential[_\s-]?stuffing|password[_\s-]?spray|brute[_\s-]?force|mass[_\s-]?login|failed_login_threshold)",
            re.IGNORECASE,
        ),
        0.95,
        "rule_credential_stuffing",
    ),

    # Account Takeover & Session Hijacking
    (
        EventType.ACCOUNT_TAKEOVER,
        re.compile(
            r"(account[_\s-]?takeover|ato_detected|session[_\s-]?hijack|mfa[_\s-]?fatigue|mfa[_\s-]?bypass|token[_\s-]?theft)",
            re.IGNORECASE,
        ),
        0.90,
        "rule_account_takeover",
    ),

    # Promo, Voucher & Multi-Accounting Abuse
    (
        EventType.PROMO_ABUSE,
        re.compile(
            r"(promo[_\s-]?code[_\s-]?abuse|coupon_farming|voucher_abuse|multi[_\s-]?accounting|bulk[_\s-]?signup|fake[_\s-]?account|referral_abuse)",
            re.IGNORECASE,
        ),
        0.90,
        "rule_promo_abuse",
    ),

    # API Scraping & Bot Automation
    (
        EventType.API_SCRAPING,
        re.compile(
            r"(scraping|data_harvesting|headless_browser|rate[_\s-]?limit[_\s-]?evasion|bad_bot|scrapy_bot|bot_traffic)",
            re.IGNORECASE,
        ),
        0.90,
        "rule_api_scraping",
    ),

    # Payment & Transaction Fraud
    (
        EventType.PAYMENT_FRAUD,
        re.compile(
            r"(payment[_\s-]?fraud|card[_\s-]?testing|carding_attack|stolen_card|chargeback|refund[_\s-]?abuse|sms[_\s-]?pumping)",
            re.IGNORECASE,
        ),
        0.90,
        "rule_payment_fraud",
    ),

    # Benign / Routine Traffic
    (
        EventType.BENIGN,
        re.compile(
            r"(routine_health_check|synthetic_heartbeat|monitoring_canary|known_good_client)",
            re.IGNORECASE,
        ),
        0.95,
        "rule_benign",
    ),
]

# Exact keyword/code lookup table for instant O(1) matching
_EXACT_KEYWORD_MAP: Dict[str, EventType] = {
    # GuardDuty / SecurityHub / CloudTrail
    "UNAUTHORIZEDACCESS:IAMUSER/INSTANCECREDENTIALEXFILTRATION": EventType.ACCOUNT_TAKEOVER,
    "USER.AUTHENTICATION.VERIFY_FAILED": EventType.CREDENTIAL_STUFFING,
    "USER.MFA.PUSH_DENIED": EventType.ACCOUNT_TAKEOVER,
    "USER.SESSION.SUSPICIOUS_LOCATION": EventType.ACCOUNT_TAKEOVER,
    "USER.ACCOUNT.LOCKOUT": EventType.CREDENTIAL_STUFFING,
    "PWD_LEAK": EventType.CREDENTIAL_STUFFING,
    "FAILED_LOGIN": EventType.CREDENTIAL_STUFFING,
    "POLICY.RATE_LIMIT.EXCEEDED": EventType.API_SCRAPING,
    "LIMIT_WC": EventType.API_SCRAPING,
    "CREDENTIAL_STUFFING": EventType.CREDENTIAL_STUFFING,
    "PASSWORD_SPRAY": EventType.CREDENTIAL_STUFFING,
    "SCRAPER": EventType.API_SCRAPING,
    "BOT": EventType.API_SCRAPING,
    "CARD_TESTING": EventType.PAYMENT_FRAUD,
    "PAYMENT_FRAUD": EventType.PAYMENT_FRAUD,
    "PROMO_ABUSE": EventType.PROMO_ABUSE,
}


class EventParser:
    """Pre-filtering parser for security and operational events."""

    def __init__(self, default_source: str = "SIEM"):
        self.default_source = default_source
        self._custom_rules: List[Tuple[EventType, Callable[[Dict[str, Any]], bool], float, str]] = []

    def add_rule(
        self,
        event_type: EventType,
        predicate: Callable[[Dict[str, Any]], bool],
        confidence: float = 0.95,
        rule_name: str = "custom_rule",
    ) -> None:
        """Register a custom rule predicate for custom alert formats."""
        self._custom_rules.append((event_type, predicate, confidence, rule_name))

    def _extract_attributes(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract standardized operational attributes while preserving 100% of custom project fields."""
        attrs: Dict[str, Any] = dict(payload)

        for ip_key in ("ip", "client_ip", "source_ip", "remote_addr", "src_ip"):
            val = payload.get(ip_key)
            if val and isinstance(val, str):
                attrs["client_ip"] = val
                break

        for user_key in ("user", "user_id", "username", "account_id", "email"):
            val = payload.get(user_key)
            if val:
                attrs["user_id"] = str(val)
                break

        for tenant_key in ("tenant", "tenant_id", "org_id", "workspace_id"):
            val = payload.get(tenant_key)
            if val:
                attrs["tenant_id"] = str(val)
                break

        for action_key in ("endpoint", "uri", "path", "url", "action", "event_name"):
            val = payload.get(action_key)
            if val:
                attrs["endpoint"] = str(val)
                break

        for ua_key in ("user_agent", "useragent", "http_user_agent", "ua"):
            val = payload.get(ua_key)
            if val:
                attrs["user_agent"] = str(val)
                break

        for count_key in ("failed_attempts", "attempts", "count", "volume"):
            val = payload.get(count_key)
            if val is not None:
                attrs["failed_attempts"] = val
                break

        # Fuzzy fallback for failure counts (e.g. 'bad_logins', 'login_errors', 'retries')
        if "failed_attempts" not in attrs:
            for k, val in payload.items():
                if isinstance(val, (int, float)) and re.search(r"(fail|err|retry|attempt|bad|drop|tries)", k, re.IGNORECASE):
                    attrs["failed_attempts"] = val
                    break

        for status_key in ("status", "status_code", "http_status"):
            val = payload.get(status_key)
            if val is not None:
                attrs["status_code"] = val
                break

        for asn_key in ("asn", "isp", "network"):
            val = payload.get(asn_key)
            if val:
                attrs["asn"] = str(val)
                break

        return attrs

    def parse(self, raw_event: Union[Dict[str, Any], str], source: Optional[str] = None) -> SecurityEventFrame:
        """Parse a single raw event into a standardized SecurityEventFrame."""
        src = source or self.default_source
        parsed_payload: Dict[str, Any] = {}
        raw_text: str = ""

        if isinstance(raw_event, Mapping):
            parsed_payload = dict(raw_event)
            raw_text = json.dumps(parsed_payload, ensure_ascii=False)
        elif isinstance(raw_event, str):
            raw_text = raw_event.strip()
            if raw_text.startswith("{") and raw_text.endswith("}"):
                try:
                    loaded = json.loads(raw_text)
                    if isinstance(loaded, Mapping):
                        parsed_payload = dict(loaded)
                except Exception:
                    pass
        else:
            raw_text = str(raw_event)

        # 1. Check custom predicate rules first
        if parsed_payload:
            for ev_type, predicate, conf, rule_name in self._custom_rules:
                try:
                    if predicate(parsed_payload):
                        return SecurityEventFrame(
                            event_type=ev_type,
                            confidence=conf,
                            source=src,
                            raw_payload=raw_event,
                            attributes=self._extract_attributes(parsed_payload),
                            matched_rules=[rule_name],
                        )
                except Exception:
                    continue

        # 2. Check exact keyword / code mapping in standard fields
        for field_name in ("event_name", "alert_name", "finding_type", "rule_name", "type", "action"):
            val = parsed_payload.get(field_name)
            if val and isinstance(val, str):
                cleaned_key = val.strip().upper()
                if cleaned_key in _EXACT_KEYWORD_MAP:
                    mapped_type = _EXACT_KEYWORD_MAP[cleaned_key]
                    return SecurityEventFrame(
                        event_type=mapped_type,
                        confidence=0.98,
                        source=src,
                        raw_payload=raw_event,
                        attributes=self._extract_attributes(parsed_payload),
                        matched_rules=[f"exact_match:{field_name}={val}"],
                    )

        # 3. Regex pattern matching against full serialized raw text
        matched_rules: List[str] = []
        best_type: Optional[EventType] = None
        best_conf: float = 0.0

        for ev_type, pattern, conf, rule_id in _PATTERNS:
            if pattern.search(raw_text):
                matched_rules.append(rule_id)
                if conf > best_conf:
                    best_conf = conf
                    best_type = ev_type

        if best_type is not None:
            return SecurityEventFrame(
                event_type=best_type,
                confidence=best_conf,
                source=src,
                raw_payload=raw_event,
                attributes=self._extract_attributes(parsed_payload) if parsed_payload else {},
                matched_rules=matched_rules,
            )

        # 4. Fallback classification
        status_code = parsed_payload.get("status_code") or parsed_payload.get("status")
        if status_code in (200, 204, "SUCCESS", "ok"):
            fallback_type = EventType.BENIGN
            fallback_conf = 0.80
        else:
            fallback_type = EventType.UNKNOWN
            fallback_conf = 0.20

        return SecurityEventFrame(
            event_type=fallback_type,
            confidence=fallback_conf,
            source=src,
            raw_payload=raw_event,
            attributes=self._extract_attributes(parsed_payload) if parsed_payload else {},
            matched_rules=["fallback_classification"],
        )

    def parse_batch(
        self,
        raw_events: List[Union[Dict[str, Any], str]],
        source: Optional[str] = None,
    ) -> List[SecurityEventFrame]:
        """Parse a batch of raw events into SecurityEventFrame envelopes."""
        return [self.parse(ev, source=source) for ev in raw_events]
