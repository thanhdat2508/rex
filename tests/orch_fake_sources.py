"""Mock MCP sources simulating internal microservices for testing (Step B5)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from rex.orch.tool_client import ToolSession


class FakeMCPSources(ToolSession):
    """Simulates multi-service responses matching examples/rex/registry.example.json."""

    def __init__(self, artificial_latency_s: float = 0.01) -> None:
        self.latency = artificial_latency_s
        self.call_history: list[tuple[str, dict]] = []

    async def call_tool(
        self,
        remote_name: str,
        args: Dict[str, Any],
        timeout_s: float,
    ) -> Dict[str, Any]:
        self.call_history.append((remote_name, dict(args)))
        if self.latency > 0:
            await asyncio.sleep(self.latency)

        if remote_name == "geo_ip_lookup":
            return {
                "country_mismatch": True,
                "ip_distance_km": 9200.0,
            }
        elif remote_name == "device_fingerprint":
            return {
                "device_seen_before": False,
                "unique_device_ratio": 0.25,
                "emulator_ratio": 0.10,
            }
        elif remote_name == "auth_history":
            return {
                "prior_compromise_flag": True,
                "password_reset_recent": True,
            }
        elif remote_name == "ip_reputation":
            return {
                "ip_risk_score": 0.85,
                "datacenter_ip_ratio": 0.90,
            }
        elif remote_name == "network_trace":
            return {
                "shared_asn_ratio": 0.40,
                "carrier_nat": True,
            }
        elif remote_name == "behavioral_jitter":
            return {
                "keystroke_entropy": 0.15,
                "mouse_linearity": 0.95,
            }
        elif remote_name == "payment_graph":
            return {
                "card_sharing_degree": 12,
                "chargeback_history": True,
            }
        elif remote_name == "transaction_velocity":
            return {
                "velocity_1h": 8,
                "declined_rate": 0.75,
            }
        else:
            return {}
