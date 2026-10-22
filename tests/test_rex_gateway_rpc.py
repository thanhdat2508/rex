"""Tests for Rex Gateway JSON-RPC 2.0 Protocol, WebSocket Session, and Dynamic Tools."""
import asyncio
import json
import unittest
from typing import List

import torch
import torch.nn as nn

from rex.dynamic_tools import DynamicToolEncoder
from rex.orch.fake_agent import FakeAgent
from rex.orch.gateway import (
    GatewayManager,
    GatewaySession,
)


class MockEncoder(nn.Module):
    def __init__(self, d: int = 1024):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": d})()
        self.d = d

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        b, s = input_ids.shape
        return (torch.randn(b, s, self.d),)


class MockModel(nn.Module):
    def __init__(self, tools: List[str], d: int = 1024):
        super().__init__()
        self.tools = list(tools)
        self.tool_to_idx = {name: i for i, name in enumerate(tools)}
        self.tool_embeddings = torch.randn(len(tools), d)
        self.tool_matcher = type("Matcher", (), {
            "set_tool_embeddings": lambda self, emb: None,
        })()
        self.encoder = MockEncoder(d)


class TestDynamicToolEncoder(unittest.TestCase):
    def test_encode_descriptions_fallback(self):
        encoder = DynamicToolEncoder()
        descriptions = [
            "[SLOT_NETWORK] auth_history: Lịch sử đăng nhập 24h",
            "[SLOT_DEVICE] device_fingerprint: Kiểm tra giả lập",
        ]
        embs = encoder.encode_descriptions(descriptions, d_model=1024)
        self.assertEqual(embs.shape, (2, 1024))
        # Check normalized
        norm = torch.norm(embs, p=2, dim=-1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))

    def test_register_dynamic_tools_updates_model(self):
        model = MockModel(["old_tool"])
        encoder = DynamicToolEncoder()

        tools_schema = [
            {
                "name": "auth_history",
                "description": "Lịch sử đăng nhập và số lần thử sai",
                "slot": "SLOT_NETWORK",
                "inputSchema": {"properties": {"user_id": {"type": "string"}}},
            },
            {
                "name": "card_velocity",
                "description": "Tốc độ quẹt thẻ trong 1h",
                "slot": "SLOT_GRAPH",
                "cost": 0.05,
            },
        ]

        specs = encoder.register_dynamic_tools(model, tools_schema)
        self.assertEqual(len(specs), 2)
        self.assertEqual(model.tools, ["auth_history", "card_velocity"])
        self.assertEqual(model.tool_embeddings.shape[0], 2)


class TestGatewaySessionJsonRpc(unittest.TestCase):
    def setUp(self):
        self.sent_messages: List[str] = []

        def send_fn(msg: str):
            self.sent_messages.append(msg)

        self.fake_agent = FakeAgent()
        self.session = GatewaySession(
            send_fn=send_fn,
            model=self.fake_agent,
            tok=None,
            project_id="test_project",
        )

    def test_ping_pong_heartbeat(self):
        loop = asyncio.new_event_loop()
        try:
            req = json.dumps({"jsonrpc": "2.0", "method": "ping"})
            resp = loop.run_until_complete(self.session.handle_message(req))
            self.assertIsNotNone(resp)
            self.assertEqual(resp.get("method"), "pong")
            self.assertIn("timestamp", resp.get("params", {}))
        finally:
            loop.close()

    def test_tools_register_notification(self):
        loop = asyncio.new_event_loop()
        try:
            register_msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "tools/register",
                "params": {
                    "tools": [
                        {
                            "name": "geo_ip_lookup",
                            "description": "Quốc gia của IP",
                            "slot": "SLOT_NETWORK",
                        }
                    ]
                },
            })
            resp = loop.run_until_complete(self.session.handle_message(register_msg))
            self.assertIsNone(resp)  # Notification returns None
            self.assertEqual(len(self.session.tools_schema), 1)
            self.assertIn("geo_ip_lookup", self.session.tool_specs)
        finally:
            loop.close()

    def test_tools_call_remote_dispatch_and_response(self):
        loop = asyncio.new_event_loop()
        try:
            async def scenario():
                # Start calling remote tool in background task
                call_task = asyncio.create_task(
                    self.session.call_remote_tool("auth_history", {"user_id": "usr_123"}, timeout_s=1.0)
                )

                # Give event loop a tick to send message
                await asyncio.sleep(0.01)

                self.assertEqual(len(self.sent_messages), 1)
                sent_rpc = json.loads(self.sent_messages[0])
                self.assertEqual(sent_rpc.get("method"), "tools/call")
                call_id = sent_rpc.get("id")
                self.assertEqual(sent_rpc.get("params", {}).get("name"), "auth_history")

                # SDK responds with tool execution result
                sdk_response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "result": {
                        "content": [{"type": "json", "data": {"failed_attempts": 5}}],
                        "isError": False,
                    },
                })
                await self.session.handle_message(sdk_response)

                res = await call_task
                self.assertFalse(res.get("isError"))
                self.assertEqual(res["content"][0]["data"]["failed_attempts"], 5)

            loop.run_until_complete(scenario())
        finally:
            loop.close()

    def test_investigate_request_and_verdict_format(self):
        loop = asyncio.new_event_loop()
        try:
            async def scenario():
                from rex.orch.agent_port import AgentCommand
                self.fake_agent.commands = [
                    AgentCommand(
                        kind="VERDICT",
                        tools=[],
                        s=0.92,
                        risk=0.15,
                        verdict_probs={"H_botnet": 0.88, "H_benign": 0.12},
                        conformal_set=["H_botnet"],
                    )
                ]

                investigate_msg = json.dumps({
                    "jsonrpc": "2.0",
                    "id": "req_test_123",
                    "method": "investigate",
                    "params": {
                        "ip": "185.220.101.5",
                        "method": "POST",
                        "path": "/api/v1/auth/login",
                        "userId": "usr_99182",
                    },
                })

                resp = await self.session.handle_message(investigate_msg)
                self.assertIsNotNone(resp)
                self.assertEqual(resp.get("id"), "req_test_123")
                result = resp.get("result", {})
                self.assertIn("action", result)
                self.assertEqual(result["action"], "BLOCK")
                self.assertIn("confidence", result)
                self.assertIn("rootCause", result)
                self.assertIn("isNovel", result)

                # Action is BLOCK -> ticket payload must be generated
                self.assertIn("ticket", result)
                ticket = result["ticket"]
                self.assertIsNotNone(ticket)
                self.assertEqual(ticket["priority"], "URGENT")
                self.assertEqual(ticket["rootCause"], "H_botnet")
                self.assertEqual(ticket["projectId"], "test_project")
                self.assertIn("[Rex Alert]", ticket["title"])

            loop.run_until_complete(scenario())
        finally:
            loop.close()


class TestGatewayManager(unittest.TestCase):
    def test_manager_session_lifecycle(self):
        manager = GatewayManager(agent=FakeAgent())
        session = manager.create_session(send_fn=lambda _: None, project_id="proj_xyz")
        self.assertIsNotNone(session)
        self.assertEqual(session.project_id, "proj_xyz")
        self.assertIn("proj_xyz", manager.sessions)


class TestFastApiGatewayEndpoints(unittest.TestCase):
    def test_post_investigate_endpoint(self):
        from fastapi.testclient import TestClient
        from rex.orch.agent_port import AgentCommand
        from rex.serve import create_app

        agent = FakeAgent([
            AgentCommand(
                kind="VERDICT",
                verdict_probs={"H_benign": 0.98},
            )
        ])
        app = create_app(agent=agent)
        client = TestClient(app)

        resp = client.post(
            "/v1/investigate",
            json={"ip": "1.2.3.4", "userId": "usr_991", "path": "/checkout"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("action"), "ALLOW")
        self.assertEqual(data.get("rootCause"), "H_benign")
        self.assertEqual(data.get("confidence"), 0.98)
        self.assertFalse(data.get("isNovel"))
        self.assertNotIn("ticket", data)

    def test_post_investigate_auth_gate(self):
        import os
        from fastapi.testclient import TestClient
        from rex.orch.agent_port import AgentCommand
        from rex.serve import create_app

        orig_key = os.environ.get("LAYA_API_KEY")
        try:
            os.environ["LAYA_API_KEY"] = "rex_secret_key"
            agent = FakeAgent([AgentCommand(kind="VERDICT", verdict_probs={"H_benign": 0.99})])
            app = create_app(agent=agent)
            client = TestClient(app)

            # Missing authorization header -> 401
            resp_no_auth = client.post("/v1/investigate", json={"ip": "1.2.3.4"})
            self.assertEqual(resp_no_auth.status_code, 401)

            # Valid bearer token -> 200
            resp_auth = client.post(
                "/v1/investigate",
                json={"ip": "1.2.3.4"},
                headers={"Authorization": "Bearer rex_secret_key"},
            )
            self.assertEqual(resp_auth.status_code, 200)
        finally:
            if orig_key is not None:
                os.environ["LAYA_API_KEY"] = orig_key
            else:
                os.environ.pop("LAYA_API_KEY", None)

    def test_websocket_gateway_ping_pong(self):
        from fastapi.testclient import TestClient
        from rex.serve import create_app

        app = create_app(agent=FakeAgent())
        client = TestClient(app)

        with client.websocket_connect("/v1/gateway") as ws:
            ws.send_text(json.dumps({"jsonrpc": "2.0", "method": "ping"}))
            raw_pong = ws.receive_text()
            pong = json.loads(raw_pong)
            self.assertEqual(pong.get("method"), "pong")
            self.assertIn("timestamp", pong.get("params", {}))

    def test_websocket_gateway_e2e_investigate_with_remote_tools(self):
        from fastapi.testclient import TestClient
        from rex.serve import create_app

        agent = FakeAgent.case_anh_a()
        app = create_app(agent=agent)
        client = TestClient(app)

        with client.websocket_connect("/v1/gateway") as ws:
            # 1. Register dynamic MCP tools
            ws.send_text(json.dumps({
                "jsonrpc": "2.0",
                "method": "tools/register",
                "params": {
                    "tools": [
                        {"name": "geo_ip_lookup", "description": "IP geolocation", "slot": "SLOT_NETWORK"},
                        {"name": "device_fingerprint", "description": "Device canvas hash", "slot": "SLOT_DEVICE"},
                        {"name": "auth_history", "description": "24h login velocity", "slot": "SLOT_GRAPH"},
                    ]
                }
            }))

            # 2. Send investigate request
            ws.send_text(json.dumps({
                "jsonrpc": "2.0",
                "id": "case_101",
                "method": "investigate",
                "params": {
                    "ip": "185.220.101.5",
                    "userId": "usr_victim_44",
                    "path": "/transfer",
                    "amount": 50000000,
                }
            }))

            # 3. Round 1 asks for geo_ip_lookup and device_fingerprint (2 calls)
            # Receive and respond to each tool call
            calls_received = {}
            for _ in range(2):
                msg = json.loads(ws.receive_text())
                self.assertEqual(msg.get("method"), "tools/call")
                call_id = msg.get("id")
                tool_name = msg["params"]["name"]
                calls_received[tool_name] = call_id
                # Respond with mock data
                ws.send_text(json.dumps({
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "result": {
                        "content": [{"type": "json", "data": {f"{tool_name}_checked": True}}],
                        "isError": False,
                    }
                }))

            self.assertIn("geo_ip_lookup", calls_received)
            self.assertIn("device_fingerprint", calls_received)

            # 4. Round 2 asks for auth_history (1 call)
            msg = json.loads(ws.receive_text())
            self.assertEqual(msg.get("method"), "tools/call")
            call_id = msg.get("id")
            self.assertEqual(msg["params"]["name"], "auth_history")
            ws.send_text(json.dumps({
                "jsonrpc": "2.0",
                "id": call_id,
                "result": {
                    "content": [{"type": "json", "data": {"auth_history": {"failed_count": 12}}}],
                    "isError": False,
                }
            }))

            # 5. Round 3 outputs VERDICT
            verdict_msg = json.loads(ws.receive_text())
            self.assertEqual(verdict_msg.get("id"), "case_101")
            result = verdict_msg.get("result", {})
            self.assertEqual(result.get("action"), "BLOCK")
            self.assertEqual(result.get("rootCause"), "H_ato")
            self.assertAlmostEqual(result.get("confidence", 0), 0.88, places=2)
            self.assertFalse(result.get("isNovel"))

            # Verify ticket payload for Dashboard ingest
            self.assertIn("ticket", result)
            ticket = result["ticket"]
            self.assertIsNotNone(ticket)
            self.assertEqual(ticket["priority"], "URGENT")
            self.assertEqual(ticket["action"], "BLOCK")
            self.assertEqual(ticket["rootCause"], "H_ato")
            self.assertIn("geo_ip_lookup", ticket["calledTools"])
            self.assertIn("device_fingerprint", ticket["calledTools"])
            self.assertIn("auth_history", ticket["calledTools"])
            self.assertIn("auth_history", ticket["summaryObservations"])
            self.assertGreater(ticket["durationMs"], 0)


if __name__ == "__main__":
    unittest.main()
