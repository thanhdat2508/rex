"""JSON-RPC 2.0 Gateway Protocol Manager for Rex Outbound Tunnel & Investigation.

Bridges @rex/rex-js SDK (RexTunnelClient and RexWsClient) with the neural
RexDecisionModel, InvestigatorAgent, and DynamicToolEncoder over full-duplex WebSocket.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from rex.dynamic_tools import DynamicToolEncoder
from rex.events.parser import EventParser
from rex.events.registry_loader import get_master_registry
from rex.rex_agent import InvestigatorAgent
from rex.rex_poe import InvestigationResult

logger = logging.getLogger(__name__)

# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR: int = -32700
INVALID_REQUEST: int = -32600
METHOD_NOT_FOUND: int = -32601
INVALID_PARAMS: int = -32602
INTERNAL_ERROR: int = -32603


def jsonrpc_response(req_id: Union[str, int, None], result: Any) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


def jsonrpc_error(
    req_id: Union[str, int, None],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": err,
    }


def jsonrpc_notification(method: str, params: Optional[Any] = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_request(req_id: Union[str, int], method: str, params: Optional[Any] = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


class GatewaySession:
    """Manages an active WebSocket connection with a client project using JSON-RPC 2.0."""

    def __init__(
        self,
        send_fn: Callable[[str], Any],
        model: Any,
        tok: Optional[Any] = None,
        project_id: Optional[str] = None,
        encoder: Optional[DynamicToolEncoder] = None,
    ) -> None:
        self.send_fn = send_fn
        self.agent_or_model = model
        self.model = getattr(model, "model", model)
        self.tok = tok or getattr(model, "tok", None)
        self.project_id = project_id
        self.encoder = encoder or DynamicToolEncoder()

        self.tools_schema: List[Dict[str, Any]] = []
        self.tool_specs: Dict[str, Any] = {}
        self.pending_tool_calls: Dict[str, asyncio.Future] = {}
        self.event_parser = EventParser()
        self.master_registry = get_master_registry()

    async def send_json(self, payload: Dict[str, Any]) -> None:
        msg = json.dumps(payload, ensure_ascii=False)
        res = self.send_fn(msg)
        if asyncio.iscoroutine(res):
            await res

    def register_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle 'tools/register' notification from RexTunnelClient."""
        tools = params.get("tools", [])
        if not isinstance(tools, list):
            return {"status": "error", "message": "'tools' must be a list"}

        self.tools_schema = tools
        specs = self.encoder.register_dynamic_tools(
            model=self.model,
            tools_schema=self.tools_schema,
            tok=self.tok,
        )
        self.tool_specs = {s.name: s for s in specs}
        logger.info("Gateway registered %d dynamic MCP tools from SDK", len(specs))
        return {
            "status": "OK",
            "registered_count": len(specs),
            "tools": [s.name for s in specs],
        }

    async def call_remote_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_s: float = 3.0,
    ) -> Dict[str, Any]:
        """Send 'tools/call' JSON-RPC request to SDK and await response."""
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.pending_tool_calls[call_id] = future

        req = jsonrpc_request(
            req_id=call_id,
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
        )

        try:
            await self.send_json(req)
            resp = await asyncio.wait_for(future, timeout=timeout_s)
            return resp
        except asyncio.TimeoutError:
            logger.warning("Remote tool call '%s' timed out after %.2fs", tool_name, timeout_s)
            return {"isError": True, "error": f"Timeout after {timeout_s}s"}
        except Exception as e:
            logger.warning("Remote tool call '%s' failed: %s", tool_name, e)
            return {"isError": True, "error": str(e)}
        finally:
            self.pending_tool_calls.pop(call_id, None)

    def _build_ticket_payload(
        self,
        req_id: Union[str, int],
        action: str,
        root_cause: str,
        confidence: float,
        is_novel: bool,
        context: Dict[str, Any],
        called_tools: List[str],
        observations: Dict[str, Any],
        duration_ms: float,
        rule_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Construct a structured Incident Ticket for Dashboard Ingest when threat is detected."""
        if action == "ALLOW" and not is_novel:
            return None

        path = context.get("path") or context.get("url") or "/"
        method = context.get("method") or "REQUEST"
        user_id = context.get("userId") or context.get("user_id") or "anonymous"
        ip = context.get("ip") or context.get("client_ip") or "unknown"

        priority = "URGENT" if action == "BLOCK" else ("HIGH" if action == "CHALLENGE" else "MEDIUM")

        title = f"[Rex Alert] {root_cause} on {method} {path}"
        tool_str = ", ".join(called_tools) if called_tools else "None"
        description = (
            f"Phát hiện nguy cơ '{root_cause}' với độ tin cậy {confidence * 100:.1f}%. "
            f"IP: {ip}, User: {user_id}. Các công cụ đã kiểm tra: {tool_str}."
        )

        return {
            "incidentId": str(uuid.uuid4()),
            "ticketId": str(req_id),
            "projectId": self.project_id,
            "title": title,
            "description": description,
            "priority": priority,
            "action": action,
            "rootCause": root_cause,
            "confidence": round(confidence, 4),
            "isNovel": is_novel,
            "ruleId": rule_id,
            "calledTools": list(dict.fromkeys(called_tools)),
            "summaryObservations": observations,
            "durationMs": round(duration_ms, 2),
            "createdAt": int(time.time() * 1000),
        }

    async def investigate(
        self,
        req_id: Union[str, int],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute autonomous multi-round investigation over incoming RexRequestContext."""
        t_start = time.time()
        all_called_tools: List[str] = []
        all_observations: Dict[str, Any] = {}

        # 1. Parse raw context with EventParser
        frame = self.event_parser.parse(context)
        domain_type = frame.event_type.value

        # 2. Resolve domain from MasterRegistry
        try:
            domain_cfg = self.master_registry.get_domain_type(domain_type)
            instruction = domain_cfg.description or f"Investigate {domain_type}"
            hypotheses = {h.id: h.text for h in domain_cfg.hypotheses}
            rules = {h.id: getattr(h, "falsification_criteria", []) for h in domain_cfg.hypotheses}
        except Exception:
            instruction = f"Investigate {domain_type}"
            hypotheses = {
                "H_botnet": "Distributed automated botnet activity",
                "H_sybil": "Coordinated sybil accounts",
                "H_shared": "Shared network infrastructure or CGNAT",
                "H_novel": "Unexplained anomalous transaction pattern",
            }
            rules = {}
        tool_hyp_map = {}

        # Ensure H_novel is present
        if "H_novel" not in hypotheses:
            hypotheses["H_novel"] = "Unexplained anomalous transaction pattern"

        # 3. Create tool dispatcher callback calling SDK over WebSocket
        async def tool_dispatcher(selected_tools: List[str]) -> Dict[str, Any]:
            all_called_tools.extend(selected_tools)
            tasks = [
                self.call_remote_tool(tool_name, arguments=context)
                for tool_name in selected_tools
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            observations: Dict[str, Any] = {}
            for name, res in zip(selected_tools, results):
                if isinstance(res, dict) and not res.get("isError"):
                    content_list = res.get("content", [])
                    for item in content_list:
                        if isinstance(item, dict):
                            data = item.get("data")
                            if isinstance(data, dict):
                                observations.update(data)
                            elif "text" in item:
                                observations[f"{name}_text"] = item["text"]
                elif isinstance(res, dict) and res.get("isError"):
                    observations[f"{name}_error"] = res.get("error", "Tool error")
            all_observations.update(observations)
            return observations

        def build_verdict_dict(
            winner: str,
            conf: float,
            is_novel: bool,
            action: str,
            rule_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            verdict: Dict[str, Any] = {
                "action": action,
                "confidence": round(conf, 4),
                "rootCause": winner,
                "isNovel": is_novel,
                "ruleId": rule_id,
            }
            ticket = self._build_ticket_payload(
                req_id=req_id,
                action=action,
                root_cause=winner,
                confidence=conf,
                is_novel=is_novel,
                context=context,
                called_tools=all_called_tools,
                observations=all_observations,
                duration_ms=(time.time() - t_start) * 1000,
                rule_id=rule_id,
            )
            if ticket is not None:
                verdict["ticket"] = ticket
            return verdict

        # Check if agent_or_model is an AgentPort (e.g. FakeAgent or RexAdapter)
        from .agent_port import AgentPort, Case
        if isinstance(self.agent_or_model, AgentPort):
            case = Case(
                ticket_id=str(req_id),
                type_token=domain_type,
                state=context,
            )
            avail = {t: True for t in getattr(self.agent_or_model, "model_tools", list(self.tool_specs.keys()))}
            cmd = self.agent_or_model.start(case, avail)
            for _ in range(3):
                if cmd.kind == "VERDICT" or not cmd.tools:
                    break
                obs = await tool_dispatcher(cmd.tools)
                obs_text = json.dumps(obs, ensure_ascii=False)
                cmd = self.agent_or_model.observe(obs_text, obs, avail)

            winner = max(cmd.verdict_probs.items(), key=lambda kv: kv[1])[0] if cmd.verdict_probs else "H_benign"
            conf = float(cmd.verdict_probs.get(winner, 0.95)) if cmd.verdict_probs else 0.95
            is_novel = (winner == "H_novel")
            action = "BLOCK" if winner in ("H_botnet", "H_stolen_instrument", "H_sybil", "H_credential_stuffing", "H_ato") else ("ALLOW" if winner in ("H_benign", "H_shared") else "CHALLENGE")
            rule_id = getattr(frame, "matched_rules", [""])[0] if getattr(frame, "matched_rules", None) else None
            return jsonrpc_response(req_id, build_verdict_dict(winner, conf, is_novel, action, rule_id))

        # 4. Run investigation loop with InvestigatorAgent
        agent = InvestigatorAgent(model=self.model, theta_elim=0.20)

        # Synchronous bridge for InvestigatorAgent
        def sync_dispatcher(tools: List[str]) -> Dict[str, Any]:
            # Run coroutine inside current async loop
            return asyncio.run(tool_dispatcher(tools)) if not asyncio.get_event_loop().is_running() else \
                asyncio.ensure_future(tool_dispatcher(tools))

        # Direct execution of agent investigate
        try:
            # Check if model has investigate method or use InvestigatorAgent
            res: InvestigationResult = await self._run_agent_investigate(
                agent=agent,
                domain_type=domain_type,
                instruction=instruction,
                hypotheses=hypotheses,
                rules=rules,
                tool_hyp_map=tool_hyp_map,
                tool_dispatcher=tool_dispatcher,
                initial_state=context,
            )

            # Map result to RexVerdict schema matching SDK expectations
            winner = res.predicted_hypothesis
            conf = float(res.probabilities.get(winner, 0.95))
            is_novel = bool(res.is_escalated or winner == "H_novel")

            # Determine action
            if is_novel:
                action = "CHALLENGE"
            elif winner in ("H_botnet", "H_stolen_instrument", "H_sybil", "H_credential_stuffing"):
                action = "BLOCK"
            elif winner in ("H_benign", "H_shared", "H_friendly_fraud"):
                action = "ALLOW"
            else:
                action = "CHALLENGE"

            rule_id = getattr(frame, "matched_rules", [""])[0] if getattr(frame, "matched_rules", None) else None
            return jsonrpc_response(req_id, build_verdict_dict(winner, conf, is_novel, action, rule_id))
        except Exception as e:
            logger.exception("Investigation failed for req_id=%s: %s", req_id, e)
            return jsonrpc_response(
                req_id,
                {
                    "action": "BYPASS",
                    "confidence": 0.0,
                    "rootCause": f"ERROR_{type(e).__name__}",
                    "isNovel": True,
                },
            )

    async def _run_agent_investigate(
        self,
        agent: InvestigatorAgent,
        domain_type: str,
        instruction: str,
        hypotheses: Dict[str, str],
        rules: Any,
        tool_hyp_map: Any,
        tool_dispatcher: Any,
        initial_state: Dict[str, Any],
    ) -> InvestigationResult:
        """Run agent investigation with async tool dispatcher."""
        # In InvestigatorAgent, tool_dispatcher can be adapted for async
        async def async_run():
            # If agent.investigate is synchronous, we run with async tool resolution
            from rex.rex_format import encode_rex_input, encode_evidence, collate_rex
            from rex.rex_poe import TerminationReason, RoundTrace

            hyp_keys = list(hypotheses.keys())
            current_alive = {k: True for k in hyp_keys}
            state_t = json.dumps(initial_state, ensure_ascii=False)
            called_tools: List[str] = []
            traces: List[RoundTrace] = []
            state = None
            last_step_out = None
            term_reason = TerminationReason.MAX_ROUNDS_REACHED

            device = getattr(self.model, "tool_embeddings", torch.empty(0)).device or torch.device("cpu")

            for round_idx in range(1, 4):
                tool_mask_list = agent.mask_manager.compute_tool_mask(
                    self.model.tools, current_alive, tool_hyp_map, called_tools=called_tools
                )
                tool_mask_tensor = torch.tensor([tool_mask_list], dtype=torch.bool, device=device)
                alive_mask_list = [current_alive.get(k, True) for k in hyp_keys]
                alive_mask_tensor = torch.tensor([alive_mask_list], dtype=torch.bool, device=device)

                if round_idx == 1 or state is None:
                    encoded = encode_rex_input(
                        self.tok,
                        type_token=domain_type,
                        instruction=instruction,
                        hypotheses=hypotheses,
                        state=state_t,
                        alive=current_alive,
                        tool_mask=tool_mask_list,
                    )
                    batch = collate_rex([encoded], pad_token_id=getattr(self.tok, "pad_token_id", 0) or 0)
                    batch_dev = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        _, state, step_out = self.model.forward(
                            input_ids=batch_dev["input_ids"],
                            attention_mask=batch_dev["attention_mask"],
                            marker_pos=batch_dev["marker_pos"],
                            marker_mask=batch_dev["marker_mask"],
                            qtype=batch_dev.get("qtype"),
                            tool_mask=tool_mask_tensor,
                            hyp_alive=alive_mask_tensor,
                            tool_specs=list(self.tool_specs.values()) if self.tool_specs else None,
                            hyp_names=hyp_keys,
                        )
                else:
                    ev_encoded = encode_evidence(self.tok, state_t, round_idx=round_idx, max_tokens=64)
                    ev_ids = torch.tensor([ev_encoded["input_ids"]], dtype=torch.long, device=device)
                    ev_mask = torch.tensor([ev_encoded["attention_mask"]], dtype=torch.bool, device=device)
                    with torch.no_grad():
                        _, state, step_out = self.model.forward(
                            state=state,
                            evidence_new=ev_ids,
                            evidence_mask=ev_mask,
                            tool_mask=tool_mask_tensor,
                            hyp_alive=alive_mask_tensor,
                            tool_specs=list(self.tool_specs.values()) if self.tool_specs else None,
                            hyp_names=hyp_keys,
                        )

                last_step_out = step_out
                selected_tools = step_out.selected_tools[0] if step_out.selected_tools else []
                action = step_out.proposed_action[0] if step_out.proposed_action else "CALL_TOOL"

                if action == "VERDICT" or not selected_tools:
                    term_reason = TerminationReason.EMPTY_TOOL_SELECTION
                    break

                called_tools.extend(selected_tools)

                # Execute tools remotely over WebSocket
                obs = await tool_dispatcher(selected_tools)
                traces.append(
                    RoundTrace(
                        round_idx=round_idx,
                        selected_tools=selected_tools,
                        observations=obs,
                        active_hypotheses=[k for k, v in current_alive.items() if v],
                    )
                )

                # Hybrid PoE
                with torch.no_grad():
                    n_scores = self.model.hyp_scorer(state.hyp_markers, state.M, state.memory_mask)
                current_alive = agent.poe_engine.evaluate(
                    neural_scores=n_scores,
                    hyp_keys=hyp_keys,
                    observations=obs,
                    rules=rules,
                    theta_elim=0.20,
                )

                active_count = sum(1 for v in current_alive.values() if v)
                known_count = sum(1 for k, v in current_alive.items() if k != "H_novel" and v)
                state_t = json.dumps(obs, ensure_ascii=False)

                if known_count == 0 or active_count <= 1:
                    term_reason = TerminationReason.SINGLE_SURVIVOR
                    break

            # Compute final probabilities
            probs = {}
            for idx, key in enumerate(hyp_keys):
                if not current_alive.get(key, True):
                    probs[key] = 0.0
                elif last_step_out is not None:
                    probs[key] = float(last_step_out.verdict_probs[0, idx].item())
                else:
                    probs[key] = 1.0 / len(hyp_keys)

            tot = sum(probs.values()) or 1.0
            probs = {k: v / tot for k, v in probs.items()}
            winner = max(probs.items(), key=lambda kv: kv[1])[0]

            return InvestigationResult(
                predicted_hypothesis=winner,
                probabilities=probs,
                rounds_executed=len(traces) if traces else 1,
                termination_reason=term_reason,
                is_escalated=(winner == "H_novel"),
                traces=traces,
            )

        return await async_run()

    async def handle_message(self, raw_message: str) -> Optional[Dict[str, Any]]:
        """Process an incoming JSON-RPC 2.0 message."""
        try:
            msg = json.loads(raw_message)
        except Exception:
            return jsonrpc_error(None, PARSE_ERROR, "Invalid JSON format")

        if not isinstance(msg, dict):
            return jsonrpc_error(None, INVALID_REQUEST, "Message must be a JSON object")

        req_id = msg.get("id")
        method = msg.get("method")

        # 1. Handle Response to a pending tools/call request
        if req_id is not None and req_id in self.pending_tool_calls:
            future = self.pending_tool_calls[req_id]
            if not future.done():
                if "error" in msg:
                    future.set_result({"isError": True, "error": msg["error"]})
                else:
                    future.set_result(msg.get("result", {}))
            return None

        # 2. Handle Heartbeat Ping
        if method == "ping":
            return jsonrpc_notification("pong", {"timestamp": int(time.time() * 1000)})

        if method == "pong":
            return None

        # 3. Handle 'tools/register' Notification or Request
        if method == "tools/register":
            params = msg.get("params", {})
            res = self.register_tools(params)
            if req_id is not None:
                return jsonrpc_response(req_id, res)
            return None

        # 4. Handle 'investigate' Request
        if method == "investigate":
            params = msg.get("params", {})
            if not isinstance(params, dict):
                return jsonrpc_error(req_id, INVALID_PARAMS, "'params' must be an object (RexRequestContext)")
            return await self.investigate(req_id, params)

        # 5. Method not found
        if req_id is not None:
            return jsonrpc_error(req_id, METHOD_NOT_FOUND, f"Method '{method}' not found")
        return None


class GatewayManager:
    """Coordinates gateway sessions across persistent WebSockets and HTTP requests."""

    def __init__(self, router: Any = None, agent: Any = None) -> None:
        self.router = router
        self.agent = agent
        self.sessions: Dict[str, GatewaySession] = {}
        self.encoder = DynamicToolEncoder()

    def resolve_model_and_tok(self, model_name: Optional[str] = None) -> tuple[Any, Any]:
        """Resolve underlying decision model and tokenizer."""
        if self.agent is not None:
            model = getattr(self.agent, "model", self.agent)
            tok = getattr(self.agent, "tok", None)
            return model, tok

        if self.router is not None:
            try:
                if hasattr(self.router, "agent") and self.router.agent is not None:
                    ag = self.router.agent
                    return getattr(ag, "model", ag), getattr(ag, "tok", None)
                name = model_name or getattr(self.router, "default_model", None) or "english"
                if hasattr(self.router, "_agents") and name in self.router._agents:
                    ag = self.router._agents[name]
                    return getattr(ag, "model", ag), getattr(ag, "tok", None)
                if hasattr(self.router, "load"):
                    ag = self.router.load(name)
                    return getattr(ag, "model", ag), getattr(ag, "tok", None)
            except Exception as e:
                logger.warning("GatewayManager failed to resolve model from router: %s", e)
        return None, None

    def create_session(
        self,
        send_fn: Callable[[str], Any],
        project_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> GatewaySession:
        """Create a new GatewaySession tied to a WebSocket connection."""
        model, tok = self.resolve_model_and_tok(model_name)
        session = GatewaySession(
            send_fn=send_fn,
            model=model,
            tok=tok,
            project_id=project_id,
            encoder=self.encoder,
        )
        if project_id:
            self.sessions[project_id] = session
        return session

