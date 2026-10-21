"""Unit tests for tool configuration and ToolSpec builder (Step B1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rex.orch.registry import build_tool_specs, load_tools_config
from rex.rex_registry import load_registry


def test_build_tool_specs_success():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(root, "examples", "rex", "registry.example.json")
    cfg_path = os.path.join(root, "config", "rex_tools.yaml")

    registry = load_registry(reg_path)
    cfg = load_tools_config(cfg_path)
    specs = build_tool_specs(registry, cfg, registry.tools)

    assert len(specs) == len(registry.tools)
    for spec, tool_name in zip(specs, registry.tools):
        assert spec.name == tool_name
        assert spec.level == "READ"
        assert 0.0 <= spec.cost <= 0.10
        assert len(spec.description) > 0


def test_build_tool_specs_missing_tool_raises_key_error():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(root, "examples", "rex", "registry.example.json")
    registry = load_registry(reg_path)

    cfg = {"tools": {"geo_ip_lookup": {"level": "READ", "cost": 0.0}}}
    with pytest.raises(KeyError) as exc_info:
        build_tool_specs(registry, cfg, ["geo_ip_lookup", "device_fingerprint"])
    assert "device_fingerprint" in str(exc_info.value)


def test_build_tool_specs_invalid_level_raises_value_error():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(root, "examples", "rex", "registry.example.json")
    registry = load_registry(reg_path)

    cfg = {"tools": {"geo_ip_lookup": {"level": "SUPER_ADMIN", "cost": 0.0}}}
    with pytest.raises(ValueError) as exc_info:
        build_tool_specs(registry, cfg, ["geo_ip_lookup"])
    assert "SUPER_ADMIN" in str(exc_info.value) or "geo_ip_lookup" in str(exc_info.value)


def test_build_tool_specs_read_exceeds_cost_raises_value_error():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(root, "examples", "rex", "registry.example.json")
    registry = load_registry(reg_path)

    cfg = {"tools": {"geo_ip_lookup": {"level": "READ", "cost": 0.25}}}
    with pytest.raises(ValueError) as exc_info:
        build_tool_specs(registry, cfg, ["geo_ip_lookup"])
    assert "0.1" in str(exc_info.value) or "exceeds" in str(exc_info.value)


if __name__ == "__main__":
    test_build_tool_specs_success()
    test_build_tool_specs_missing_tool_raises_key_error()
    test_build_tool_specs_invalid_level_raises_value_error()
    test_build_tool_specs_read_exceeds_cost_raises_value_error()
    print("PASS: test_orch_registry")
