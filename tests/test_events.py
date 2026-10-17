"""Tests for domain event taxonomy, tokenizer vocabulary expansion, and parser."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.events import (
    ALL_EVENT_TOKENS,
    EventParser,
    EventType,
    SecurityEventFrame,
    expand_model_vocabulary,
    register_event_tokens,
    verify_event_tokens,
)


def test_taxonomy_completeness():
    """Verify all domain event types start with <TYPE_ and end with >."""
    expected_types = [
        "<TYPE_CREDENTIAL_STUFFING>",
        "<TYPE_ACCOUNT_TAKEOVER>",
        "<TYPE_PROMO_ABUSE>",
        "<TYPE_API_SCRAPING>",
        "<TYPE_PAYMENT_FRAUD>",
        "<TYPE_BENIGN>",
        "<TYPE_UNKNOWN>",
    ]
    for exp in expected_types:
        assert exp in ALL_EVENT_TOKENS, f"Missing expected token {exp}"

    for token in ALL_EVENT_TOKENS:
        assert token.startswith("<TYPE_"), f"Token {token} must start with <TYPE_"
        assert token.endswith(">"), f"Token {token} must end with >"


def test_parser_domain_events():
    """Test parser recognition of core domain events."""
    parser = EventParser(default_source="SIEM")

    # Credential stuffing
    cs_log = {"event_name": "CREDENTIAL_STUFFING", "client_ip": "198.51.100.24"}
    frame = parser.parse(cs_log)
    assert frame.event_type == EventType.CREDENTIAL_STUFFING
    assert frame.attributes["client_ip"] == "198.51.100.24"

    # Account takeover
    ato_log = {"alert_name": "USER.MFA.PUSH_DENIED", "count": 12, "user_id": "alice"}
    frame = parser.parse(ato_log)
    assert frame.event_type == EventType.ACCOUNT_TAKEOVER

    # Promo abuse
    promo_log = "Detected voucher_abuse and coupon_farming from subnet 103.1.2.0"
    frame = parser.parse(promo_log)
    assert frame.event_type == EventType.PROMO_ABUSE

    # API scraping
    scrape_log = {"user_agent": "scrapy_bot/2.1", "action": "data_harvesting"}
    frame = parser.parse(scrape_log)
    assert frame.event_type == EventType.API_SCRAPING

    # Payment fraud
    pay_log = "Alert: carding_attack identified with card_testing at stripe checkout"
    frame = parser.parse(pay_log)
    assert frame.event_type == EventType.PAYMENT_FRAUD

    # Benign
    benign_log = {"status": "ok", "message": "routine_health_check passed"}
    frame = parser.parse(benign_log)
    assert frame.event_type == EventType.BENIGN


def test_parser_batch():
    """Test batch processing of multiple logs."""
    parser = EventParser()
    events = [
        {"event_name": "CREDENTIAL_STUFFING", "ip": "1.2.3.4"},
        {"alert_name": "USER.MFA.PUSH_DENIED", "user": "bob"},
        {"status_code": 200, "message": "all good"},
    ]
    frames = parser.parse_batch(events)
    assert len(frames) == 3
    assert frames[0].event_type == EventType.CREDENTIAL_STUFFING
    assert frames[1].event_type == EventType.ACCOUNT_TAKEOVER
    assert frames[2].event_type == EventType.BENIGN


def test_security_event_frame_conversion():
    """Test converting SecurityEventFrame to state and routing text."""
    frame = SecurityEventFrame(
        event_type=EventType.CREDENTIAL_STUFFING,
        confidence=0.95,
        source="SIEM",
        raw_payload={"ip": "10.0.0.1", "attempts": 100},
        attributes={"client_ip": "10.0.0.1"},
    )
    state = frame.to_state()
    assert state["type_token"] == "<TYPE_CREDENTIAL_STUFFING>"
    assert state["confidence"] == 0.95
    assert "10.0.0.1" in state["payload_summary"]

    routing_text = frame.to_routing_text()
    assert routing_text.startswith("<TYPE_CREDENTIAL_STUFFING>")
    assert "[SRC:SIEM]" in routing_text


def test_tokenizer_mock_and_verification():
    """Test tokenizer registration behavior and verification logic."""
    class DummyTokenizer:
        def __init__(self):
            self.vocab = {"hello": 1, "world": 2, "<unk>": 0}
            self.all_special_tokens = ["<unk>"]
            self.unk_token_id = 0
            self.unk_token = "<unk>"

        def __len__(self):
            return len(self.vocab)

        def add_special_tokens(self, special_tokens_dict):
            added = 0
            tokens = special_tokens_dict.get("additional_special_tokens", [])
            for tok in tokens:
                if tok not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                    self.all_special_tokens.append(tok)
                    added += 1
            return added

        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            if text in self.vocab:
                return [self.vocab[text]]
            return [self.unk_token_id]

    tok = DummyTokenizer()
    ver_before = verify_event_tokens(tok, [EventType.CREDENTIAL_STUFFING.value])
    assert not ver_before["all_valid"]

    res = register_event_tokens(tok)
    assert res["added_count"] == len(ALL_EVENT_TOKENS)
    assert res["all_valid"] is True
    assert res["invalid_count"] == 0

    ver_after = verify_event_tokens(tok)
    assert ver_after["all_valid"] is True
    assert ver_after["valid_count"] == len(ALL_EVENT_TOKENS)


def test_expand_model_vocabulary():
    """Test resizing model embedding weights and initializing them."""
    import torch
    import torch.nn as nn

    class DummyEncoder(nn.Module):
        def __init__(self, vocab_size=10, hidden_dim=16):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_dim)
            self.config = type("Config", (), {"vocab_size": vocab_size})()

        def get_input_embeddings(self):
            return self.embeddings

        def resize_token_embeddings(self, new_vocab_size):
            old_num, dim = self.embeddings.weight.shape
            new_embeddings = nn.Embedding(new_vocab_size, dim)
            with torch.no_grad():
                new_embeddings.weight[:old_num] = self.embeddings.weight
                if new_vocab_size > old_num:
                    new_embeddings.weight[old_num:].normal_()
            self.embeddings = new_embeddings
            self.config.vocab_size = new_vocab_size
            return self.embeddings

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DummyEncoder()

    class DummyTok:
        def __init__(self):
            self.vocab = {f"word_{i}": i for i in range(10)}
            self.all_special_tokens = []

        def __len__(self):
            return len(self.vocab)

        def add_special_tokens(self, d):
            added = 0
            for t in d.get("additional_special_tokens", []):
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                    self.all_special_tokens.append(t)
                    added += 1
            return added

        def convert_tokens_to_ids(self, t):
            return self.vocab.get(t, 0)

        def encode(self, text, add_special_tokens=False):
            return [self.vocab.get(text, 0)]

    model = DummyModel()
    tok = DummyTok()
    stats = expand_model_vocabulary(model, tok, semantic_init=True)
    assert stats["new_vocab_size"] == 10 + len(ALL_EVENT_TOKENS)
    assert model.encoder.get_input_embeddings().weight.shape == (10 + len(ALL_EVENT_TOKENS), 16)
    assert model.encoder.config.vocab_size == 10 + len(ALL_EVENT_TOKENS)


if __name__ == "__main__":
    test_taxonomy_completeness()
    test_parser_domain_events()
    test_parser_batch()
    test_security_event_frame_conversion()
    test_tokenizer_mock_and_verification()
    test_expand_model_vocabulary()
    print("All tests passed successfully!")
