from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from mas.lib import std


ROOT = Path(__file__).resolve().parents[1]


class _Manager:
    def __init__(self, values=None):
        self.values = values or {}

    def get_key(self, name):
        return self.values.get(name, f"secret-{name}")


def test_standard_library_tool_inputs_match_function_signatures():
    data = json.loads((ROOT / "mas" / "lib" / "std.json").read_text(encoding="utf-8"))

    failures = []
    for component in data["components"]:
        if component["type"] != "tool":
            continue
        fn_name = component["function"].removeprefix("fn:")
        fn = getattr(std, fn_name)
        signature = inspect.signature(fn)
        accepted = set(signature.parameters)
        accepted.discard("manager")
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not has_kwargs:
            extra = sorted(set(component.get("inputs", {})) - accepted)
            if extra:
                failures.append(f"{component['name']} -> {fn_name}: {extra}")

    assert failures == []


def test_read_google_doc_does_not_print_secrets(monkeypatch, capsys):
    requests = sys.modules["requests"]

    class Response:
        status_code = 200
        content = b"document body"
        text = "document body"

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())

    out = std.read_google_doc(
        _Manager({"GOOGLE_DRIVE_API": "drive-secret", "GOOGLE_DOC_ID": "doc-secret"}),
        messages=[],
    )

    captured = capsys.readouterr()
    assert out == {"doc_text": "document body"}
    assert "drive-secret" not in captured.out
    assert "doc-secret" not in captured.out


def test_create_payment_url_maps_quantity_and_unit_price(monkeypatch):
    captured = {}

    class Preference:
        def create(self, data):
            captured.update(data)
            return {"response": {"init_point": "https://pay.example/1"}}

    class SDK:
        def __init__(self, token):
            self.token = token

        def preference(self):
            return Preference()

    mercadopago = types.ModuleType("mercadopago")
    mercadopago.SDK = SDK
    monkeypatch.setitem(sys.modules, "mercadopago", mercadopago)

    result = std.create_payment_url(
        _Manager({"MERCADOPAGO_ACCESS_TOKEN": "mp-token"}),
        name="Class",
        price=100.0,
        currency="ARS",
        qty=2,
    )

    assert result == {"payment_url": "https://pay.example/1"}
    assert captured["items"][0]["quantity"] == 2
    assert captured["items"][0]["unit_price"] == 100.0


def test_weather_documented_units_match_implementation():
    data = json.loads((ROOT / "mas" / "lib" / "std.json").read_text(encoding="utf-8"))
    weather = next(c for c in data["components"] if c["name"] == "weather_search")
    unit_doc = weather["inputs"]["unit"]

    assert "standard" not in unit_doc
