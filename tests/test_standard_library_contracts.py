from __future__ import annotations

import inspect
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest

from mas import AgentSystemManager
from mas.lib import std


ROOT = Path(__file__).resolve().parents[1]
STD_COMPONENTS = json.loads((ROOT / "mas" / "lib" / "std.json").read_text(encoding="utf-8"))["components"]


def _component(name):
    return next(c for c in STD_COMPONENTS if c["name"] == name)


def _function(component):
    return getattr(std, component["function"].removeprefix("fn:"))


def _content(output):
    """The dict a component produced, whether run() returned it directly or as blocks."""
    if isinstance(output, list):
        return next(block["content"] for block in output if block.get("type") == "text")
    return output


class _Manager:
    def __init__(self, values=None, timezone=None):
        self.values = values or {}
        if timezone is not None:
            self.timezone = timezone

    def get_key(self, name):
        return self.values.get(name, f"secret-{name}")


class _Response:
    def __init__(self, data=None, status_code=200, content=b"", text=""):
        self._data = data if data is not None else {}
        self.status_code = status_code
        self.content = content
        self.text = text

    def json(self):
        return self._data


# --- Stubs for the optional SDKs (CI installs none of them) -------------------------

def _stub_google(monkeypatch, *, events_pages=None, search_result=None, error=None):
    calls = {"build": [], "events": [], "search": []}

    class _Request:
        def __init__(self, result):
            self.result = result

        def execute(self):
            if error is not None:
                raise error
            return self.result

    class _Events:
        def list(self, **kwargs):
            calls["events"].append(kwargs)
            pages = events_pages or [{"items": []}]
            return _Request(pages[len(calls["events"]) - 1])

    class _Search:
        def list(self, **kwargs):
            calls["search"].append(kwargs)
            return _Request(search_result or {"items": []})

    class _Service:
        def events(self):
            return _Events()

        def search(self):
            return _Search()

    def build(name, version, **kwargs):
        calls["build"].append({"name": name, "version": version, **kwargs})
        return _Service()

    googleapiclient = types.ModuleType("googleapiclient")
    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = build
    googleapiclient.discovery = discovery
    httplib2 = types.ModuleType("httplib2")

    class Http:
        def __init__(self, timeout=None):
            self.timeout = timeout

    httplib2.Http = Http
    monkeypatch.setitem(sys.modules, "googleapiclient", googleapiclient)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    monkeypatch.setitem(sys.modules, "httplib2", httplib2)
    return calls


def _stub_transcripts(monkeypatch, *, new_api=True, error=None):
    module = types.ModuleType("youtube_transcript_api")

    if new_api:
        class YouTubeTranscriptApi:  # 1.x: instance .fetch(), no get_transcript
            def fetch(self, video_id, languages=("en",)):
                if error is not None:
                    raise error
                return [types.SimpleNamespace(text="hello"), types.SimpleNamespace(text="world")]
    else:
        class YouTubeTranscriptApi:  # 0.x: class-level get_transcript returning dicts
            @staticmethod
            def get_transcript(video_id, languages=("en",)):
                if error is not None:
                    raise error
                return [{"text": "old"}, {"text": "api"}]

    module.YouTubeTranscriptApi = YouTubeTranscriptApi
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", module)


def _stub_mercadopago(monkeypatch, response):
    captured = {}

    class Preference:
        def create(self, data):
            captured.update(data)
            return response

    class SDK:
        def __init__(self, token):
            captured["token"] = token

        def preference(self):
            return Preference()

    module = types.ModuleType("mercadopago")
    module.SDK = SDK
    monkeypatch.setitem(sys.modules, "mercadopago", module)
    return captured


def _stub_requests(monkeypatch, routes):
    requests = sys.modules.get("requests") or __import__("requests")
    calls = []

    def fake_get(url, params=None, timeout=None, **kwargs):
        calls.append({"url": url, "params": params, "timeout": timeout})
        for fragment, response in routes.items():
            if fragment in url:
                return response
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# --- Declared inputs and outputs ---------------------------------------------------

def test_standard_library_tool_inputs_match_function_signatures():
    failures = []
    for component in STD_COMPONENTS:
        if component["type"] != "tool":
            continue
        signature = inspect.signature(_function(component))
        accepted = set(signature.parameters)
        accepted.discard("manager")
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not has_kwargs:
            extra = sorted(set(component.get("inputs", {})) - accepted)
            if extra:
                failures.append(f"{component['name']} -> {component['function']}: {extra}")

    assert failures == []


def _case_google_places_search(monkeypatch):
    _stub_requests(monkeypatch, {
        "geocode": _Response({"status": "OK", "results": [{"geometry": {"location": {"lat": 1, "lng": 2}}}]}),
        "nearbysearch": _Response({"status": "OK", "results": [{"name": "A", "place_id": "p1"}]}),
        "details": _Response({"status": "OK", "result": {"name": "A"}}),
    })
    return {"radius": 500, "keyword": "pizza", "location": "Palermo"}


def _case_youtube_search(monkeypatch):
    _stub_google(monkeypatch, search_result={
        "items": [{"id": {"videoId": "abc"}, "snippet": {"title": "T", "description": "D"}}]
    })
    _stub_transcripts(monkeypatch)
    return {"query": "cats", "n_results": "2"}


def _case_calendar_search(monkeypatch):
    _stub_google(monkeypatch, events_pages=[{"items": [{"summary": "Meet", "start": {}, "end": {}}]}])
    return {"start_date": "2026-10-31", "end_date": "2026-10-31", "calendar_id": "primary"}


def _case_create_mercadopago_payment_url(monkeypatch):
    _stub_mercadopago(monkeypatch, {"status": 201, "response": {"init_point": "https://pay.example/1"}})
    return {"name": "Class", "price": "100", "currency": "ars", "quantity": "2"}


def _case_weather_search(monkeypatch):
    _stub_requests(monkeypatch, {"weatherapi.com": _Response({
        "current": {"temp_c": 21, "temp_f": 70, "condition": {"text": "Sunny"}, "wind_kph": 5, "wind_mph": 3},
        "location": {"name": "Buenos Aires", "country": "Argentina"},
    })})
    return {"location": "Buenos Aires", "unit": "metric"}


TOOL_CASES = {
    "google_places_search": _case_google_places_search,
    "youtube_search": _case_youtube_search,
    "calendar_search": _case_calendar_search,
    "create_mercadopago_payment_url": _case_create_mercadopago_payment_url,
    "weather_search": _case_weather_search,
}


def test_every_standard_library_tool_has_a_contract_case():
    tools = {c["name"] for c in STD_COMPONENTS if c["type"] == "tool"}
    assert tools == set(TOOL_CASES), "add a case to TOOL_CASES for every tool in std.json"


@pytest.mark.parametrize("tool_name", sorted(TOOL_CASES))
def test_standard_library_tools_return_exactly_their_declared_outputs(tool_name, monkeypatch):
    component = _component(tool_name)
    inputs = TOOL_CASES[tool_name](monkeypatch)

    result = _function(component)(_Manager(), **inputs)

    assert isinstance(result, dict), f"{tool_name} returned {type(result).__name__}, not a dict"
    assert set(result) == set(component["outputs"])
    if "default_output" in component:
        assert set(component["default_output"]) == set(component["outputs"])


def test_processes_return_what_their_descriptions_promise(workspace_tmp_path, monkeypatch):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _stub_markdown(monkeypatch)
    _fake_wkhtmltopdf(monkeypatch, workspace_tmp_path)
    requests = sys.modules["requests"]
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response(content=b"body"))
    monkeypatch.setenv("GOOGLE_DRIVE_API", "drive-key")
    monkeypatch.setenv("GOOGLE_DOC_ID", "doc-id")
    promised = {"current_date": "current_date", "get_google_doc": "doc_text", "markdown_to_pdf": "pdf_file_path"}
    processes = {c["name"]: c for c in STD_COMPONENTS if c["type"] == "process"}
    assert set(processes) == set(promised)

    for name, key in promised.items():
        assert key in processes[name]["description"]
        messages = [{"source": "user", "message": [{"type": "text", "content": {"markdown": "# Hi", "title": "t"}}]}]
        result = _function(processes[name])(manager, messages)
        if isinstance(result, list):
            text = next(b for b in result if b["type"] == "text")
            result = text["content"]
        assert key in result, name


def test_a_tool_that_breaks_its_contract_gets_its_default_output(workspace_tmp_path, monkeypatch):
    """The Tool checks returned keys against the declared outputs: a mismatch is never passed on."""
    (workspace_tmp_path / "fns.py").write_text(
        "def broken(manager, location, unit='metric'):\n    return {'temp': 1}\n", encoding="utf-8"
    )
    weather = dict(_component("weather_search"), function="fn:broken")
    (workspace_tmp_path / "config.json").write_text(json.dumps({"components": [weather]}), encoding="utf-8")
    manager = AgentSystemManager(config=str(workspace_tmp_path / "config.json"), base_directory=str(workspace_tmp_path))

    out = _content(manager.run(component_name="weather_search", input={"location": "X", "unit": "metric"}, user_id="u"))

    assert out["condition"].startswith("Error:")
    assert "temp" not in out


# --- Secrets ----------------------------------------------------------------------

def test_read_google_doc_does_not_print_secrets(monkeypatch, capsys):
    requests = sys.modules["requests"]
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response(content=b"document body", text="document body"))

    out = std.read_google_doc(
        _Manager({"GOOGLE_DRIVE_API": "drive-secret", "GOOGLE_DOC_ID": "doc-secret"}),
        messages=[],
    )

    captured = capsys.readouterr()
    assert out == {"doc_text": "document body"}
    assert "drive-secret" not in captured.out
    assert "doc-secret" not in captured.out


def test_read_google_doc_raises_instead_of_returning_the_error_as_text(monkeypatch):
    requests = sys.modules["requests"]
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response(status_code=403, text="denied for drive-secret"))

    with pytest.raises(RuntimeError) as exc:
        std.read_google_doc(_Manager({"GOOGLE_DRIVE_API": "drive-secret", "GOOGLE_DOC_ID": "doc"}), messages=[])

    assert "403" in str(exc.value)
    assert "drive-secret" not in str(exc.value)


def test_network_errors_never_carry_the_api_key(monkeypatch):
    requests = sys.modules["requests"]

    def failing_get(url, params=None, timeout=None, **kwargs):
        raise requests.RequestException(
            "Max retries exceeded with url: /v1/current.json?key=weather-secret&q=Paris"
        )

    monkeypatch.setattr(requests, "get", failing_get)

    with pytest.raises(RuntimeError) as exc:
        std.weather_query(_Manager({"WEATHER_API_KEY": "weather-secret"}), "Paris")

    assert "weather-secret" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__  # the original is not chained


def test_weather_failure_uses_the_default_output_and_logs_no_key(workspace_tmp_path, monkeypatch, caplog):
    requests = sys.modules["requests"]
    seen = {}

    def failing_get(url, params=None, timeout=None, **kwargs):
        seen.update(url=url, timeout=timeout)
        raise requests.RequestException(f"failed for {url}?key={params['key']}")

    monkeypatch.setattr(requests, "get", failing_get)
    monkeypatch.setenv("WEATHER_API_KEY", "weather-secret")
    (workspace_tmp_path / "config.json").write_text(
        json.dumps({"general_parameters": {"imports": "std->weather_search"}, "components": []}), encoding="utf-8"
    )
    manager = AgentSystemManager(config=str(workspace_tmp_path / "config.json"), base_directory=str(workspace_tmp_path))

    with caplog.at_level(logging.DEBUG):
        out = _content(manager.run(component_name="weather_search", input={"location": "Paris", "unit": "metric"}, user_id="u"))

    assert seen["url"].startswith("https://")  # the key travels in the query string
    assert seen["timeout"] == std.HTTP_TIMEOUT
    assert out["condition"].startswith("Error:")
    assert "weather-secret" not in caplog.text


def test_google_api_errors_never_carry_the_api_key(monkeypatch):
    _stub_google(monkeypatch, error=Exception(
        '<HttpError 403 when requesting https://www.googleapis.com/calendar/v3/calendars/x/events'
        '?key=calendar-secret&alt=json returned "Forbidden">'
    ))

    with pytest.raises(RuntimeError) as exc:
        std.get_calendar(_Manager({"CALENDAR_API_KEY": "calendar-secret"}), "2026-10-31", "2026-10-31", "x")

    assert "calendar-secret" not in str(exc.value)
    assert "403" in str(exc.value)


# --- Calendar and date ----------------------------------------------------------------

def test_calendar_end_date_is_inclusive_in_the_manager_timezone(monkeypatch):
    from zoneinfo import ZoneInfo

    calls = _stub_google(monkeypatch)
    manager = _Manager(timezone=ZoneInfo("America/Argentina/Buenos_Aires"))

    assert std.get_calendar(manager, "2026-10-31", "2026-10-31", "primary") == {"event_list": []}

    params = calls["events"][0]
    assert params["timeMin"] == "2026-10-31T00:00:00-03:00"
    assert params["timeMax"] == "2026-11-01T00:00:00-03:00"  # one whole day, not an empty range
    assert params["timeZone"] == "America/Argentina/Buenos_Aires"
    assert calls["build"][0]["http"].timeout == std.HTTP_TIMEOUT


def test_calendar_days_follow_daylight_saving_changes(monkeypatch):
    calls = _stub_google(monkeypatch)

    std.get_calendar(_Manager(timezone="America/New_York"), "2026-11-01", "2026-11-01", "primary")

    assert calls["events"][0]["timeMin"] == "2026-11-01T00:00:00-04:00"
    assert calls["events"][0]["timeMax"] == "2026-11-02T00:00:00-05:00"


def test_calendar_reads_every_page_and_rejects_reversed_ranges(monkeypatch):
    calls = _stub_google(monkeypatch, events_pages=[
        {"items": [{"summary": "a", "start": {"date": "2026-10-31"}, "end": {"date": "2026-11-01"}}], "nextPageToken": "p2"},
        {"items": [{"summary": "b", "start": {"dateTime": "x"}, "end": {"dateTime": "y"}}]},
    ])

    out = std.get_calendar(_Manager(), "2026-10-01", "2026-10-31", "primary")

    assert [e["summary"] for e in out["event_list"]] == ["a", "b"]
    assert calls["events"][1]["pageToken"] == "p2"
    assert calls["events"][0]["timeMax"] == "2026-11-01T00:00:00+00:00"
    with pytest.raises(ValueError):
        std.get_calendar(_Manager(), "2026-10-31", "2026-10-01", "primary")


def test_current_date_uses_the_manager_timezone():
    from zoneinfo import ZoneInfo

    assert std.get_current_date(_Manager(timezone=ZoneInfo("Asia/Tokyo")), [])["current_date"].endswith("+09:00")
    assert std.get_current_date(_Manager(timezone="America/Argentina/Buenos_Aires"))["current_date"].endswith("-03:00")
    assert std.get_current_date(_Manager())["current_date"].endswith("+00:00")


def test_current_date_process_reads_the_configured_timezone(workspace_tmp_path):
    (workspace_tmp_path / "config.json").write_text(
        json.dumps({"general_parameters": {"timezone": "Asia/Tokyo", "imports": "std->current_date"}, "components": []}),
        encoding="utf-8",
    )
    manager = AgentSystemManager(config=str(workspace_tmp_path / "config.json"), base_directory=str(workspace_tmp_path))

    out = _content(manager.run(component_name="current_date", input="what day is it?", user_id="u"))

    assert out["current_date"].endswith("+09:00")


# --- YouTube ------------------------------------------------------------------------

@pytest.mark.parametrize("new_api, expected", [(True, "hello world"), (False, "old api")])
def test_transcripts_work_with_old_and_new_youtube_transcript_api(monkeypatch, new_api, expected):
    _stub_transcripts(monkeypatch, new_api=new_api)

    assert std.get_video_transcript("abc") == expected


def test_missing_transcript_is_null_and_logged(monkeypatch, caplog):
    _stub_transcripts(monkeypatch, error=Exception("Subtitles are disabled for this video"))
    _stub_google(monkeypatch, search_result={
        "items": [{"id": {"videoId": "abc"}, "snippet": {"title": "T", "description": "D"}}]
    })

    with caplog.at_level(logging.WARNING, logger="mas.lib.std"):
        out = std.youtube_search(_Manager(), "cats")

    assert out["video_list"][0]["transcript"] is None
    assert "Subtitles are disabled" in caplog.text


# --- Markdown to PDF -----------------------------------------------------------------

def _stub_markdown(monkeypatch):
    module = types.ModuleType("markdown")

    class _Registry:
        def deregister(self, name, strict=True):
            return None

    class Markdown:
        def __init__(self, *args, **kwargs):
            self.preprocessors = _Registry()
            self.inlinePatterns = _Registry()

        def convert(self, text):
            return f"<p>{text}</p>"

    module.Markdown = Markdown
    module.markdown = lambda text, **kwargs: f"<p>{text}</p>"
    monkeypatch.setitem(sys.modules, "markdown", module)


def _fake_wkhtmltopdf(monkeypatch, folder, *, configure=True):
    exe = folder / "wkhtmltopdf-fake.exe"
    exe.write_text("", encoding="utf-8")
    calls = []

    def fake_run(args, input=None, stdout=None, stderr=None, timeout=None, check=False):
        calls.append({"args": list(args), "html": input.decode("utf-8"), "timeout": timeout})
        return subprocess.CompletedProcess(args, 0, stdout=b"%PDF-1.4 fake", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    if configure:
        monkeypatch.setenv("WKHTMLTOPDF_PATH", str(exe))
    else:
        monkeypatch.delenv("WKHTMLTOPDF_PATH", raising=False)
    return calls, exe


def _pdf_project(folder):
    (folder / "config.json").write_text(
        json.dumps({"general_parameters": {"imports": "std->markdown_to_pdf"}, "components": []}), encoding="utf-8"
    )
    return AgentSystemManager(config=str(folder / "config.json"), base_directory=str(folder))


def test_markdown_to_pdf_saves_a_document_in_the_user_files_folder(workspace_tmp_path, monkeypatch):
    _stub_markdown(monkeypatch)
    calls, exe = _fake_wkhtmltopdf(monkeypatch, workspace_tmp_path)
    manager = _pdf_project(workspace_tmp_path)
    request = {"markdown": "# Report", "title": "../../outside/evil"}

    first = manager.run(component_name="markdown_to_pdf", input=request, user_id="u1")
    second = manager.run(component_name="markdown_to_pdf", input=request, user_id="u1")

    text, document = first
    path = Path(text["content"]["pdf_file_path"])
    assert path.parent == Path(manager.files_folder) / "u1"
    assert path.name == "outside_evil.pdf"
    assert path.read_bytes() == b"%PDF-1.4 fake"
    assert document == {
        "type": "document",
        "content": {"kind": "file", "path": f"file:{path}", "filename": "outside_evil.pdf", "mime_type": "application/pdf"},
    }
    assert manager._is_managed_file_path(str(path))
    assert Path(second[0]["content"]["pdf_file_path"]).name == "outside_evil-1.pdf"  # never overwrites
    assert not list(workspace_tmp_path.glob("*.pdf"))
    assert not (workspace_tmp_path.parent / "outside").exists()

    run = calls[0]
    assert run["args"][0] == str(exe)
    assert "--disable-javascript" in run["args"] and "--disable-local-file-access" in run["args"]
    assert run["timeout"] == std.PDF_TIMEOUT


@pytest.mark.parametrize("title, expected", [
    ("Quarterly report", "Quarterly report.pdf"),
    ("..\\..\\Windows\\win.ini", "Windows_win.ini.pdf"),
    ("CON", "_CON.pdf"),
    ("", "document.pdf"),
    ("notes.pdf", "notes.pdf"),
    ("Informe año 2026", "Informe año 2026.pdf"),
])
def test_pdf_file_names_are_sanitized(title, expected):
    assert std._safe_pdf_filename(title) == expected


def test_markdown_to_pdf_escapes_raw_html():
    pytest.importorskip("markdown")

    out = std._markdown_to_safe_html("# T\n\n<script>alert(1)</script>\n\n<img src=x onerror=alert(1)> and `<b>`")

    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;" in out
    assert "<h1>T</h1>" in out


def test_wkhtmltopdf_is_found_on_path_or_reported_clearly(workspace_tmp_path, monkeypatch):
    monkeypatch.delenv("WKHTMLTOPDF_PATH", raising=False)
    monkeypatch.setattr(std, "_LEGACY_WKHTMLTOPDF_PATH", str(workspace_tmp_path / "not-installed.exe"))
    manager = _Manager()
    manager.get_key = lambda name: None

    monkeypatch.setattr(std.shutil, "which", lambda name: "/usr/local/bin/wkhtmltopdf" if name == "wkhtmltopdf" else None)
    assert std._find_wkhtmltopdf(manager) == "/usr/local/bin/wkhtmltopdf"

    monkeypatch.setattr(std.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="WKHTMLTOPDF_PATH"):
        std._find_wkhtmltopdf(manager)

    monkeypatch.setenv("WKHTMLTOPDF_PATH", str(workspace_tmp_path / "missing.exe"))
    with pytest.raises(RuntimeError, match="not an executable"):
        std._find_wkhtmltopdf(manager)


def test_wkhtmltopdf_failures_raise(workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=b"", stderr=b"boom"))
    with pytest.raises(RuntimeError, match="boom"):
        std._render_pdf("wkhtmltopdf", "<p>x</p>")

    def too_slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", too_slow)
    with pytest.raises(RuntimeError, match="did not finish"):
        std._render_pdf("wkhtmltopdf", "<p>x</p>")


# --- MercadoPago and Places -----------------------------------------------------------

def test_create_payment_url_maps_quantity_and_unit_price(monkeypatch):
    captured = _stub_mercadopago(monkeypatch, {"response": {"init_point": "https://pay.example/1"}})

    result = std.create_payment_url(
        _Manager({"MERCADOPAGO_ACCESS_TOKEN": "mp-token"}),
        name="Class",
        price="100.5",
        currency="ars",
        qty="2",
    )

    assert result == {"payment_url": "https://pay.example/1"}
    assert captured["items"][0]["quantity"] == 2
    assert captured["items"][0]["unit_price"] == 100.5
    assert captured["items"][0]["currency_id"] == "ARS"


def test_create_payment_url_raises_when_mercadopago_refuses(monkeypatch):
    _stub_mercadopago(monkeypatch, {"status": 400, "response": {"message": "invalid unit_price"}})

    with pytest.raises(RuntimeError, match="invalid unit_price"):
        std.create_payment_url(_Manager(), name="Class", price=1, currency="USD")


def test_places_errors_are_not_reported_as_no_results(monkeypatch):
    _stub_requests(monkeypatch, {
        "geocode": _Response({"status": "OK", "results": [{"geometry": {"location": {"lat": 1, "lng": 2}}}]}),
        "nearbysearch": _Response({"status": "REQUEST_DENIED", "error_message": "API key invalid"}),
    })

    with pytest.raises(RuntimeError, match="REQUEST_DENIED"):
        std.find_food_places(_Manager(), radius=500, keyword="pizza", location="Palermo")


def test_weather_documented_units_match_implementation():
    unit_doc = _component("weather_search")["inputs"]["unit"]

    assert "standard" not in unit_doc
