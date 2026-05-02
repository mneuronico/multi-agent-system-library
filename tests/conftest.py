"""Test support for optional integration dependencies.

The library imports bot, web, and cloud SDKs at module import time. The unit
tests below do not exercise those real integrations, so lightweight stubs keep
the test suite offline and runnable in a minimal Python environment.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_requests_stub() -> None:
    if importlib.util.find_spec("requests"):
        return

    requests = types.ModuleType("requests")

    class RequestException(Exception):
        def __init__(self, *args, response=None):
            super().__init__(*args)
            self.response = response

    class HTTPError(RequestException):
        pass

    def _network_disabled(*args, **kwargs):
        raise RequestException("network disabled in tests")

    requests.exceptions = types.SimpleNamespace(
        RequestException=RequestException,
        HTTPError=HTTPError,
    )
    requests.RequestException = RequestException
    requests.HTTPError = HTTPError
    requests.get = _network_disabled
    requests.post = _network_disabled
    sys.modules["requests"] = requests


def _install_dotenv_stub() -> None:
    if importlib.util.find_spec("dotenv"):
        return

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv


def _install_telegram_stub() -> None:
    if importlib.util.find_spec("telegram"):
        return

    telegram = types.ModuleType("telegram")
    telegram_ext = types.ModuleType("telegram.ext")
    telegram_request = types.ModuleType("telegram.request")

    class Update:
        @classmethod
        def de_json(cls, data, bot):
            return data

    class HTTPXRequest:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _Filter:
        def __and__(self, other):
            return self

        def __or__(self, other):
            return self

        def __invert__(self):
            return self

    class _Filters:
        TEXT = _Filter()
        COMMAND = _Filter()
        PHOTO = _Filter()
        VOICE = _Filter()
        AUDIO = _Filter()
        VIDEO = _Filter()
        DOCUMENT = _Filter()
        STICKER = _Filter()
        Document = types.SimpleNamespace(ALL=_Filter())
        Sticker = types.SimpleNamespace(ALL=_Filter())

    class _ApplicationBuilder:
        def token(self, token):
            return self

        def request(self, request):
            return self

        def build(self):
            return _Application()

    class _Application:
        bot = object()

        @classmethod
        def builder(cls):
            return _ApplicationBuilder()

        def add_handler(self, *args, **kwargs):
            return None

        def add_error_handler(self, *args, **kwargs):
            return None

        async def initialize(self):
            return None

        async def start(self):
            return None

        async def process_update(self, update):
            return None

        def run_polling(self):
            return None

    class CommandHandler:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class MessageHandler(CommandHandler):
        pass

    telegram.Update = Update
    telegram_ext.Application = _Application
    telegram_ext.CommandHandler = CommandHandler
    telegram_ext.MessageHandler = MessageHandler
    telegram_ext.filters = _Filters()
    telegram_ext.ContextTypes = types.SimpleNamespace(DEFAULT_TYPE=object)
    telegram_request.HTTPXRequest = HTTPXRequest

    sys.modules["telegram"] = telegram
    sys.modules["telegram.ext"] = telegram_ext
    sys.modules["telegram.request"] = telegram_request


def _install_flask_stub() -> None:
    if importlib.util.find_spec("flask"):
        return

    flask = types.ModuleType("flask")

    class Flask:
        def __init__(self, *args, **kwargs):
            pass

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def run(self, *args, **kwargs):
            return None

    flask.Flask = Flask
    flask.request = types.SimpleNamespace(
        method=None,
        args={},
        get_json=lambda *args, **kwargs: {},
    )
    flask.jsonify = lambda value=None, *args, **kwargs: value
    sys.modules["flask"] = flask


def _install_boto_stubs() -> None:
    if not importlib.util.find_spec("botocore"):
        botocore = types.ModuleType("botocore")
        botocore_exceptions = types.ModuleType("botocore.exceptions")

        class ClientError(Exception):
            def __init__(self, error_response=None, operation_name=None):
                super().__init__(str(error_response or {}))
                self.response = error_response or {"Error": {"Code": "Stubbed"}}
                self.operation_name = operation_name

        botocore_exceptions.ClientError = ClientError
        botocore.exceptions = botocore_exceptions
        sys.modules["botocore"] = botocore
        sys.modules["botocore.exceptions"] = botocore_exceptions

    if not importlib.util.find_spec("boto3"):
        boto3 = types.ModuleType("boto3")

        class _Client:
            def __getattr__(self, name):
                def _missing(*args, **kwargs):
                    raise RuntimeError(f"boto3 client method {name} is not stubbed")

                return _missing

        boto3.client = lambda *args, **kwargs: _Client()
        sys.modules["boto3"] = boto3


_install_requests_stub()
_install_dotenv_stub()
_install_telegram_stub()
_install_flask_stub()
_install_boto_stubs()


@pytest.fixture
def workspace_tmp_path():
    root = ROOT / "test_workdir"
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
