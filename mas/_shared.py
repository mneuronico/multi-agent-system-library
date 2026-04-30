from __future__ import annotations
# mas.py

import os
import json
import sqlite3
import traceback
import uuid
import threading
from typing import Optional, List, Dict, Callable, Any, Union
import pickle
import importlib.util
import asyncio
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.request import HTTPXRequest
except ImportError:
    Update = None
    Application = None
    CommandHandler = None
    MessageHandler = None
    filters = None
    HTTPXRequest = None

    class ContextTypes:
        DEFAULT_TYPE = object

import re
import tempfile
import requests
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        raise ImportError("python-dotenv is required to load .env files. Install mas[env].")
import inspect
from datetime import datetime, timezone, timedelta
import mimetypes, shutil
from importlib import resources
import logging
import base64
import collections
import textwrap
try:
    from flask import Flask, request, jsonify
except ImportError:
    Flask = None
    request = None
    jsonify = None
import copy
import subprocess
import abc
import wave

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)



def what(file: Union[str, bytes, None] = None, h: Optional[bytes] = None) -> Optional[str]:
    data = None

    # imghdr.what supports either a file path/obj or a header bytes blob via h=
    if h is not None:
        data = h
    elif file is not None:
        try:
            if hasattr(file, "read"):
                data = file.read(32)
            elif isinstance(file, (bytes, bytearray)):
                data = bytes(file)
            else:
                with open(file, "rb") as f:
                    data = f.read(32)
        except Exception:
            return None
    else:
        return None

    # JPEG
    if data.startswith(b"\xFF\xD8\xFF"):
        return "jpeg"
    # PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    # GIF
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    # BMP
    if data.startswith(b"BM"):
        return "bmp"
    # TIFF (little/big endian)
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    # WebP (RIFF....WEBP)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    # HEIC/HEIF / AVIF ('ftyp' brand)
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"heis", b"mif1", b"msf1"):
            return "heic"
        if brand in (b"avif", b"avis"):
            return "avif"

    return None

def get_readme(owner: str, repo: str, branch: str = None,
               token: str = None) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    if branch:
        url += f"?ref={branch}"

    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["content"]).decode("utf-8")

