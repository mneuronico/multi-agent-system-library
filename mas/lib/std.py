"""MAS standard library: the functions behind the components in std.json.

Third-party packages are imported inside the function that needs them, so importing this module
needs none of them. Network calls use timeouts, failures raise (the Tool then logs the error and
returns its default_output) instead of returning made-up data, and error messages never contain
API keys: several APIs take the key in the query string, which requests and googleapiclient
include in their exception messages.
"""

import html
import logging
import os
import re
import shutil
import subprocess

# Named explicitly: the manager loads this file by path, so __name__ is not "mas.lib.std".
logger = logging.getLogger("mas.lib.std")

HTTP_TIMEOUT = 30          # seconds, for every HTTP request made here
PDF_TIMEOUT = 120          # seconds wkhtmltopdf may take to render one document
_LEGACY_WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}


# --- helpers ---------------------------------------------------------------------------------

def _redact(text, secrets):
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "***")
    return text


def _http_get(url, params, *, what, secrets=(), timeout=HTTP_TIMEOUT):
    """GET with a timeout. Raises RuntimeError (without the secrets) on network or HTTP errors."""
    import requests

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        # "from None": the original exception (and its URL with the key) is not chained.
        raise RuntimeError(f"{what} failed: {_redact(exc, secrets)}") from None
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or not 200 <= status < 300:
        detail = _redact(getattr(response, "text", "") or "", secrets).strip()[:300]
        raise RuntimeError(f"{what} failed with HTTP {status}" + (f": {detail}" if detail else ""))
    return response


def _google_service(name, version, api_key, *, what):
    """A googleapiclient service whose requests time out after HTTP_TIMEOUT seconds."""
    from googleapiclient.discovery import build

    kwargs = {"developerKey": api_key, "cache_discovery": False}
    try:
        import httplib2
    except ImportError:  # googleapiclient depends on it; keep working if a stub lacks it
        httplib2 = None
    if httplib2 is not None:
        kwargs["http"] = httplib2.Http(timeout=HTTP_TIMEOUT)
    try:
        return build(name, version, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"{what} failed: {_redact(exc, (api_key,))}") from None


def _execute_google(request, *, what, secrets):
    try:
        return request.execute()
    except Exception as exc:
        # HttpError messages contain the request URL, including ?key=<API key>.
        raise RuntimeError(f"{what} failed: {_redact(exc, secrets)}") from None


def _manager_timezone(manager):
    """The manager's timezone (a tzinfo), falling back to UTC."""
    from datetime import timezone

    tz = getattr(manager, "timezone", None)
    if isinstance(tz, str):
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz)
        except Exception:
            logger.warning("Unknown timezone %r; using UTC.", tz)
            return timezone.utc
    return tz if tz is not None else timezone.utc


def _localize(naive, tz):
    if hasattr(tz, "localize"):  # pytz
        return tz.localize(naive)
    return naive.replace(tzinfo=tz)


def _timezone_name(tz):
    name = getattr(tz, "key", None) or getattr(tz, "zone", None)
    if name:
        return str(name)
    return "UTC" if str(tz) in ("UTC", "UTC+00:00") else None


# --- Google Docs -----------------------------------------------------------------------------

def read_google_doc(manager, messages):
    from urllib.parse import quote

    api_key = manager.get_key("GOOGLE_DRIVE_API")
    doc_id = manager.get_key("GOOGLE_DOC_ID")
    if not api_key or not doc_id:
        raise RuntimeError("read_google_doc needs the GOOGLE_DRIVE_API and GOOGLE_DOC_ID keys.")

    # Google Drive API endpoint for exporting a Google Doc as plain text
    url = f"https://www.googleapis.com/drive/v3/files/{quote(str(doc_id), safe='')}/export"
    response = _http_get(
        url,
        {"mimeType": "text/plain", "key": api_key},
        what="Google Docs export",
        secrets=(api_key, doc_id),
    )
    return {"doc_text": response.content.decode("utf-8-sig")}


# --- Weather ---------------------------------------------------------------------------------

def weather_query(manager, location: str, unit: str = "metric") -> dict:
    api_key = manager.get_key("WEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("weather_query needs the WEATHER_API_KEY key.")

    # Models often answer "celsius" or "F" instead of the documented values.
    unit_aliases = {
        "metric": "metric", "celsius": "metric", "c": "metric",
        "imperial": "imperial", "fahrenheit": "imperial", "f": "imperial",
    }
    unit = unit_aliases.get(str(unit or "metric").strip().lower())
    if unit is None:
        raise ValueError("Invalid unit. Use 'metric' or 'imperial'.")
    temp_unit = "c" if unit == "metric" else "f"

    response = _http_get(
        # HTTPS: the API key travels in the query string.
        "https://api.weatherapi.com/v1/current.json",
        {"key": api_key, "q": location, "aqi": "no"},
        what="Weather API request",
        secrets=(api_key,),
    )
    data = response.json()
    current = data["current"]
    return {
        "temperature": current["temp_" + temp_unit],
        "condition": current["condition"]["text"],
        "wind_speed": current["wind_kph" if unit == "metric" else "wind_mph"],
        "location_name": f"{data['location']['name']}, {data['location']['country']}",
    }


# --- Markdown to PDF -------------------------------------------------------------------------

def _find_wkhtmltopdf(manager):
    """WKHTMLTOPDF_PATH (key or environment variable), else wkhtmltopdf on PATH."""
    configured = None
    get_key = getattr(manager, "get_key", None)
    if callable(get_key):
        configured = get_key("WKHTMLTOPDF_PATH")
    configured = configured or os.environ.get("WKHTMLTOPDF_PATH")
    if configured:
        found = shutil.which(configured) or (configured if os.path.isfile(configured) else None)
        if not found:
            raise RuntimeError(f"WKHTMLTOPDF_PATH is set to '{configured}', which is not an executable file.")
        return found
    found = shutil.which("wkhtmltopdf")
    if found:
        return found
    if os.name == "nt" and os.path.isfile(_LEGACY_WKHTMLTOPDF_PATH):
        return _LEGACY_WKHTMLTOPDF_PATH
    raise RuntimeError(
        "wkhtmltopdf was not found. Install it (https://wkhtmltopdf.org) and put it on PATH, "
        "or set WKHTMLTOPDF_PATH to the executable."
    )


def _markdown_to_safe_html(md_text):
    """Markdown to HTML with raw HTML disabled: tags in the text are shown as text, never run."""
    import markdown

    converter = markdown.Markdown()
    try:
        converter.preprocessors.deregister("html_block")
        converter.inlinePatterns.deregister("html")
    except (AttributeError, KeyError, ValueError):
        # Markdown releases before 3.0 cannot drop the raw HTML handlers: escape every tag instead.
        return markdown.markdown(html.escape(md_text, quote=False))
    return converter.convert(md_text)


def _safe_pdf_filename(title):
    """A file name made only of the title's letters, digits, spaces, dots, dashes and underscores."""
    name = re.sub(r"[^\w\- .]+", "_", str(title or ""))
    name = re.sub(r"\s+", " ", name).strip(" ._")[:100].strip(" ._")
    if name.lower().endswith(".pdf"):
        name = name[:-4].rstrip(" ._")
    if not name:
        name = "document"
    if name.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    return f"{name}.pdf"


def _user_files_dir(manager):
    """manager.files_folder/<user>: the folder manager.save_file uses for the current user."""
    folder = getattr(manager, "files_folder", None)
    if not folder:
        raise RuntimeError("markdown_to_pdf needs a manager with a files_folder.")
    user_id = None
    active_user = getattr(manager, "_active_user_id", None)
    if callable(active_user):
        user_id = active_user()
    user_id = str(user_id) if user_id is not None else "unknown"
    safe_user = getattr(manager, "_safe_user_id", None)
    subfolder = safe_user(user_id) if callable(safe_user) else re.sub(r"[^A-Za-z0-9_-]", "_", user_id)
    path = os.path.join(folder, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def _write_new_file(folder, filename, data):
    base, ext = os.path.splitext(filename)
    for i in range(1000):
        path = os.path.join(folder, filename if i == 0 else f"{base}-{i}{ext}")
        try:
            with open(path, "xb") as fh:  # never overwrites an existing file
                fh.write(data)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not find a free file name for '{filename}' in {folder}.")


def _render_pdf(binary, html_doc):
    args = [
        binary, "--quiet", "--encoding", "utf-8",
        # The markdown may come from a model or a user: run no scripts and read no local files.
        "--disable-javascript", "--disable-local-file-access",
        "-", "-",  # read HTML from stdin, write the PDF to stdout
    ]
    try:
        result = subprocess.run(
            args,
            input=html_doc.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PDF_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"wkhtmltopdf did not finish within {PDF_TIMEOUT} seconds.") from None
    if not result.stdout.startswith(b"%PDF"):
        detail = result.stderr.decode("utf-8", "replace").strip()[:500]
        raise RuntimeError(f"wkhtmltopdf failed (exit code {result.returncode}): {detail}")
    return result.stdout


def markdown_to_pdf(manager, messages):
    # The latest message must contain a text block with a JSON object: {"markdown": ..., "title": ...}.
    if not messages:
        raise ValueError("markdown_to_pdf needs a message with 'markdown' and 'title'.")
    data = manager._blocks_as_tool_input(messages[-1]["message"])
    md_text = data.get("markdown") if isinstance(data, dict) else None
    if not isinstance(md_text, str):
        raise ValueError("markdown_to_pdf needs a text block with JSON containing 'markdown' and 'title'.")
    filename = _safe_pdf_filename(data.get("title"))

    html_doc = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n"
        f"<title>{html.escape(filename[:-4])}</title>\n</head>\n<body>\n"
        f"{_markdown_to_safe_html(md_text)}\n</body>\n</html>\n"
    )
    pdf_bytes = _render_pdf(_find_wkhtmltopdf(manager), html_doc)
    path = _write_new_file(_user_files_dir(manager), filename, pdf_bytes)
    name = os.path.basename(path)

    # A document block is what the Telegram and WhatsApp bots deliver as a file.
    return [
        {"type": "text", "content": {"response": f"PDF created: {name}", "pdf_file_path": path}},
        {
            "type": "document",
            "content": {
                "kind": "file",
                "path": f"file:{path}",
                "filename": name,
                "mime_type": "application/pdf",
            },
        },
    ]


# --- MercadoPago -----------------------------------------------------------------------------

def create_payment_url(manager, name, price, currency, quantity=1, qty=None):
    import mercadopago
    from datetime import datetime, timedelta, timezone

    access_token = manager.get_key("MERCADOPAGO_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("create_payment_url needs the MERCADOPAGO_ACCESS_TOKEN key.")
    final_quantity = qty if qty is not None else quantity
    try:
        unit_price = float(price)
        final_quantity = int(final_quantity)
    except (TypeError, ValueError):
        raise ValueError("price must be a number and quantity a whole number.") from None

    sdk = mercadopago.SDK(access_token)
    expiration_date = datetime.now(timezone.utc) + timedelta(days=1)
    preference_data = {
        "items": [
            {
                "title": name,
                "quantity": final_quantity,
                "currency_id": str(currency).strip().upper(),
                "unit_price": unit_price,
            }
        ],
        "expiration_date_to": expiration_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    preference = sdk.preference().create(preference_data)
    body = preference.get("response") if isinstance(preference, dict) else None
    url = body.get("init_point") if isinstance(body, dict) else None
    if not url:
        status = preference.get("status") if isinstance(preference, dict) else None
        message = body.get("message") if isinstance(body, dict) else None
        raise RuntimeError(f"MercadoPago did not create the payment link (status {status}): {message}")
    return {"payment_url": url}


# --- YouTube ---------------------------------------------------------------------------------

def get_video_transcript(video_id):
    """The transcript text of a YouTube video, or None when it has none or it cannot be fetched.

    Works with youtube-transcript-api 1.x (``YouTubeTranscriptApi().fetch``) and with older
    releases (``YouTubeTranscriptApi.get_transcript``, removed in 1.x).
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        if hasattr(YouTubeTranscriptApi, "fetch"):
            entries = YouTubeTranscriptApi().fetch(video_id)
        else:
            entries = YouTubeTranscriptApi.get_transcript(video_id)
        parts = []
        for entry in entries:
            text = entry.get("text") if isinstance(entry, dict) else getattr(entry, "text", None)
            if text:
                parts.append(str(text))
        return " ".join(parts)
    except Exception as exc:
        logger.warning("Could not retrieve the transcript of YouTube video %s: %s", video_id, exc)
        return None


def youtube_search(manager, query, n_results=3, max_results=None):
    api_key = manager.get_key("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("youtube_search needs the YOUTUBE_API_KEY key.")
    if max_results is not None:
        n_results = max_results
    try:
        n_results = int(n_results)
    except (TypeError, ValueError):
        n_results = 3
    n_results = max(1, min(n_results, 50))  # the API accepts 0-50

    youtube = _google_service("youtube", "v3", api_key, what="YouTube search")
    search_response = _execute_google(
        youtube.search().list(q=query, part="id,snippet", type="video", maxResults=n_results),
        what="YouTube search",
        secrets=(api_key,),
    )

    results = []
    for item in search_response.get("items", []):
        video_id = (item.get("id") or {}).get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet") or {}
        results.append({
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "transcript": get_video_transcript(video_id),
        })
    return {"video_list": results}


# --- Date and calendar -----------------------------------------------------------------------

def get_current_date(manager, messages=None):
    """The current date and time in the manager's timezone (general_parameters.timezone)."""
    from datetime import datetime

    return {"current_date": datetime.now(_manager_timezone(manager)).isoformat(timespec="seconds")}


def get_calendar(manager, start_date: str, end_date: str, calendar_id: str):
    """Events from start_date to end_date, both included, as days in the manager's timezone."""
    from datetime import datetime, timedelta

    try:
        start_day = datetime.strptime(str(start_date).strip(), "%Y-%m-%d")
        end_day = datetime.strptime(str(end_date).strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date and end_date must use the YYYY-MM-DD format.") from None
    if end_day < start_day:
        raise ValueError("end_date must not be before start_date.")

    tz = _manager_timezone(manager)
    time_min = _localize(start_day, tz)
    time_max = _localize(end_day + timedelta(days=1), tz)  # end_date is inclusive

    api_key = manager.get_key("CALENDAR_API_KEY")
    if not api_key:
        raise RuntimeError("get_calendar needs the CALENDAR_API_KEY key.")
    service = _google_service("calendar", "v3", api_key, what="Google Calendar search")

    params = {
        "calendarId": calendar_id,
        "timeMin": time_min.isoformat(),
        "timeMax": time_max.isoformat(),
        "singleEvents": True,
        "orderBy": "startTime",
    }
    tz_name = _timezone_name(tz)
    if tz_name:
        params["timeZone"] = tz_name

    events = []
    for _ in range(20):  # at most 20 pages of (by default) 250 events
        page = _execute_google(
            service.events().list(**params), what="Google Calendar search", secrets=(api_key,)
        )
        events.extend(page.get("items", []))
        params["pageToken"] = page.get("nextPageToken")
        if not params["pageToken"]:
            break

    event_list = []
    for event in events:
        start = event.get("start") or {}
        end = event.get("end") or {}
        event_list.append({
            "summary": event.get("summary", "Untitled"),
            "description": event.get("description", "Not available"),
            "start": start.get("dateTime", start.get("date")),
            "end": end.get("dateTime", end.get("date")),
            "location": event.get("location", "Not specified"),
        })
    return {"event_list": event_list}


# --- Google Places ---------------------------------------------------------------------------

def find_food_places(manager, radius, keyword, location):
    google_api_key = manager.get_key("PLACES_API_KEY")
    if not google_api_key:
        raise RuntimeError("find_food_places needs the PLACES_API_KEY key.")
    lat, lng = get_lat_lng(google_api_key, address=location)
    response = _http_get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
        {"key": google_api_key, "location": f"{lat},{lng}", "radius": radius, "keyword": keyword},
        what="Google Places search",
        secrets=(google_api_key,),
    ).json()
    status = response.get("status")
    if status and status not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(f"Google Places search failed: {status} {response.get('error_message', '')}".strip())

    food_places = api_formatting(response)

    results_list = []
    max_results = 5

    # Retrieve place details (a billable call per place) only for the places that are returned.
    for place in food_places:
        if len(results_list) >= max_results:
            break
        try:
            results_list.append(get_place_details(google_api_key, place["place_id"]))
        except Exception as exc:
            logger.warning("Skipped a place whose details could not be retrieved: %s", _redact(exc, (google_api_key,)))

    return {"search_response": api_formatting_2(results_list, max_results=max_results)}


def get_lat_lng(api_key, address):
    data = _http_get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        {"key": api_key, "address": address},
        what="Geocoding request",
        secrets=(api_key,),
    ).json()
    if data.get("status") != "OK" or not data.get("results"):
        raise RuntimeError(f"Geocoding API error: {data.get('status')} {data.get('error_message', '')}".strip())
    # Extract latitude and longitude from the first result
    location = data["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]


def get_place_details(api_key, place_id):
    data = _http_get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        {
            "key": api_key,
            "place_id": place_id,
            "fields": "name,rating,price_level,formatted_address,opening_hours,website,formatted_phone_number,user_ratings_total,editorial_summary,reviews",
        },
        what="Google Place details request",
        secrets=(api_key,),
    ).json()
    status = data.get("status")
    if status and status != "OK":
        raise RuntimeError(f"Google Place details error: {status}")
    return data


def api_formatting(data):
    """Basic information for each place in a Nearby Search response."""
    parsed_data = []

    for result in data.get('results', []):
        # Build the basic information for each place
        place_info = {
            "name": result.get("name"),
            "place_id": result.get("place_id"),
            "rating": result.get("rating"),
            "price_level": result.get("price_level"),
            "formatted_address": result.get("vicinity"),
            "opening_hours": result.get("opening_hours", {}).get("open_now", "N/A"),
            "website": result.get("website", "Not available"),
            "formatted_phone_number": result.get("formatted_phone_number", "Not available")
        }
        parsed_data.append(place_info)

    return parsed_data


def api_formatting_2(results, max_results=5):
    formatted_results = []

    for result in results:
        # Extract relevant fields
        data = result.get('result', {})
        formatted_result = {
            'name': data.get('name'),
            'formatted_address': data.get('formatted_address'),
            'rating': data.get('rating', 0),
            'user_ratings_total': data.get('user_ratings_total', 0),  # Default to 0 if not available
            'price_level': data.get('price_level', 'N/A'),  # 'N/A' if not available
            'opening_hours': data.get('opening_hours', {}).get('weekday_text', []),
            'open_now': data.get('opening_hours', {}).get('open_now', 'N/A'),
            'phone_number': data.get('formatted_phone_number', 'N/A'),
            'website': data.get('website', 'N/A'),
            'editorial_summary': data.get('editorial_summary', 'N/A'),
            'reviews': data.get('reviews', [])[:3]
        }

        formatted_results.append(formatted_result)

        if len(formatted_results) >= max_results:
            break

    return formatted_results
