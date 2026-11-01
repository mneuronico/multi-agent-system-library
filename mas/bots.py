from __future__ import annotations

import collections as _collections
import concurrent.futures as _futures
import contextlib
import contextvars as _contextvars
import time as _time
from typing import Iterable, Tuple

from ._shared import *

DEFAULT_ERROR_MSG = "Sorry, something went wrong while processing your message. Please try again later."
DEFAULT_MEDIA_TOO_LARGE_MSG = "Sorry, that file is too large for me to process (the limit is {limit_mb} MB)."
# Telegram's cloud Bot API refuses to hand out files above 20 MB, and WhatsApp's own limits for
# images, audio and video are below that, so 20 MB covers what both platforms normally deliver.
DEFAULT_MAX_MEDIA_BYTES = 20 * 1024 * 1024
TELEGRAM_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024
# Graph API v23.0 was released in May 2025; Meta keeps each version available for about two
# years, so it should keep working until around mid-2027. Pass api_version to use another one.
DEFAULT_WHATSAPP_API_VERSION = "v23.0"

_COMMAND_NAME_RE = re.compile(r"/[a-z0-9_]{1,32}")
_API_VERSION_RE = re.compile(r"v\d+(\.\d+)?")
_SAFE_EXTENSION_RE = re.compile(r"\.[a-z0-9]{1,10}")
_MEDIA_TYPES = ("image", "audio", "video", "document", "sticker")

# Holds the bot whose turn the current task is running. Admin maintenance commands run inside
# the admin's own turn and must not wait for themselves.
_ACTIVE_TURN = _contextvars.ContextVar("mas_bot_active_turn", default=None)


class MediaTooLargeError(Exception):
    """An incoming file is larger than ``max_media_bytes`` or than the platform lets bots download."""

    def __init__(self, size: Optional[int], limit: Optional[int]):
        self.size = size
        self.limit = limit
        shown = "unknown" if size is None else str(size)
        super().__init__(f"file of {shown} bytes is over the {limit}-byte limit")


class WhatsappAPIError(RuntimeError):
    """The WhatsApp Cloud API rejected a request (any non-2xx answer)."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, error_code: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code

    @classmethod
    def from_response(cls, action: str, response: Any) -> "WhatsappAPIError":
        # Only Meta's error description is kept: never the request payload, which holds the
        # message text and the recipient's phone number.
        status = getattr(response, "status_code", None)
        detail = code = None
        try:
            error = (response.json() or {}).get("error") or {}
            if isinstance(error, dict):
                detail = error.get("message")
                code = error.get("code")
        except Exception:
            pass
        text = f"WhatsApp Cloud API rejected the {action} request (HTTP {status})"
        if detail:
            text += f": {detail}"
        if code is not None:
            text += f" [code {code}]"
        return cls(text, status_code=status, error_code=code)


# ---------------------------------------------------------------------------
# Long text splitting
# ---------------------------------------------------------------------------

def _text_length(text: str) -> int:
    """Length the way Telegram and WhatsApp count it (UTF-16 code units: most emoji count 2)."""
    return len(text.encode("utf-16-le")) // 2


def _prefix_within(text: str, limit: int) -> int:
    units = 0
    for index, char in enumerate(text):
        units += 2 if ord(char) > 0xFFFF else 1
        if units > limit:
            return index
    return len(text)


def _split_text_once(text: str, limit: int) -> Tuple[str, str]:
    """Return ``(head, rest)`` with ``head`` within ``limit``, cut at the best natural boundary.

    Paragraph breaks are preferred, then line breaks, then spaces; a boundary in the second
    half of the allowed window wins over a better kind of boundary early in it. A text with no
    boundary at all is cut hard.
    """
    if _text_length(text) <= limit:
        return text, ""
    end = max(1, _prefix_within(text, limit))
    window = text[:end]
    cut = skip = None
    for min_index in (max(1, end // 2), 1):
        for separator in ("\n\n", "\n", " "):
            index = window.rfind(separator)
            if index >= min_index:
                cut, skip = index, len(separator)
                break
        if cut is not None:
            break
    if cut is None:
        cut, skip = end, 0
    head = text[:cut].rstrip()
    rest = text[cut + skip:]
    if skip and window[cut] == "\n":
        rest = rest.lstrip("\n")
    return head, rest


def _split_text(text: str, limit: int) -> List[str]:
    """Split ``text`` into ordered chunks of at most ``limit`` characters; blank chunks are dropped.

    A text that fits is returned unchanged.
    """
    text = text or ""
    if _text_length(text) <= limit:
        return [text] if text.strip() else []
    chunks = []
    rest = text
    while rest:
        head, rest = _split_text_once(rest, limit)
        head = head.rstrip()
        if head:
            chunks.append(head)
    return chunks


class _SendQueue:
    """Outgoing messages of one user on one event loop, sent one at a time in order."""

    __slots__ = ("items", "worker", "last")

    def __init__(self):
        self.items = _collections.deque()
        self.worker = None
        self.last = None


class Bot(abc.ABC):
    # Platform limits, in UTF-16 code units. Longer texts are split; longer captions are cut
    # and the rest is sent as text right after the media.
    TEXT_LIMIT = 4096
    CAPTION_LIMIT = 1024
    # Upper bound for the duplicate-delivery cache (entries also expire after dedup_ttl).
    DEDUP_MAX_ENTRIES = 10000
    _ACTIVITY_POLL_SECONDS = 0.05

    def __init__(
        self,
        manager: AgentSystemManager,
        component_name: Optional[str] = None,
        verbose: bool = False,
        on_update: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        speech_to_text: Optional[Callable] = None,
        whisper_provider: Optional[str] = None,
        whisper_model: Optional[str] = None,
        on_start_msg: str = "Hey! Talk to me or type '/clear' to erase your message history.",
        on_clear_msg: str = "Message history deleted.",
        on_help_msg: str = "Here are the available commands:",
        unknown_command_msg: Optional[str] = "I don't recognize that command. Type /help to see what I can do.",
        custom_commands: Optional[Union[Dict, List[Dict]]] = None,
        return_token_count: bool = False,
        ensure_delivery: bool = False,
        delivery_timeout: float = 5.0,
        max_allowed_message_delay: Optional[float] = 120.0,
        error_msg: Optional[str] = DEFAULT_ERROR_MSG,
        allowed_user_ids: Optional[Iterable[Union[str, int]]] = None,
        media_dirs: Optional[Iterable[str]] = None,
        max_media_bytes: Optional[int] = DEFAULT_MAX_MEDIA_BYTES,
        media_too_large_msg: Optional[str] = DEFAULT_MEDIA_TOO_LARGE_MSG,
        store_original_payload: bool = False,
        dedup_ttl: Optional[float] = 3600.0,
        send_queue_timeout: Optional[float] = 120.0,
        maintenance_timeout: Optional[float] = 60.0,
    ):
        # Bot logs go through the library logger (mas._shared), like the rest of MAS, so they
        # show up with its handler and follow AgentSystemManager(log_level=...).
        self.logger = logger
        self.manager = manager
        self.component_name = component_name
        self.verbose = verbose
        self.on_update_user_callback = on_update or self.manager.on_update
        self.on_complete_user_callback = on_complete or self.manager.on_complete
        self.speech_to_text = speech_to_text
        self.whisper_provider = whisper_provider
        self.whisper_model = whisper_model

        self.on_start_msg = on_start_msg
        self.on_clear_msg = on_clear_msg
        self.on_help_msg = on_help_msg
        self.unknown_command_msg = unknown_command_msg
        self.commands = {}

        self._register_commands(custom_commands)

        self.return_token_count = return_token_count
        self.ensure_delivery = ensure_delivery
        self.delivery_timeout = delivery_timeout
        self.max_allowed_message_delay = max_allowed_message_delay
        self.error_msg = error_msg
        self.allowed_user_ids = self._normalize_allowed_user_ids(allowed_user_ids)

        if isinstance(media_dirs, (str, os.PathLike)):
            media_dirs = [media_dirs]
        self.media_dirs = [os.path.abspath(os.fspath(d)) for d in (media_dirs or [])]
        if max_media_bytes is not None and (not isinstance(max_media_bytes, int) or max_media_bytes <= 0):
            raise ValueError("max_media_bytes must be a positive number of bytes, or None for no limit.")
        self.max_media_bytes = max_media_bytes
        self.media_too_large_msg = media_too_large_msg
        self.store_original_payload = bool(store_original_payload)
        self.dedup_ttl = dedup_ttl
        self.send_queue_timeout = send_queue_timeout
        self.maintenance_timeout = maintenance_timeout

        # (event loop, user_id) -> [asyncio.Lock, number of messages holding or waiting for it].
        # Locks are created lazily on the loop that uses them and removed when the count drops
        # to zero, so the map only ever holds users that have a message in flight.
        self._user_locks: Dict[Any, List[Any]] = {}
        self._user_locks_guard = threading.Lock()

        # (event loop, user_id) -> _SendQueue; removed once the queue is empty.
        self._send_queues: Dict[Any, _SendQueue] = {}
        self._send_queues_guard = threading.Lock()

        # (chat id, platform message id) -> expiry time; oldest first.
        self._seen_messages: "_collections.OrderedDict[Tuple[str, str], float]" = _collections.OrderedDict()
        self._seen_messages_guard = threading.Lock()

        # Turns currently being processed, and pending admin maintenance operations.
        self._activity_guard = threading.Lock()
        self._active_turns = 0
        self._maintenance_requests = 0

    # ------------------------------------------------------------------
    # Users, admins and allow lists
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_user_id(value: Any) -> str:
        text = str(value).strip()
        # WhatsApp sends phone numbers without "+"; accept "+5491..." in the allow list too.
        return text[1:] if text.startswith("+") else text

    @classmethod
    def _normalize_allowed_user_ids(cls, allowed_user_ids: Any) -> Optional[frozenset]:
        if allowed_user_ids is None:
            return None
        if isinstance(allowed_user_ids, (str, int)):
            allowed_user_ids = [allowed_user_ids]
        return frozenset(cls._normalize_user_id(uid) for uid in allowed_user_ids)

    def _is_admin(self, sender_id: Any) -> bool:
        admin_user_id = getattr(self.manager, "admin_user_id", None)
        if not admin_user_id or sender_id is None:
            return False
        return self._normalize_user_id(sender_id) == self._normalize_user_id(admin_user_id)

    def _is_user_allowed(self, user_id: Any, chat_id: Any = None) -> bool:
        """``user_id`` is the sender. ``chat_id`` lets an explicitly listed group admit its members."""
        if self.allowed_user_ids is None:
            return True
        normalized = self._normalize_user_id(user_id)
        if normalized in self.allowed_user_ids:
            return True
        if chat_id is not None:
            chat = self._normalize_user_id(chat_id)
            if chat != normalized and chat in self.allowed_user_ids:
                return True
        return self._is_admin(user_id)

    @staticmethod
    def _sender_id(parsed_message: Any) -> Optional[str]:
        if not isinstance(parsed_message, dict):
            return None
        sender = parsed_message.get("sender_id")
        if sender in (None, ""):
            sender = parsed_message.get("user_id")
        return None if sender is None else str(sender)

    # ------------------------------------------------------------------
    # Platform hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _parse_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def _send_blocks(self, user_id: str, blocks: List[Dict], original_message: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def _download_media_and_save(self, user_id: str, media_info: Any) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def _send_log_files(self, user_id: str, files_to_send: List[Dict[str, str]], original_message: Dict[str, Any]) -> int:
        raise NotImplementedError

    def _command_bot_name(self) -> Optional[str]:
        """The bot's own username, for ``/command@username``; None when the platform has none."""
        return None

    # ------------------------------------------------------------------
    # Direct actions
    # ------------------------------------------------------------------

    async def send_text(
        self,
        user_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Send ``text``; texts over the platform limit go out as several messages, in order.

        ``reply_to_message_id`` applies to the first message. Returns the platform result of
        the last message sent.
        """
        text = text if isinstance(text, str) else str(text)
        chunks = _split_text(text, self.TEXT_LIMIT) or [text]
        result = None
        for index, chunk in enumerate(chunks):
            result = await self._send_text_message(
                str(user_id),
                chunk,
                reply_to_message_id=reply_to_message_id if index == 0 else None,
                **kwargs
            )
        return result

    async def send_image(
        self,
        user_id: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await self._send_media(
            str(user_id),
            "image",
            media,
            caption=caption,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )

    async def send_audio(
        self,
        user_id: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await self._send_media(
            str(user_id),
            "audio",
            media,
            caption=caption,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )

    async def send_video(
        self,
        user_id: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await self._send_media(
            str(user_id),
            "video",
            media,
            caption=caption,
            filename=filename,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )

    async def send_document(
        self,
        user_id: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await self._send_media(
            str(user_id),
            "document",
            media,
            caption=caption,
            filename=filename,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )

    async def send_sticker(
        self,
        user_id: str,
        media: Any,
        *,
        emoji: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await self._send_media(
            str(user_id),
            "sticker",
            media,
            emoji=emoji,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )

    async def react_to_message(
        self,
        user_id: str,
        message_id: str,
        emoji: str,
        **kwargs
    ) -> Any:
        return await self._send_reaction(str(user_id), str(message_id), emoji, **kwargs)

    async def remove_reaction(self, user_id: str, message_id: str, **kwargs) -> Any:
        return await self.react_to_message(user_id, message_id, "", **kwargs)

    async def _send_text_message(
        self,
        user_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    async def _send_media(
        self,
        user_id: str,
        media_type: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    async def _send_reaction(
        self,
        user_id: str,
        message_id: str,
        emoji: str,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Outgoing media
    # ------------------------------------------------------------------

    def _block_media_source(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, dict):
            return content
        for key in ("path", "url", "link", "media_id", "id", "file_id"):
            value = content.get(key)
            if value:
                return value
        return content

    def _block_caption(self, block: Dict[str, Any]) -> Optional[str]:
        content = block.get("content")
        if isinstance(content, dict):
            caption = content.get("caption")
            if caption is not None:
                return str(caption)
        metadata = block.get("metadata")
        if isinstance(metadata, dict) and metadata.get("caption") is not None:
            return str(metadata["caption"])
        return None

    def _block_filename(self, block: Dict[str, Any]) -> Optional[str]:
        content = block.get("content")
        if isinstance(content, dict):
            filename = content.get("filename") or content.get("file_name")
            if filename is not None:
                return str(filename)
        return None

    @staticmethod
    def _is_voice_note_block(block: Dict[str, Any]) -> bool:
        for part in (block.get("content"), block.get("metadata")):
            if isinstance(part, dict) and part.get("is_voice_note"):
                return True
        return False

    def _media_roots(self) -> List[str]:
        roots = []
        files_folder = getattr(self.manager, "files_folder", None)
        if files_folder:
            roots.append(files_folder)
        roots.extend(self.media_dirs)
        return roots

    def _is_allowed_media_path(self, path: str) -> bool:
        try:
            target = os.path.normcase(os.path.realpath(path))
        except (TypeError, ValueError, OSError):
            return False
        for root in self._media_roots():
            try:
                real_root = os.path.normcase(os.path.realpath(root))
                if target != real_root and os.path.commonpath([real_root, target]) == real_root:
                    return True
            except (TypeError, ValueError, OSError):
                # For example a root on another Windows drive.
                continue
        return False

    def _local_path(self, media: Any) -> Optional[str]:
        """Return the local file ``media`` points to, or None for URLs and platform ids.

        Blocks can come from tools or model output, so only files inside the manager's
        ``files_folder`` or one of ``media_dirs`` may be sent; anything else raises
        ``PermissionError`` (and is logged) instead of being uploaded to the chat.
        """
        if isinstance(media, os.PathLike):
            media = os.fspath(media)
        if not isinstance(media, str) or not media:
            return None
        if media.lower().startswith(("http://", "https://")):
            return None
        if media.startswith("file:"):
            path = media[5:]
        elif os.path.isabs(media) or os.path.isfile(media):
            path = media
        else:
            return None
        if not self._is_allowed_media_path(path):
            self.logger.warning(
                "[Bot] Refused to send local file %r: it is not inside the manager's files_folder "
                "or the bot's media_dirs.", path,
            )
            raise PermissionError(
                f"Refusing to send {path!r}: outgoing files must be inside the manager's files_folder "
                "or one of the bot's media_dirs."
            )
        return path

    @contextlib.contextmanager
    def _media_input(self, media: Any):
        """Yield an open file for allowed local paths, or ``media`` unchanged for URLs and ids."""
        local_path = self._local_path(media)
        if local_path is None:
            yield media
        else:
            with open(local_path, "rb") as handle:
                yield handle

    # ------------------------------------------------------------------
    # Incoming media
    # ------------------------------------------------------------------

    def _check_media_size(self, size: Any) -> None:
        try:
            size = None if size is None else int(size)
        except (TypeError, ValueError):
            return
        if self.max_media_bytes is not None and size is not None and size > self.max_media_bytes:
            raise MediaTooLargeError(size, self.max_media_bytes)

    def _media_too_large_text(self, exc: MediaTooLargeError) -> Optional[str]:
        if not self.media_too_large_msg:
            return None
        limit = exc.limit or self.max_media_bytes
        limit_mb = f"{round(limit / (1024 * 1024), 1):g}" if limit else "?"
        try:
            return self.media_too_large_msg.format(limit_mb=limit_mb)
        except (KeyError, IndexError, ValueError):
            return self.media_too_large_msg

    @staticmethod
    def _media_extension(head: bytes, name_hint: Any = None, mime_hint: Any = None) -> str:
        kind = what(None, h=head) if head else None
        if kind:
            return f".{kind}"
        if head[:3] == b"ID3":
            return ".mp3"
        if head[:4] == b"OggS":
            return ".ogg"
        if name_hint:
            ext = os.path.splitext(str(name_hint))[1].lower()
            if _SAFE_EXTENSION_RE.fullmatch(ext):
                return ext
        if mime_hint:
            ext = (mimetypes.guess_extension(str(mime_hint).split(";")[0].strip().lower()) or "").lower()
            if _SAFE_EXTENSION_RE.fullmatch(ext):
                return ext
        return ".bin"

    def _store_downloaded_media(
        self,
        user_id: str,
        temp_path: str,
        *,
        name_hint: Any = None,
        mime_hint: Any = None,
    ) -> str:
        """Hand a downloaded temp file to ``manager.save_file`` without reading it into memory."""
        with open(temp_path, "rb") as handle:
            head = handle.read(32)
        named_path = temp_path + self._media_extension(head, name_hint, mime_hint)
        os.replace(temp_path, named_path)
        try:
            file_ref = self.manager.save_file(named_path, user_id)
        finally:
            with contextlib.suppress(OSError):
                os.remove(named_path)
        # save_file(path) caches the *source path*; that temp file is gone now, so drop the
        # entry and let the manager read the saved copy from disk when it needs it.
        cache = getattr(self.manager, "_file_cache", None)
        if isinstance(file_ref, str) and file_ref.startswith("file:") and cache is not None:
            with contextlib.suppress(Exception):
                cache.pop(file_ref[5:], None)
        return file_ref

    @staticmethod
    def _new_temp_file() -> str:
        fd, path = tempfile.mkstemp(prefix="mas-media-")
        os.close(fd)
        return path

    # ------------------------------------------------------------------
    # User message metadata
    # ------------------------------------------------------------------

    def _normalize_metadata_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if isinstance(value, dict):
            return {
                str(k): self._normalize_metadata_value(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [self._normalize_metadata_value(v) for v in value]

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return self._normalize_metadata_value(to_dict())
            except Exception:
                pass

        useful_attrs = {}
        for attr in (
            "file_id",
            "file_unique_id",
            "mime_type",
            "file_name",
            "file_size",
            "width",
            "height",
            "duration",
        ):
            if hasattr(value, attr):
                useful_attrs[attr] = self._normalize_metadata_value(getattr(value, attr))

        if useful_attrs:
            return useful_attrs

        return str(value)

    def _build_user_message_metadata(self, parsed_message: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "channel": self.__class__.__name__,
            "user_id": str(parsed_message.get("user_id")),
            "sender_id": self._sender_id(parsed_message),
            "message_type": parsed_message.get("message_type"),
            "timestamp": self._normalize_metadata_value(parsed_message.get("timestamp")),
            "is_voice_note": bool(parsed_message.get("is_voice_note", False)),
        }

        if parsed_message.get("message_id") is not None:
            metadata["message_id"] = str(parsed_message["message_id"])

        if parsed_message.get("chat_type"):
            metadata["chat_type"] = str(parsed_message["chat_type"])

        if parsed_message.get("media_info") is not None:
            metadata["media_info"] = self._normalize_metadata_value(parsed_message.get("media_info"))

        if parsed_message.get("metadata") is not None:
            metadata["metadata"] = self._normalize_metadata_value(parsed_message.get("metadata"))

        # The raw update holds names, usernames and phone numbers; it is persisted only on request.
        if self.store_original_payload and parsed_message.get("original_payload") is not None:
            metadata["original_payload"] = self._normalize_metadata_value(parsed_message.get("original_payload"))

        return metadata

    def _attach_user_message_metadata(self, blocks: List[Dict], parsed_message: Dict[str, Any]) -> List[Dict]:
        if not blocks:
            return blocks

        first = blocks[0]
        if not isinstance(first, dict):
            return blocks

        first.setdefault("metadata", {})
        first["metadata"]["user_message"] = self._build_user_message_metadata(parsed_message)
        return blocks

    # ------------------------------------------------------------------
    # Message age and duplicate deliveries
    # ------------------------------------------------------------------

    def _message_age_seconds(self, msg_dt) -> Optional[float]:
        if not isinstance(msg_dt, datetime):
            return None
        if msg_dt.tzinfo is None:
            msg_dt = msg_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - msg_dt).total_seconds()

    def _is_too_old(self, msg_dt, *, max_age=120) -> bool:
        if max_age is None:
            return False
        age = self._message_age_seconds(msg_dt)
        return age is not None and age > max_age

    def _is_duplicate(self, parsed_message: Dict[str, Any]) -> bool:
        """True when this platform message id was already seen by this process within ``dedup_ttl``."""
        if not self.dedup_ttl:
            return False
        message_id = parsed_message.get("message_id")
        if message_id in (None, ""):
            return False
        key = (str(parsed_message.get("user_id")), str(message_id))
        now = _time.monotonic()
        with self._seen_messages_guard:
            seen = self._seen_messages
            # The TTL is the same for every entry, so the oldest entries expire first.
            while seen:
                oldest_key, expires_at = next(iter(seen.items()))
                if expires_at > now:
                    break
                seen.popitem(last=False)
            if key in seen:
                return True
            seen[key] = now + float(self.dedup_ttl)
            while len(seen) > self.DEDUP_MAX_ENTRIES:
                seen.popitem(last=False)
        return False

    # ------------------------------------------------------------------
    # Per-user turns, admin maintenance and ordered delivery
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _user_turn(self, user_id: str):
        """Run one message of ``user_id`` at a time; later messages wait in arrival order.

        Different users never wait for each other. ``asyncio.Lock`` wakes waiters in FIFO
        order, and callers reach this point without suspending after the message arrives,
        so a user's messages are handled in the order they were received.
        """
        key = (asyncio.get_running_loop(), str(user_id))
        with self._user_locks_guard:
            entry = self._user_locks.get(key)
            if entry is None:
                entry = self._user_locks[key] = [asyncio.Lock(), 0]
            entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            with self._user_locks_guard:
                entry[1] -= 1
                if entry[1] <= 0 and self._user_locks.get(key) is entry:
                    del self._user_locks[key]

    @contextlib.asynccontextmanager
    async def _activity_slot(self):
        """Count this turn as in flight; while an admin reset is pending, wait for it first."""
        while True:
            with self._activity_guard:
                if self._maintenance_requests == 0:
                    self._active_turns += 1
                    break
            await asyncio.sleep(self._ACTIVITY_POLL_SECONDS)
        token = _ACTIVE_TURN.set(self)
        try:
            yield
        finally:
            _ACTIVE_TURN.reset(token)
            with self._activity_guard:
                self._active_turns -= 1

    async def _run_maintenance(self, operation: Callable[[], Any]) -> Tuple[bool, Any]:
        """Run ``operation`` once no other user's message is in flight.

        New messages are held from the moment this is called until it returns. Returns
        ``(True, result)``, or ``(False, number_still_running)`` when ``maintenance_timeout``
        passes first, in which case ``operation`` is not run.
        """
        with self._activity_guard:
            self._maintenance_requests += 1
        try:
            own_turn = 1 if _ACTIVE_TURN.get() is self else 0
            timeout = self.maintenance_timeout
            deadline = None if timeout is None else _time.monotonic() + timeout
            while True:
                with self._activity_guard:
                    others = self._active_turns - own_turn
                if others <= 0:
                    break
                if deadline is not None and _time.monotonic() >= deadline:
                    self.logger.warning(
                        "[Bot] Admin maintenance command gave up after %s s: %d message(s) from "
                        "other users were still being processed.", timeout, others,
                    )
                    return False, others
                await asyncio.sleep(self._ACTIVITY_POLL_SECONDS)
            return True, await asyncio.to_thread(operation)
        finally:
            with self._activity_guard:
                self._maintenance_requests -= 1

    def _enqueue_send(
        self,
        user_id: str,
        blocks: List[Dict],
        original_message: Dict[str, Any],
        *,
        log_failures: bool,
    ) -> "asyncio.Future":
        """Queue ``blocks`` behind the user's earlier outgoing messages. Must run on the loop."""
        loop = asyncio.get_running_loop()
        key = (loop, str(user_id))
        future = loop.create_future()

        def _on_done(fut, user_id=str(user_id)):
            if fut.cancelled():
                return
            exc = fut.exception()  # also marks the exception as retrieved
            if exc is not None and log_failures:
                self.logger.error(
                    "[Bot] Could not deliver a reply to user %s: %s: %s",
                    user_id, type(exc).__name__, exc,
                )

        future.add_done_callback(_on_done)
        with self._send_queues_guard:
            queue = self._send_queues.get(key)
            if queue is None:
                queue = self._send_queues[key] = _SendQueue()
            queue.items.append((blocks, original_message, future))
            queue.last = future
            if queue.worker is None:
                queue.worker = loop.create_task(self._send_worker(key, queue))
        return future

    async def _send_worker(self, key: Any, queue: _SendQueue) -> None:
        current = None
        try:
            while True:
                with self._send_queues_guard:
                    if not queue.items:
                        queue.worker = None
                        if self._send_queues.get(key) is queue:
                            del self._send_queues[key]
                        return
                    blocks, original_message, current = queue.items.popleft()
                if current.done():
                    continue
                try:
                    await self._send_blocks(key[1], blocks, original_message)
                except Exception as exc:
                    if not current.done():
                        current.set_exception(exc)
                else:
                    if not current.done():
                        current.set_result(None)
        except asyncio.CancelledError:
            with self._send_queues_guard:
                pending = [item[2] for item in queue.items]
                queue.items.clear()
                queue.worker = None
                if self._send_queues.get(key) is queue:
                    del self._send_queues[key]
            for fut in [current, *pending]:
                if fut is not None and not fut.done():
                    fut.cancel()
            raise

    async def _deliver(self, user_id: str, blocks: List[Dict], original_message: Dict[str, Any]) -> None:
        """Send ``blocks`` in order with the user's other replies and wait until they are out.

        Gives up waiting (the blocks stay queued) after ``send_queue_timeout`` seconds.
        Raises what ``_send_blocks`` raised.
        """
        if not blocks:
            return
        future = self._enqueue_send(user_id, blocks, original_message, log_failures=False)
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=self.send_queue_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(
                "[Bot] A reply to user %s is still queued after send_queue_timeout=%s s; it will "
                "be sent in order.", user_id, self.send_queue_timeout,
            )

            def _log_late_failure(fut):
                if not fut.cancelled() and fut.exception() is not None:
                    self.logger.error(
                        "[Bot] Could not deliver a reply to user %s: %s: %s",
                        user_id, type(fut.exception()).__name__, fut.exception(),
                    )

            future.add_done_callback(_log_late_failure)

    async def _wait_for_sends(self, user_id: str) -> bool:
        """Wait until every reply queued for ``user_id`` has been sent, or ``send_queue_timeout``."""
        key = (asyncio.get_running_loop(), str(user_id))
        with self._send_queues_guard:
            queue = self._send_queues.get(key)
            last = queue.last if queue is not None else None
        if last is None or last.done():
            return True
        try:
            await asyncio.wait_for(asyncio.shield(last), timeout=self.send_queue_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(
                "[Bot] Replies to user %s were still being sent after send_queue_timeout=%s s; "
                "moving on to their next message (replies stay in order).",
                user_id, self.send_queue_timeout,
            )
            return False
        except Exception:
            pass  # failed sends are logged where they fail
        return True

    def _enqueue_from_thread(self, user_id, blocks, original_message, done: "_futures.Future") -> None:
        try:
            future = self._enqueue_send(user_id, blocks, original_message, log_failures=True)
        except BaseException as exc:
            done.set_exception(exc)
            return

        def _relay(fut):
            if fut.cancelled():
                done.cancel()
            elif fut.exception() is not None:
                done.set_exception(fut.exception())
            else:
                done.set_result(None)

        future.add_done_callback(_relay)

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    async def process_payload(self, payload: Any) -> None:
        try:
            parsed_message = await self._parse_payload(payload)
        except Exception:
            self.logger.exception("[Bot] Could not parse an incoming payload; it was skipped.")
            return

        if not parsed_message:
            if self.verbose:
                self.logger.debug("[Bot] Payload ignored or not parseable.")
            return

        user_id = parsed_message['user_id']
        sender_id = self._sender_id(parsed_message)

        if self._is_duplicate(parsed_message):
            if self.verbose:
                self.logger.info(
                    "[Bot] Skipped a duplicate delivery of message %s from user %s.",
                    parsed_message.get("message_id"), user_id,
                )
            return

        if not self._is_user_allowed(sender_id, chat_id=user_id):
            self.logger.info(
                "[Bot] Ignored a message from user %s: not in allowed_user_ids.", sender_id
            )
            return

        # The age is measured now, when the message arrives, and not after it has waited for
        # this user's earlier messages: time spent queued behind a slow run must not get it dropped.
        if self._is_too_old(parsed_message.get('timestamp'), max_age=self.max_allowed_message_delay):
            age = self._message_age_seconds(parsed_message.get('timestamp'))
            self.logger.warning(
                "[Bot] Dropped a message from user %s: it was %.0f s old on arrival "
                "(max_allowed_message_delay=%s).",
                user_id, age, self.max_allowed_message_delay,
            )
            return

        async with self._user_turn(user_id):
            async with self._activity_slot():
                await self._run_turn(user_id, parsed_message)

    async def _run_turn(self, user_id: str, parsed_message: Dict[str, Any]) -> None:
        try:
            await self._process_parsed_message(user_id, parsed_message)
        except MediaTooLargeError as exc:
            self.logger.warning(
                "[Bot] Did not process a %s from user %s: %s.",
                parsed_message.get("message_type"), user_id, exc,
            )
            await self._send_notice(user_id, parsed_message, self._media_too_large_text(exc))
        except Exception:
            self.logger.exception("[Bot] Error while processing a message from user %s.", user_id)
            await self._send_error_reply(user_id, parsed_message)
        # The turn (and with it the user's next message) waits for this message's replies.
        await self._wait_for_sends(user_id)

    async def _send_notice(self, user_id: str, parsed_message: Dict[str, Any], text: Optional[str]) -> None:
        if not text:
            return
        try:
            await self._deliver(user_id, [{"type": "text", "content": {"response": text}}], parsed_message)
        except Exception:
            self.logger.exception("[Bot] Could not send a reply to user %s.", user_id)

    async def _send_error_reply(self, user_id: str, parsed_message: Dict[str, Any]) -> None:
        if not self.error_msg:
            return
        try:
            await self._deliver(
                user_id,
                [{"type": "text", "content": {"response": self.error_msg}}],
                parsed_message,
            )
        except Exception:
            self.logger.exception("[Bot] Could not send the error reply to user %s.", user_id)

    async def _process_parsed_message(self, user_id: str, parsed_message: Dict[str, Any]) -> None:
        if parsed_message.get('message_type') == 'text' and (parsed_message.get('text') or '').startswith('/'):
            is_command = await self._handle_command(user_id, parsed_message['text'], parsed_message)
            if is_command:
                return

        mas_blocks = await self._build_mas_blocks(parsed_message)

        if not mas_blocks:
            if self.verbose:
                self.logger.debug(f"[Bot] No blocks generated for message from {user_id}.")
            return

        loop = asyncio.get_running_loop()
        callback_params = {
            "user_id": user_id,
            "sender_id": self._sender_id(parsed_message),
            "original_payload": parsed_message['original_payload'],
            "event_loop": loop,
        }

        def _send_response_from_callback(response):
            if response is None:
                return
            blocks = self.manager._to_blocks(response, user_id=user_id)
            try:
                on_loop_thread = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_loop_thread = False
            if on_loop_thread:
                # Waiting here would block the loop that does the sending.
                self._enqueue_send(user_id, blocks, parsed_message, log_failures=True)
                return
            done = _futures.Future()
            # Callbacks run in the manager's worker thread. Every reply goes through the user's
            # send queue, so progress updates and the final answer arrive in the order produced.
            loop.call_soon_threadsafe(self._enqueue_from_thread, user_id, blocks, parsed_message, done)
            if not self.ensure_delivery:
                return
            try:
                done.result(timeout=self.delivery_timeout)
            except _futures.TimeoutError:
                self.logger.warning(
                    "[Bot] A reply to user %s was not sent within delivery_timeout=%s s; it stays "
                    "queued and will be sent in order.", user_id, self.delivery_timeout,
                )
            except (_futures.CancelledError, Exception):
                pass  # logged by the send queue

        def on_update_wrapper(messages, manager, params):
            if self.on_update_user_callback:
                response = self.manager._invoke_callback(
                    self.on_update_user_callback, messages, manager, params
                )
                _send_response_from_callback(response)

        def on_complete_wrapper(messages, manager, params):
            response = None
            if self.on_complete_user_callback:
                response = self.manager._invoke_callback(
                    self.on_complete_user_callback, messages, manager, params
                )
            elif messages:
                response = messages[-1].get("message")

            _send_response_from_callback(response)

        await asyncio.to_thread(
            self.manager.run,
            input=mas_blocks,
            component_name=self.component_name,
            user_id=user_id,
            role="user",
            verbose=self.verbose,
            blocking=True,
            on_update=on_update_wrapper,
            on_update_params=callback_params,
            on_complete=on_complete_wrapper,
            on_complete_params=callback_params,
            return_token_count=self.return_token_count
        )

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_command_name(command: Any) -> str:
        normalized = command.strip().lower() if isinstance(command, str) else None
        if not normalized or not _COMMAND_NAME_RE.fullmatch(normalized):
            raise ValueError(
                f"Invalid custom command name {command!r}: a command must be '/' followed by 1-32 "
                "letters, digits or underscores, for example '/my_cmd' (Telegram's rule, applied "
                "to every bot so the same commands work on Telegram and WhatsApp)."
            )
        return normalized

    def _register_commands(self, custom_commands: Optional[Union[Dict, List[Dict]]]):
        self.commands["/start"] = {
            "message": self.on_start_msg, "function": None,
            "description": "Starts the conversation.", "admin_only": False
        }
        self.commands["/clear"] = {
            "message": self.on_clear_msg, "function": self._clear_history_cmd,
            "description": "Clears your message history.", "admin_only": False
        }
        self.commands["/help"] = {
            "message": None, "function": self._generate_help_message,
            "description": "Shows this help message.", "admin_only": False
        }

        if self.manager.admin_user_id:
            self.commands["/clear_all_users"] = {
                "message": None, "function": self._clear_all_cmd,
                "description": "[Admin] Clears history for all users.", "admin_only": True
            }
            self.commands["/reset_system"] = {
                "message": None, "function": self._reset_system_cmd,
                "description": "[Admin] Resets the entire system (history and files).", "admin_only": True
            }
            self.commands["/logs"] = {
                 "message": None, "function": self._logs_command_handler,
                 "description": "[Admin] Requests log files.", "admin_only": True
            }

        if custom_commands:
            cmds = [custom_commands] if isinstance(custom_commands, dict) else custom_commands
            for cmd_def in cmds:
                if not isinstance(cmd_def, dict):
                    raise ValueError(f"Invalid custom command definition {cmd_def!r}: expected a dict.")
                command = self._validate_command_name(cmd_def.get("command"))

                func = cmd_def.get("function")
                if isinstance(func, str):
                    try:
                        func = self.manager._get_function_from_string(func)
                    except (ValueError, FileNotFoundError, AttributeError) as e:
                        self.logger.error(f"Could not resolve function for command {command}: {e}")
                        func = None

                self.commands[command] = {
                    "message": cmd_def.get("message"),
                    "function": func,
                    "description": cmd_def.get("description", "Custom command."),
                    "admin_only": cmd_def.get("admin_only", False)
                }

    async def _generate_help_message(self, **kwargs) -> str:
        lines = [self.on_help_msg, ""]
        for cmd, details in sorted(self.commands.items()):
            lines.append(f"{cmd} - {details['description']}")
        return "\n".join(lines)

    async def _clear_history_cmd(self, manager: AgentSystemManager, user_id: str, **kwargs) -> None:
        manager.clear_message_history(user_id)

    def _maintenance_timeout_text(self, still_running: int, outcome: str) -> str:
        return (
            f"Timed out after {self.maintenance_timeout:g} s waiting for {still_running} message(s) "
            f"from other users to finish. {outcome}; please try again."
        )

    async def _clear_all_cmd(self, manager: AgentSystemManager, **kwargs) -> str:
        completed, value = await self._run_maintenance(manager.clear_global_history)
        if not completed:
            return self._maintenance_timeout_text(value, "No history was cleared")
        return f"Histories for all users cleared ({value} DBs)."

    async def _reset_system_cmd(self, manager: AgentSystemManager, **kwargs) -> str:
        completed, value = await self._run_maintenance(manager.reset_system)
        if not completed:
            return self._maintenance_timeout_text(value, "Nothing was reset")
        return "System reset: all user histories and files have been deleted."

    async def _logs_command_handler(self, manager: AgentSystemManager, user_id: str, original_message: Dict[str, Any], **kwargs) -> Optional[str]:

        logs_dir = getattr(manager, "logs_folder", None)
        if not logs_dir or not manager._usage_logging_enabled:
            return "Usage logging is disabled."

        manager._refresh_cost_summary()

        files_to_send = []
        usage_path = os.path.join(logs_dir, "usage.log")
        summary_path = os.path.join(logs_dir, "summary.log")

        if os.path.isfile(usage_path):
            files_to_send.append({"path": usage_path, "filename": "usage.log"})
        if os.path.isfile(summary_path):
            files_to_send.append({"path": summary_path, "filename": "summary.json"})

        if not files_to_send:
            return "No log files found."

        try:
            sent_count = await self._send_log_files(user_id, files_to_send, original_message)
            if sent_count == 0 and len(files_to_send) > 0:
                 return "Failed to send log files."
            return None

        except Exception as e:
            self.logger.error(f"Error during log sending process for user {user_id}: {e}")
            return "An error occurred while trying to send the log files."

    async def _handle_command(self, user_id: str, text: str, original_message: Dict[str, Any]) -> bool:
        if not text or not text.startswith('/'):
            return False

        first_word = text.split(None, 1)[0]
        command_name, at_sign, addressee = first_word.partition("@")
        if at_sign and addressee:
            # In groups Telegram clients send "/command@BotName". A command for another bot in
            # the same group is not ours to answer.
            own_name = self._command_bot_name()
            if own_name and addressee.lower() != own_name.lower():
                if self.verbose:
                    self.logger.debug("[Bot] Ignored %s addressed to another bot.", command_name)
                return True

        command_str = command_name.lower()
        cmd_handler = self.commands.get(command_str)

        if not cmd_handler:
            if self.unknown_command_msg:
                response_blocks = self.manager._to_blocks(self.unknown_command_msg, user_id=user_id)
                await self._deliver(user_id, response_blocks, original_message)
            return True

        if cmd_handler.get("admin_only", False):
            if not self._is_admin(self._sender_id(original_message) or user_id):
                response_blocks = self.manager._to_blocks("Unauthorized.", user_id=user_id)
                await self._deliver(user_id, response_blocks, original_message)
                return True

        response_text = None
        func = cmd_handler.get("function")
        if func:
            try:
                sig = inspect.signature(func)
                params = {"user_id": user_id, "original_message": original_message}
                if 'manager' in sig.parameters:
                    params['manager'] = self.manager

                if asyncio.iscoroutinefunction(func):
                    func_result = await func(**params)
                else:
                    func_result = func(**params)

                if isinstance(func_result, str):
                    response_text = func_result

            except Exception as e:
                self.logger.exception(f"Error executing function for command {command_str}: {e}")
                response_text = f"An error occurred while running the command."

        if response_text is None and cmd_handler.get("message"):
            response_text = cmd_handler.get("message")

        if response_text:
            response_blocks = self.manager._to_blocks(response_text, user_id=user_id)
            await self._deliver(user_id, response_blocks, original_message)

        return True


    async def _build_mas_blocks(self, parsed_message: Dict[str, Any]) -> List[Dict]:
        user_id = parsed_message['user_id']
        msg_type = parsed_message['message_type']
        text = parsed_message.get('text')
        media_info = parsed_message.get('media_info')
        is_voice = parsed_message.get('is_voice_note', False)
        blocks = []

        transcript = None
        if msg_type == 'audio':
            audio_ref = await self._download_media_and_save(user_id, media_info)
            transcript = None
            if self.speech_to_text:
                stt_params = {
                    "manager": self.manager,
                    "file_path": audio_ref[5:],
                    "user_id": user_id,
                }
                # Other users' messages run on this event loop too. Functions that still rely on
                # the manager's current user get it set right before the call (no await in between);
                # new code should pass stt_params["user_id"] explicitly.
                self.manager.set_current_user(str(user_id))
                transcript = self.speech_to_text(stt_params)
                if inspect.isawaitable(transcript):
                    transcript = await transcript
            else:
                try:
                    # manager.stt is a blocking network call: keep it off the event loop so other
                    # users' messages keep flowing while it runs.
                    transcript = await asyncio.to_thread(
                        self.manager.stt,
                        audio_ref,
                        self.whisper_provider,
                        self.whisper_model,
                        user_id=user_id,
                    )
                except Exception as e:
                    self.logger.error(f"Error in automatic STT for {user_id}: {e}")

            if transcript:
                blocks.extend(self.manager._to_blocks({"response": transcript}, user_id=user_id))
            else:
                blocks.extend(self.manager._to_blocks({"response": "Could not transcribe audio."}, user_id=user_id))

            blocks.append({
                "type": "audio",
                "content": {
                    "kind": "file",
                    "path": audio_ref,
                    "detail": "auto",
                    "is_voice_note": is_voice
                }
            })

        elif msg_type == 'text' and text:
            blocks.extend(self.manager._to_blocks({"response": text}, user_id=user_id))

        elif msg_type == 'reaction':
            reaction_payload = {"response": text or ""}
            normalized_media_info = self._normalize_metadata_value(media_info)
            if isinstance(normalized_media_info, dict):
                reaction_payload.update({
                    k: v for k, v in normalized_media_info.items()
                    if k in ("message_id", "emoji")
                })
            blocks.extend(self.manager._to_blocks(reaction_payload, user_id=user_id))

        elif msg_type in ('image', 'video', 'document', 'sticker'):
            if text:
                 blocks.extend(self.manager._to_blocks({"response": text}, user_id=user_id))
            media_ref = await self._download_media_and_save(user_id, media_info)
            content = {"kind": "file", "path": media_ref, "detail": "auto"}
            normalized_media_info = self._normalize_metadata_value(media_info)
            if isinstance(normalized_media_info, dict):
                if normalized_media_info.get("filename"):
                    content["filename"] = normalized_media_info["filename"]
                if normalized_media_info.get("file_name"):
                    content["filename"] = normalized_media_info["file_name"]
                if normalized_media_info.get("mime_type"):
                    content["mime_type"] = normalized_media_info["mime_type"]
            blocks.append({
                "type": msg_type,
                "content": content
            })

        return self._attach_user_message_metadata(blocks, parsed_message)



class TelegramBot(Bot):
    def __init__(
        self,
        manager: AgentSystemManager,
        telegram_token: str = None,
        concurrent_updates: Union[bool, int] = True,
        **kwargs
    ):
        super().__init__(manager=manager, **kwargs)
        if Application is None or HTTPXRequest is None:
            raise ImportError("python-telegram-bot is required for TelegramBot: pip install 'python-telegram-bot>=22,<23'")

        token = telegram_token or self.manager.get_key("telegram_token")
        if not token:
            raise ValueError("Telegram token was not provided or found in the manager's API keys.")

        self.telegram_token = token
        self.concurrent_updates = concurrent_updates

        request = HTTPXRequest(
            # Concurrent updates send replies in parallel; a single pooled connection would
            # make them queue (and hit pool_timeout) behind each other.
            connection_pool_size=256,
            connect_timeout=15.0,
            read_timeout=60.0,
            write_timeout=60.0,
            pool_timeout=15.0,
            http_version="1.1",
        )

        builder = Application.builder().token(self.telegram_token).request(request)
        # PTB processes one update at a time by default, so one user's slow run would block
        # every other user. Updates run concurrently here (True = up to 256 at once); each
        # user's own messages are still handled one at a time, in order, by _user_turn.
        builder = builder.concurrent_updates(concurrent_updates)
        self.application = builder.build()
        self.bot = self.application.bot
        self._register_handlers()

        if self.verbose:
            self.logger.info("[TelegramBot] Instance of TelegramBot created.")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Only ids: the update itself holds the user's name, username and message text.
        self.logger.error(
            "[TelegramBot] python-telegram-bot reported an error (update_id=%s).",
            getattr(update, "update_id", None), exc_info=context.error,
        )

    def _register_handlers(self):
        # Every command, known or not, goes to _handle_command: it strips "@BotName", answers
        # unknown commands with unknown_command_msg and applies the admin check.
        self.application.add_handler(MessageHandler(filters.COMMAND, self.process_payload_wrapper))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.process_payload_wrapper))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.process_payload_wrapper))
        self.application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.process_payload_wrapper))
        for media_filter in self._optional_media_filters():
            self.application.add_handler(MessageHandler(media_filter, self.process_payload_wrapper))
        self.application.add_error_handler(self._on_error)

    def _optional_media_filters(self) -> List[Any]:
        filter_values = []
        seen = set()

        def add_filter(value):
            if value is None or id(value) in seen:
                return
            seen.add(id(value))
            filter_values.append(value)

        for attr in ("VIDEO", "DOCUMENT", "STICKER"):
            add_filter(getattr(filters, attr, None))

        for attr in ("Document", "Sticker"):
            namespace = getattr(filters, attr, None)
            add_filter(getattr(namespace, "ALL", None))

        return filter_values

    def _command_bot_name(self) -> Optional[str]:
        try:
            username = self.bot.username
        except Exception:
            return None  # not initialized yet: accept any "@name" suffix
        return str(username) if username else None

    async def initialize(self):
        await self.application.initialize()

    async def process_webhook_update(self, update_data: dict):
        update = Update.de_json(update_data, self.bot)
        await self.application.process_update(update)

    async def process_payload_wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await self.process_payload(update)
        except Exception as e:
            self.logger.exception("[TelegramBot] Error while processing payload", exc_info=e)
            raise

    async def _parse_payload(self, payload: Update) -> Optional[Dict[str, Any]]:
        if not payload.message:
            return None

        message = payload.message

        is_voice = message.voice is not None
        is_audio = message.audio is not None

        msg_type = None
        if message.text:
            msg_type = 'text'
        elif message.photo:
            msg_type = 'image'
        elif is_voice or is_audio:
            msg_type = 'audio'
        elif getattr(message, "video", None):
            msg_type = 'video'
        elif getattr(message, "document", None):
            msg_type = 'document'
        elif getattr(message, "sticker", None):
            msg_type = 'sticker'

        if not msg_type:
            return None

        media_info = None
        if message.photo:
            media_info = message.photo[-1]
        elif is_voice or is_audio:
            media_info = message.voice or message.audio
        elif msg_type == "video":
            media_info = message.video
        elif msg_type == "document":
            media_info = message.document
        elif msg_type == "sticker":
            media_info = message.sticker

        chat = message.chat
        # History is keyed by chat (a group shares one history); permissions use the sender.
        sender = getattr(getattr(message, "from_user", None), "id", None)
        return {
            'user_id': str(chat.id),
            'sender_id': str(sender) if sender is not None else str(chat.id),
            'message_id': getattr(message, "message_id", None),
            'chat_type': getattr(chat, "type", None),
            'message_type': msg_type,
            'text': message.text or message.caption,
            'media_info': media_info,
            'is_voice_note': is_voice,
            'timestamp': message.date,
            'original_payload': payload
        }

    async def _download_media_and_save(self, user_id: str, media_info: Any) -> str:
        expected_size = getattr(media_info, "file_size", None)
        self._check_media_size(expected_size)
        try:
            tg_file = await self.bot.get_file(media_info.file_id)
        except Exception as exc:
            # The cloud Bot API answers "File is too big" for anything over 20 MB.
            if "too big" in str(exc).lower():
                raise MediaTooLargeError(
                    expected_size, min(self.max_media_bytes or TELEGRAM_DOWNLOAD_LIMIT_BYTES, TELEGRAM_DOWNLOAD_LIMIT_BYTES)
                ) from exc
            raise
        self._check_media_size(getattr(tg_file, "file_size", None))

        temp_path = self._new_temp_file()
        try:
            await tg_file.download_to_drive(custom_path=temp_path)
            self._check_media_size(os.path.getsize(temp_path))
            return await asyncio.to_thread(
                self._store_downloaded_media,
                user_id,
                temp_path,
                name_hint=getattr(tg_file, "file_path", None) or getattr(media_info, "file_name", None),
                mime_hint=getattr(media_info, "mime_type", None),
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    async def _send_text_message(
        self,
        user_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        payload = {"chat_id": user_id, "text": text, **kwargs}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self.bot.send_message(**payload)

    async def _send_media(
        self,
        user_id: str,
        media_type: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        method_by_type = {
            "image": ("send_photo", "photo"),
            "audio": ("send_audio", "audio"),
            "video": ("send_video", "video"),
            "document": ("send_document", "document"),
            "sticker": ("send_sticker", "sticker"),
        }
        if media_type not in method_by_type:
            raise ValueError(f"Unsupported Telegram media type: {media_type}")

        if isinstance(media, dict):
            media = self._block_media_source(media)
        method_name, field_name = method_by_type[media_type]
        method = getattr(self.bot, method_name)
        payload = {"chat_id": user_id, **kwargs}
        caption_rest = ""
        if caption is not None and media_type != "sticker":
            caption, caption_rest = _split_text_once(str(caption), self.CAPTION_LIMIT)
            payload["caption"] = caption
        if filename is not None and media_type == "document":
            payload["filename"] = filename
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id

        with self._media_input(media) as value:
            payload[field_name] = value
            result = await method(**payload)
        if caption_rest.strip():
            await self.send_text(user_id, caption_rest)
        return result

    async def _send_reaction(
        self,
        user_id: str,
        message_id: str,
        emoji: str,
        **kwargs
    ) -> Any:
        reaction = [] if emoji == "" else [{"type": "emoji", "emoji": emoji}]
        if hasattr(self.bot, "set_message_reaction"):
            return await self.bot.set_message_reaction(
                chat_id=user_id,
                message_id=int(message_id),
                reaction=reaction,
                **kwargs
            )

        def do_request():
            payload = {
                "chat_id": user_id,
                "message_id": int(message_id),
                "reaction": reaction,
                **kwargs,
            }
            response = requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/setMessageReaction",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        return await asyncio.to_thread(do_request)

    async def _send_blocks(self, user_id: str, blocks: List[Dict], original_message: Dict[str, Any]):
        update = original_message.get('original_payload') if isinstance(original_message, dict) else None
        message = getattr(update, "message", None)
        if not blocks or message is None:
            return

        if self.verbose:
            self.logger.debug("[TelegramBot] Sending %d block(s) to %s.", len(blocks), user_id)

        for block in blocks:
            block_type = block.get("type") if isinstance(block, dict) else None
            try:
                if block_type == "text":
                    text_content = self.manager._block_to_plain_text(block)
                    for chunk in _split_text(text_content, self.TEXT_LIMIT):
                        await message.reply_text(chunk)
                elif block_type in _MEDIA_TYPES:
                    await self._reply_media_block(message, block)
            except Exception as exc:
                # Never log the block itself: it is the user's conversation.
                self.logger.error(
                    "[TelegramBot] Could not send a %s block to %s: %s: %s",
                    block_type, user_id, type(exc).__name__, exc,
                )

    async def _reply_media_block(self, message: Any, block: Dict[str, Any]) -> Any:
        block_type = block["type"]
        content = block.get("content") or {}
        media = self._block_media_source(content)
        if not media or isinstance(media, dict):
            raise ValueError(f"the {block_type} block has no path, url or id to send")

        if block_type == "audio" and self._is_voice_note_block(block):
            method = message.reply_voice
        else:
            method = getattr(message, {
                "image": "reply_photo",
                "audio": "reply_audio",
                "video": "reply_video",
                "document": "reply_document",
                "sticker": "reply_sticker",
            }[block_type])

        kwargs = {}
        caption_rest = ""
        caption = self._block_caption(block) if block_type != "sticker" else None
        if caption:
            caption, caption_rest = _split_text_once(caption, self.CAPTION_LIMIT)
            kwargs["caption"] = caption
        filename = self._block_filename(block)
        if filename is not None and block_type == "document":
            kwargs["filename"] = filename

        with self._media_input(media) as value:
            result = await method(value, **kwargs)
        for chunk in _split_text(caption_rest, self.TEXT_LIMIT):
            await message.reply_text(chunk)
        return result

    async def _send_log_files(self, user_id: str, files_to_send: List[Dict[str, str]], original_message: Dict[str, Any]) -> int:
        sent_count = 0
        for file_info in files_to_send:
            try:
                with open(file_info["path"], "rb") as f:
                    await self.bot.send_document(chat_id=user_id, document=f, filename=file_info["filename"])
                sent_count += 1
            except Exception as e:
                self.logger.error(f"Error sending log file {file_info['filename']} to Telegram user {user_id}: {e}")
        return sent_count

    def start_polling(self):
        if self.verbose:
            self.logger.info("[TelegramBot] Starting bot in polling mode...")
        self.application.run_polling()

class WhatsappBot(Bot):
    def __init__(
        self,
        manager: AgentSystemManager,
        whatsapp_token: str = None,
        phone_number_id: str = None,
        webhook_verify_token: str = None,
        app_secret: str = None,
        verify_signature: bool = True,
        api_version: Optional[str] = None,
        **kwargs
    ):
        super().__init__(manager=manager, **kwargs)

        self.whatsapp_token = whatsapp_token or self.manager.get_key("whatsapp_token")
        self.phone_number_id = phone_number_id or self.manager.get_key("whatsapp_phone_number_id")
        self.webhook_verify_token = webhook_verify_token or self.manager.get_key("webhook_verify_token") or self.manager.get_key("whatsapp_verify_token")
        self.app_secret = app_secret or self.manager.get_key("whatsapp_app_secret")
        self.verify_signature = verify_signature

        if not all([self.whatsapp_token, self.phone_number_id, self.webhook_verify_token]):
            raise ValueError(
                "Whatsapp credentials are missing. Provide whatsapp_token, "
                "phone_number_id, and webhook_verify_token."
            )

        self.api_version = self._normalize_api_version(
            api_version or self.manager.get_key("whatsapp_api_version") or DEFAULT_WHATSAPP_API_VERSION
        )
        self.graph_base_url = f"https://graph.facebook.com/{self.api_version}"
        self.graph_url = f"{self.graph_base_url}/{self.phone_number_id}"
        self.headers_json = {"Authorization": f"Bearer {self.whatsapp_token}", "Content-Type": "application/json"}
        self.headers_auth = {"Authorization": f"Bearer {self.whatsapp_token}"}
        self.persistent_loop = None

        if self.verbose:
            self.logger.info("[WhatsappBot] Instance of WhatsappBot created and configured.")

    @staticmethod
    def _normalize_api_version(value: Any) -> str:
        text = str(value).strip()
        if text and text[0] in "vV":
            text = text[1:]
        text = "v" + text
        if not _API_VERSION_RE.fullmatch(text):
            raise ValueError(f"Invalid WhatsApp Graph API version {value!r}; expected something like 'v23.0'.")
        return text

    def handle_webhook_verification(self, query_params: dict) -> tuple[str, int]:
        if query_params.get("hub.verify_token") == self.webhook_verify_token:
            challenge = query_params.get("hub.challenge", "")
            return challenge, 200
        return "Forbidden", 403

    def verify_webhook_signature(self, raw_body: bytes, signature_header: Optional[str]) -> bool:
        """Check Meta's X-Hub-Signature-256 header against the raw POST body."""
        if not self.app_secret or not signature_header:
            return False
        expected = "sha256=" + hmac.new(
            self.app_secret.encode("utf-8"), raw_body or b"", hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature_header.strip().encode("utf-8"), expected.encode("utf-8"))

    def _webhook_messages(self, update_data: Any) -> List[Dict[str, Any]]:
        messages = []
        if not isinstance(update_data, dict):
            self.logger.warning("[WhatsappBot] Ignored a webhook body that is not a JSON object.")
            return messages
        for entry in update_data.get("entry") or []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes") or []:
                if not isinstance(change, dict):
                    continue
                value = change.get("value") or {}
                if not isinstance(value, dict):
                    continue
                for msg in value.get("messages") or []:
                    if isinstance(msg, dict):
                        messages.append(msg)
        return messages

    async def process_webhook_update(self, update_data: dict):
        # One webhook can carry several messages. Each one is handled on its own: a failure in
        # one never drops the others, and different users are served concurrently while each
        # user's messages keep their order (process_payload serializes per user).
        messages = self._webhook_messages(update_data)
        if not messages:
            return
        results = await asyncio.gather(
            *(self.process_payload(msg) for msg in messages), return_exceptions=True
        )
        for msg, result in zip(messages, results):
            if isinstance(result, Exception):
                self.logger.error(
                    "[WhatsappBot] Error processing WhatsApp message %s from %s.",
                    msg.get("id"), msg.get("from"), exc_info=result,
                )


    async def _parse_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg_type = payload.get("type")
        sender = payload.get("from")
        if not msg_type or not sender:
            return None

        ts_raw = payload.get("timestamp")
        try:
            ts = datetime.fromtimestamp(int(ts_raw), timezone.utc) if ts_raw else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        text = None
        media_info = None
        is_voice = False

        if msg_type == "text":
            text = (payload.get("text") or {}).get("body")

        elif msg_type in ("audio", "voice", "image", "video", "document", "sticker"):
            if msg_type == "voice":
                media_payload = payload.get("voice") or payload.get("audio") or {}
                is_voice = True
            else:
                media_payload = payload.get(msg_type) or {}
                if msg_type == "audio":
                    is_voice = bool(media_payload.get("voice", False))

            if not media_payload.get("id"):
                return None
            media_info = {
                "id": media_payload.get("id"),
                "mime_type": media_payload.get("mime_type"),
                "sha256": media_payload.get("sha256"),
                "filename": media_payload.get("filename"),
            }
            media_info = {k: v for k, v in media_info.items() if v is not None}
            text = media_payload.get("caption")

        elif msg_type == "reaction":
            reaction = payload.get("reaction") or {}
            text = reaction.get("emoji")
            media_info = {
                "message_id": reaction.get("message_id"),
                "emoji": reaction.get("emoji"),
            }
            media_info = {k: v for k, v in media_info.items() if v is not None}

        else:
            return None

        normalized_type = "audio" if msg_type == "voice" else msg_type

        return {
            "user_id": sender,
            "sender_id": sender,
            "message_id": payload.get("id"),
            "message_type": normalized_type,
            "text": text,
            "media_info": media_info,
            "is_voice_note": is_voice,
            "timestamp": ts,
            "original_payload": payload,
        }


    async def _download_media_and_save(self, user_id: str, media_info: Any) -> str:
        media_id = media_info.get("id") if isinstance(media_info, dict) else media_info
        if not media_id:
            raise ValueError("WhatsApp media id is required to download media.")
        info = media_info if isinstance(media_info, dict) else {}

        def do_download():
            media_url_info = requests.get(
                f"{self.graph_base_url}/{media_id}",
                headers=self.headers_auth, timeout=30
            )
            media_url_info.raise_for_status()
            details = media_url_info.json() or {}
            media_url = details.get("url")
            if not media_url:
                raise ValueError("Could not obtain WhatsApp media URL.")
            self._check_media_size(details.get("file_size"))

            temp_path = self._new_temp_file()
            try:
                # Streamed to disk in chunks, never held in memory as a whole.
                with requests.get(media_url, headers=self.headers_auth, timeout=30, stream=True) as response:
                    response.raise_for_status()
                    self._check_media_size((getattr(response, "headers", None) or {}).get("Content-Length"))
                    received = 0
                    with open(temp_path, "wb") as out:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            self._check_media_size(received)
                            out.write(chunk)
                return self._store_downloaded_media(
                    user_id,
                    temp_path,
                    name_hint=info.get("filename"),
                    mime_hint=info.get("mime_type") or details.get("mime_type"),
                )
            finally:
                with contextlib.suppress(OSError):
                    os.remove(temp_path)

        return await asyncio.to_thread(do_download)

    async def _send_text_message(
        self,
        user_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        return await self._send_api_request(
            "text",
            to=user_id,
            body=text,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )

    async def _send_media(
        self,
        user_id: str,
        media_type: str,
        media: Any,
        *,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        if media_type not in _MEDIA_TYPES:
            raise ValueError(f"Unsupported WhatsApp media type: {media_type}")

        media_payload: Dict[str, Any]
        if isinstance(media, dict):
            source = self._block_media_source(media)
            if source is media:
                media_payload = dict(media)
            else:
                media_payload = await self._whatsapp_media_payload_from_source(source)
                for key in ("caption", "filename"):
                    if media.get(key) is not None:
                        media_payload[key] = media[key]
        else:
            media_payload = await self._whatsapp_media_payload_from_source(media)

        if caption is not None and media_type not in ("audio", "sticker"):
            media_payload["caption"] = caption
        if filename is not None and media_type in ("document", "video"):
            media_payload["filename"] = filename

        caption_rest = ""
        current_caption = media_payload.get("caption")
        if isinstance(current_caption, str) and _text_length(current_caption) > self.CAPTION_LIMIT:
            media_payload["caption"], caption_rest = _split_text_once(current_caption, self.CAPTION_LIMIT)

        result = await self._send_api_request(
            media_type,
            to=user_id,
            media_payload=media_payload,
            reply_to_message_id=reply_to_message_id,
            **kwargs
        )
        if caption_rest.strip():
            await self.send_text(user_id, caption_rest)
        return result

    async def _whatsapp_media_payload_from_source(self, media: Any) -> Dict[str, Any]:
        if isinstance(media, os.PathLike):
            media = os.fspath(media)
        media_value = str(media)
        local_path = self._local_path(media_value)
        if local_path:
            return {"id": await self._upload_media(local_path)}
        if media_value.startswith("http://") or media_value.startswith("https://"):
            return {"link": media_value}
        return {"id": media_value}

    async def _send_reaction(
        self,
        user_id: str,
        message_id: str,
        emoji: str,
        **kwargs
    ) -> Any:
        return await self._send_api_request(
            "reaction",
            to=user_id,
            reaction_payload={"message_id": message_id, "emoji": emoji},
            **kwargs
        )

    async def _send_blocks(self, user_id: str, blocks: List[Dict], original_message: Dict[str, Any]):

        if not blocks:
            return

        for block in blocks:
            block_type = block.get("type") if isinstance(block, dict) else None
            try:
                content = block.get("content", {})

                if block_type == "text":
                    text_content = self.manager._block_to_plain_text(block)
                    if text_content:
                        await self.send_text(user_id, text_content)

                elif block_type in _MEDIA_TYPES:
                    await self._send_media(
                        user_id,
                        block_type,
                        self._block_media_source(content),
                        caption=self._block_caption(block),
                        filename=self._block_filename(block),
                    )

            except Exception as exc:
                # Never log the block itself: it is the user's conversation.
                self.logger.error(
                    "[WhatsappBot] Could not send a %s block to %s: %s: %s",
                    block_type, user_id, type(exc).__name__, exc,
                )


    async def _send_document(self, to: str, path: str, filename: str = None, caption: str = None):
        media_id = await self._upload_media(path)

        doc_payload = {"id": media_id}
        if filename: doc_payload["filename"] = filename
        if caption: doc_payload["caption"] = caption

        return await self._send_api_request("document", to=to, media_payload=doc_payload)

    async def _send_api_request(
        self,
        msg_type: str,
        to: str,
        body: str = None,
        media_id: str = None,
        media_payload: dict = None,
        document_payload: dict = None,
        reaction_payload: dict = None,
        reply_to_message_id: Optional[str] = None,
        **kwargs
    ):
        """POST one message to the Cloud API. Raises ``WhatsappAPIError`` if it is rejected."""
        payload = {"messaging_product": "whatsapp", "to": to}
        if msg_type == "text":
            payload["type"] = "text"
            payload["text"] = {"body": body}
        elif msg_type in ("image", "audio", "video", "sticker"):
            payload["type"] = msg_type
            payload[msg_type] = media_payload or {"id": media_id}
        elif msg_type == "document":
            payload["type"] = "document"
            payload["document"] = media_payload or document_payload
        elif msg_type == "reaction":
            payload["type"] = "reaction"
            payload["reaction"] = reaction_payload
        else:
            raise ValueError(f"Unsupported WhatsApp message type: {msg_type}")

        if reply_to_message_id:
            payload["context"] = {"message_id": reply_to_message_id}
        payload.update(kwargs)

        resp = await asyncio.to_thread(
            requests.post, f"{self.graph_url}/messages", headers=self.headers_json, json=payload, timeout=60
        )

        if not resp.ok:
            raise WhatsappAPIError.from_response(msg_type, resp)
        if self.verbose:
            self.logger.debug("[WhatsappBot] Sent a %s message to %s (HTTP %s).", msg_type, to, resp.status_code)
        return resp

    async def _send_log_files(self, user_id: str, files_to_send: List[Dict[str, str]], original_message: Dict[str, Any]) -> int:
        sent_count = 0
        for file_info in files_to_send:
            try:
                await self._send_document(user_id, file_info["path"], filename=file_info["filename"])
                sent_count += 1
            except Exception as e:
                self.logger.error(f"Error sending log file {file_info['filename']} to WhatsApp user {user_id}: {e}")

        if sent_count > 0:
            await self._send_blocks(user_id, self.manager._to_blocks(f"Sent {sent_count} log file(s).", user_id=user_id), original_message)

        return sent_count

    async def _upload_media(self, path: str) -> str:
        """Upload a local file and return its media id. Raises ``WhatsappAPIError`` on failure."""
        def do_upload():
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            with open(path, "rb") as fh:
                files = {"file": (os.path.basename(path), fh, mime)}
                r = requests.post(f"{self.graph_url}/media", headers=self.headers_auth,
                                  data={"messaging_product": "whatsapp"}, files=files, timeout=60)
            if not r.ok:
                raise WhatsappAPIError.from_response("media upload", r)
            media_id = (r.json() or {}).get("id")
            if not media_id:
                raise WhatsappAPIError(
                    "WhatsApp Cloud API accepted the media upload but returned no media id",
                    status_code=getattr(r, "status_code", None),
                )
            return media_id
        return await asyncio.to_thread(do_upload)

    def create_flask_app(self, base_path: str = "/webhook"):
        if Flask is None:
            raise ImportError("Flask is required to run the WhatsApp webhook server: pip install 'Flask>=3,<4'")
        if self.verify_signature and not self.app_secret:
            raise ValueError(
                "WhatsApp webhook POSTs must be authenticated with your Meta app secret. "
                "Pass app_secret=... (or set WHATSAPP_APP_SECRET). To accept unsigned POSTs, "
                "which lets anyone impersonate any user, pass verify_signature=False."
            )

        app = Flask(__name__)
        @app.route(base_path, methods=["GET", "POST"])
        def webhook_handler():
            if request.method == "GET":
                body, status = self.handle_webhook_verification(request.args)
                return body, status
            elif request.method == "POST":
                if self.verify_signature and not self.verify_webhook_signature(
                    request.get_data(cache=True), request.headers.get("X-Hub-Signature-256")
                ):
                    self.logger.warning("[WhatsappBot] Rejected webhook POST with a missing or invalid X-Hub-Signature-256.")
                    return "Forbidden", 403
                data = request.get_json(force=True, silent=True) or {}
                asyncio.run_coroutine_threadsafe(
                    self.process_webhook_update(data), self.persistent_loop
                )
                return "OK", 200
        return app

    def run_server(self, host: str = "0.0.0.0", port: int = 5000, base_path: str = "/webhook"):
        app = self.create_flask_app(base_path)

        self.persistent_loop = asyncio.new_event_loop()

        def run_asyncio_loop():
            asyncio.set_event_loop(self.persistent_loop)
            self.persistent_loop.run_forever()

        loop_thread = threading.Thread(target=run_asyncio_loop, daemon=True)
        loop_thread.start()

        if self.verbose: self.logger.info(f"[WhatsappBot] Flask server running at http://{host}:{port}{base_path}")
        try:
            app.run(host=host, port=port)
        except Exception as e:
            self.logger.exception(f"Error starting Flask server: {e}")
        finally:
            if self.persistent_loop and self.persistent_loop.is_running():
                if self.verbose:
                    self.logger.info("[WhatsappBot] Stopping async event loop.")
                self.persistent_loop.call_soon_threadsafe(self.persistent_loop.stop)
