from __future__ import annotations

from ._shared import *

class Bot(abc.ABC):
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
        max_allowed_message_delay: float = 120.0
    ):
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

        self.logger = logging.getLogger(__name__)

        self._processing_users = set()
        self._user_lock = asyncio.Lock()


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


    def _is_too_old(self, msg_dt, *, max_age=120) -> bool:
        if not isinstance(msg_dt, datetime):
            return False
        if msg_dt.tzinfo is None:
            msg_dt = msg_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - msg_dt) > timedelta(seconds=max_age)

    async def process_payload(self, payload: Any) -> None:
        parsed_message = await self._parse_payload(payload)

        if not parsed_message:
            self.logger.debug("[Bot] Payload ignored or not parseable.")
            return

        user_id = parsed_message['user_id']

        async with self._user_lock:
            if user_id in self._processing_users:
                self.logger.debug(f"[Bot] User {user_id} is already being processed. Ignoring new message.")
                return
            self._processing_users.add(user_id)

        try:
            
            if self._is_too_old(parsed_message['timestamp'], max_age = self.max_allowed_message_delay):
                self.logger.debug(f"[Bot] Message from {user_id} ignored for being too old.")
                return
                
            if parsed_message.get('message_type') == 'text' and parsed_message.get('text', '').startswith('/'):
                is_command = await self._handle_command(user_id, parsed_message['text'], parsed_message)
                if is_command:
                    return

            mas_blocks = await self._build_mas_blocks(parsed_message)
            
            if not mas_blocks:
                if self.verbose:
                    self.logger.debug(f"[Bot] No blocks generated for message from {user_id}.")
                return
            
            callback_params = {
                "user_id": user_id,
                "original_payload": parsed_message['original_payload'],
                "event_loop": asyncio.get_running_loop()
            }

            def _send_response_from_callback(response):
                if response is not None:
                    blocks = self.manager._to_blocks(response, user_id=user_id)

                    loop = callback_params["event_loop"]
                    fut = asyncio.run_coroutine_threadsafe(
                    self._send_blocks(user_id, blocks, parsed_message), loop
                    )
                    if self.ensure_delivery:
                        try:
                            fut.result(timeout=self.delivery_timeout)
                        except Exception as e:
                            self.logger.exception(f"Bot delivery failed or timed out: {e}")

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
        finally:
            async with self._user_lock:
                if user_id in self._processing_users:
                    self._processing_users.remove(user_id)

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
                command = cmd_def.get("command")
                if not command or not isinstance(command, str) or not command.startswith('/'):
                    self.logger.warning(f"Invalid custom command definition skipped: {cmd_def}")
                    continue
                
                func = cmd_def.get("function")
                if isinstance(func, str):
                    try:
                        func = self.manager._get_function_from_string(func)
                    except (ValueError, FileNotFoundError, AttributeError) as e:
                        self.logger.error(f"Could not resolve function for command {command}: {e}")
                        func = None

                self.commands[command.lower().strip()] = {
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

    async def _clear_all_cmd(self, manager: AgentSystemManager, **kwargs) -> str:
        count = manager.clear_global_history()
        return f"Histories for all users cleared ({count} DBs)."

    async def _reset_system_cmd(self, manager: AgentSystemManager, **kwargs) -> str:
        manager.reset_system()
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

        command_str = text.split(' ', 1)[0].lower()
        cmd_handler = self.commands.get(command_str)

        if not cmd_handler:
            if self.unknown_command_msg:
                response_blocks = self.manager._to_blocks(self.unknown_command_msg, user_id=user_id)
                await self._send_blocks(user_id, response_blocks, original_message)
            return True
        
        if cmd_handler.get("admin_only", False):
            if not self.manager.admin_user_id or str(user_id) != str(self.manager.admin_user_id):
                response_blocks = self.manager._to_blocks("Unauthorized.", user_id=user_id)
                await self._send_blocks(user_id, response_blocks, original_message)
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
            await self._send_blocks(user_id, response_blocks, original_message)
            
        return True


    async def _build_mas_blocks(self, parsed_message: Dict[str, Any]) -> List[Dict]:
        user_id = parsed_message['user_id']
        msg_type = parsed_message['message_type']
        text = parsed_message.get('text')
        media_info = parsed_message.get('media_info')
        is_voice = parsed_message.get('is_voice_note', False)
        blocks = []

        try:
            self.manager.set_current_user(str(user_id))
        except Exception:
            pass

        transcript = None
        if msg_type == 'audio':
            audio_ref = await self._download_media_and_save(user_id, media_info)
            transcript = None
            if self.speech_to_text:
                stt_params = {
                    "manager": self.manager, 
                    "file_path": audio_ref[5:],
                }
                transcript = self.speech_to_text(stt_params)
            else:
                try:
                    transcript = self.manager.stt(audio_ref, self.whisper_provider, self.whisper_model, user_id=user_id)
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
        
        elif msg_type == 'image':
            if text:
                 blocks.extend(self.manager._to_blocks({"response": text}, user_id=user_id))
            img_ref = await self._download_media_and_save(user_id, media_info)
            blocks.append({
                "type": "image",
                "content": {"kind": "file", "path": img_ref, "detail": "auto"}
            })
            
        return blocks
    


class TelegramBot(Bot):
    def __init__(
        self,
        manager: AgentSystemManager,
        telegram_token: str = None,
        **kwargs
    ):
        super().__init__(manager=manager, **kwargs)
        if Application is None or HTTPXRequest is None:
            raise ImportError("python-telegram-bot is required for TelegramBot. Install mas[telegram].")
        
        token = telegram_token or self.manager.get_key("telegram_token")
        if not token:
            raise ValueError("Telegram token was not provided or found in the manager's API keys.")

        self.telegram_token = token

        request = HTTPXRequest(
            connect_timeout=15.0,
            read_timeout=60.0,
            write_timeout=60.0,
            pool_timeout=15.0,
            http_version="1.1",
        )


        self.application = Application.builder().token(self.telegram_token).request(request).build()
        self.bot = self.application.bot
        self._register_handlers()
        
        if self.verbose:
            self.logger.info("[TelegramBot] Instance of TelegramBot created.")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.logger.exception(
            "PTB caught an error. update=%r", getattr(update, "to_dict", lambda: update)(),
            exc_info=context.error
        )

    def _register_handlers(self):
        command_list = [cmd.lstrip('/') for cmd in self.commands.keys()]
        for command in command_list:
            self.application.add_handler(CommandHandler(command, self.process_payload_wrapper))

        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.process_payload_wrapper))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.process_payload_wrapper))
        self.application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.process_payload_wrapper))
        self.application.add_error_handler(self._on_error)

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
        
        if not msg_type:
            return None

        return {
            'user_id': str(message.chat.id),
            'message_type': msg_type,
            'text': message.text or message.caption,
            'media_info': message.photo[-1] if message.photo else message.voice or message.audio,
            'is_voice_note': is_voice,
            'timestamp': message.date,
            'original_payload': payload 
        }

    async def _download_media_and_save(self, user_id: str, media_info: Any) -> str:
        file = await self.bot.get_file(media_info.file_id)
        media_bytes = await file.download_as_bytearray()
        return self.manager.save_file(bytes(media_bytes), user_id)

    async def _send_blocks(self, user_id: str, blocks: List[Dict], original_message: Dict[str, Any]):
        update = original_message['original_payload']
        if not blocks or not update or not update.message:
            return

        self.logger.debug("[TelegramBot] _send_blocks: n=%d", len(blocks))

        for block in blocks:
            try:
                block_type = block.get("type")
                content = block.get("content", {})                
                if block_type == "text":
                    text_content = self.manager._block_to_plain_text(block)
                    if text_content:
                        await update.message.reply_text(text_content)
                
                elif block_type == "image" and "path" in content:
                    path = content["path"]
                    if path.startswith("file:"):
                        with open(path[5:], "rb") as f:
                            await update.message.reply_photo(f)
                    else:
                        await update.message.reply_photo(path)

                elif block_type == "audio" and "path" in content and content["path"].startswith("file:"):
                    with open(content["path"][5:], "rb") as f:
                        await update.message.reply_voice(f)
            except Exception as e:
                self.logger.error(f"Error sending block to Telegram for {user_id}: {e}\nBlock: {block}")

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
        **kwargs
    ):
        super().__init__(manager=manager, **kwargs)

        self.whatsapp_token = whatsapp_token or self.manager.get_key("whatsapp_token")
        self.phone_number_id = phone_number_id or self.manager.get_key("whatsapp_phone_number_id")
        self.webhook_verify_token = webhook_verify_token or self.manager.get_key("webhook_verify_token") or self.manager.get_key("whatsapp_verify_token")

        if not all([self.whatsapp_token, self.phone_number_id, self.webhook_verify_token]):
            raise ValueError(
                "Whatsapp credentials are missing. Provide whatsapp_token, "
                "phone_number_id, and webhook_verify_token."
            )

        self.api_version = "v18.0"
        self.graph_url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}"
        self.headers_json = {"Authorization": f"Bearer {self.whatsapp_token}", "Content-Type": "application/json"}
        self.headers_auth = {"Authorization": f"Bearer {self.whatsapp_token}"}
        self.persistent_loop = None

        if self.verbose:
            self.logger.info("[WhatsappBot] Instance of WhatsappBot created and configured.")

    def handle_webhook_verification(self, query_params: dict) -> tuple[str, int]:
        if query_params.get("hub.verify_token") == self.webhook_verify_token:
            challenge = query_params.get("hub.challenge", "")
            return challenge, 200
        return "Forbidden", 403
    
    async def process_webhook_update(self, update_data: dict):
        try:
            for entry in update_data.get("entry", []):
                for change in entry.get("changes", []):
                    for msg in change.get("value", {}).get("messages", []):
                        await self.process_payload(msg)
        except Exception as e:
            self.logger.exception(f"Error processing WhatsApp webhook payload: {e}")


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

        elif msg_type in ("audio", "voice", "image"):
            if msg_type == "audio":
                a = payload.get("audio") or {}
                media_info = a.get("id")
                is_voice = bool(a.get("voice", False))
                text = a.get("caption")

            elif msg_type == "voice":
                v = payload.get("voice") or payload.get("audio") or {}
                media_info = v.get("id")
                text = v.get("caption")
                is_voice = True

            elif msg_type == "image":
                img = payload.get("image") or {}
                media_info = img.get("id")
                text = img.get("caption")

            if msg_type in ("audio", "voice", "image") and not media_info:
                return None

        else:
            # Ignore other types for now: 'sticker', 'document', 'video', etc.
            return None

        normalized_type = "audio" if msg_type == "voice" else msg_type

        return {
            "user_id": sender,
            "message_type": normalized_type,
            "text": text,
            "media_info": media_info,
            "is_voice_note": is_voice,
            "timestamp": ts,
            "original_payload": payload,
        }


    async def _download_media_and_save(self, user_id: str, media_info: str) -> str:
        media_id = media_info
        
        def do_download():
            media_url_info = requests.get(
                f"https://graph.facebook.com/{self.api_version}/{media_id}",
                headers=self.headers_auth, timeout=30
            )
            media_url_info.raise_for_status()
            media_url = media_url_info.json().get("url")
            if not media_url:
                raise ValueError("Could not obtain WhatsApp media URL.")

            media_bytes_response = requests.get(media_url, headers=self.headers_auth, timeout=30)
            media_bytes_response.raise_for_status()
            return self.manager.save_file(media_bytes_response.content, user_id)
        
        return await asyncio.to_thread(do_download)

    async def _send_blocks(self, user_id: str, blocks: List[Dict], original_message: Dict[str, Any]):

        if not blocks:
            return

        for block in blocks:
            try:
                block_type = block.get("type")
                content = block.get("content", {})
                
                if block_type == "text":
                    text_content = self.manager._block_to_plain_text(block)
                    if text_content:
                        await self._send_api_request("text", to=user_id, body=text_content)
                
                elif block_type == "image" and content.get("path", "").startswith("file:"):
                    media_id = await self._upload_media(content["path"][5:])
                    if media_id:
                        await self._send_api_request("image", to=user_id, media_id=media_id)

                elif block_type == "audio" and content.get("path", "").startswith("file:"):
                    media_id = await self._upload_media(content["path"][5:])
                    if media_id:
                        await self._send_api_request("audio", to=user_id, media_id=media_id)

            except Exception as e:
                self.logger.error(f"Error sending block to WhatsApp for {user_id}: {e}\nBlock: {block}")


    async def _send_document(self, to: str, path: str, filename: str = None, caption: str = None):
        media_id = await self._upload_media(path)
        if not media_id: return
        
        doc_payload = {"id": media_id}
        if filename: doc_payload["filename"] = filename
        if caption: doc_payload["caption"] = caption
        
        await self._send_api_request("document", to=to, document_payload=doc_payload)

    async def _send_api_request(self, msg_type: str, to: str, body: str = None, media_id: str = None, document_payload: dict = None):
        payload = {"messaging_product": "whatsapp", "to": to}
        if msg_type == "text":
            payload["type"] = "text"
            payload["text"] = {"body": body}
        elif msg_type in ("image", "audio"):
            payload["type"] = msg_type
            payload[msg_type] = {"id": media_id}
        elif msg_type == "document":
            payload["type"] = "document"
            payload["document"] = document_payload
        
        resp = await asyncio.to_thread(
            requests.post, f"{self.graph_url}/messages", headers=self.headers_json, json=payload, timeout=60
        )

        if not resp.ok:
            short_payload = json.dumps(payload)[:1200]
            self.logger.error(
                "[WA SEND] FAILED %s -> %s | status=%s | resp=%s | payload=%s",
                msg_type, to, resp.status_code, resp.text, short_payload
            )
        else:
            self.logger.debug(
                "[WA SEND] OK %s -> %s | status=%s",
                msg_type, to, resp.status_code
            )

    async def _send_log_files(self, user_id: str, files_to_send: List[Dict[str, str]], original_message: Dict[str, Any]) -> int:
        sent_count = 0
        for file_info in files_to_send:
            try:
                await self._send_document(user_id, file_info["path"], filename=file_info["filename"])
                sent_count += 1
            except Exception as e:
                self.logger.error(f"Error sending log file {file_info['filename']} to WhatsApp user {user_id}: {e}")
        
        if sent_count > 0:
            await self._send_blocks(user_id, self.manager._to_blocks(f"Sent {sent_count} log file(s)."), original_message)
        
        return sent_count

    async def _upload_media(self, path: str) -> Optional[str]:
        def do_upload():
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            with open(path, "rb") as fh:
                files = {"file": (os.path.basename(path), fh, mime)}
                r = requests.post(f"{self.graph_url}/media", headers=self.headers_auth,
                                  data={"messaging_product": "whatsapp"}, files=files, timeout=60)
                if r.ok:
                    return r.json().get("id")
            self.logger.error("Failed to upload media to WhatsApp: %s", r.text)
            return None
        return await asyncio.to_thread(do_upload)
        
    def run_server(self, host: str = "0.0.0.0", port: int = 5000, base_path: str = "/webhook"):
        if Flask is None:
            raise ImportError("Flask is required to run the WhatsApp webhook server. Install mas[whatsapp].")

        self.persistent_loop = asyncio.new_event_loop()

        def run_asyncio_loop():
            asyncio.set_event_loop(self.persistent_loop)
            self.persistent_loop.run_forever()
        
        loop_thread = threading.Thread(target=run_asyncio_loop, daemon=True)
        loop_thread.start()

        app = Flask(__name__)
        @app.route(base_path, methods=["GET", "POST"])
        def webhook_handler():
            if request.method == "GET":
                body, status = self.handle_webhook_verification(request.args)
                return body, status
            elif request.method == "POST":
                data = request.get_json(force=True, silent=True) or {}
                asyncio.run_coroutine_threadsafe(
                    self.process_webhook_update(data), self.persistent_loop
                )
                return "OK", 200
        if self.verbose: self.logger.info(f"[WhatsappBot] Flask server running at http://{host}:{port}{base_path}")
        try:
            app.run(host=host, port=port)
        except Exception as e:
            self.logger.exception(f"Error starting Flask server: {e}")
        finally:
            if self.persistent_loop and self.persistent_loop.is_running():
                self.logger.info("[WhatsappBot] Stopping async event loop.")
                self.persistent_loop.call_soon_threadsafe(self.persistent_loop.stop)
