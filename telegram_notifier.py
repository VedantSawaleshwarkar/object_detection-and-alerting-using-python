import os
import requests
from typing import Optional

# Load configuration from environment variables or config.py
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
except Exception:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")

API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _get_token_and_chat():
    token = "8164979830:AAGMPFSvc-yGPfUTxIfFG2AV_70IlrQ9yCk"
    chat_id = "7361910235"
    if not token or not chat_id:
        raise RuntimeError(
            "Telegram configuration missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID in config.py or as environment variables.")
    return token, chat_id


def send_text_message(text: str, parse_mode: Optional[str] = None) -> bool:
    token, chat_id = _get_token_and_chat()
    url = API_BASE.format(token=token, method="sendMessage")
    data = {"chat_id": chat_id, "text": text}
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, data=data, timeout=20)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[Telegram] Failed to send message: {e}")
        return False


def send_video(video_path: str, caption: Optional[str] = None) -> bool:
    token, chat_id = _get_token_and_chat()
    url = API_BASE.format(token=token, method="sendVideo")
    files = {"video": open(video_path, "rb")}
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    try:
        resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[Telegram] Failed to send video: {e}")
        return False
    finally:
        try:
            files["video"].close()
        except Exception:
            pass
