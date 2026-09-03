import os
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_notification(message: str, silent: bool = False) -> bool:
    """
    Send plain-text Telegram message.
    No Markdown (Markdown often breaks and Telegram rejects the message).
    """
    message = str(message or "").strip()
    if not message:
        print("⚠️ Empty notification message. Skipping.")
        return False

    # Keep under Telegram limit
    if len(message) > 4000:
        message = message[:3990] + "\n...[truncated]"

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram secrets missing.")
        print(f"   TELEGRAM_BOT_TOKEN set: {bool(TELEGRAM_BOT_TOKEN)}")
        print(f"   TELEGRAM_CHAT_ID set: {bool(TELEGRAM_CHAT_ID)}")
        print(f"📢 [LOCAL ONLY] {message[:300]}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_notification": silent,
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        print(f"📡 Telegram status code: {response.status_code}")
        print(f"📡 Telegram response: {response.text}")

        if response.status_code == 200:
            print("✅ Telegram notification sent successfully.")
            return True

        print("❌ Telegram API rejected the message.")
        return False

    except Exception as e:
        print(f"❌ Telegram send failed: {e}")
        return False


def send_agent_report(agent_name: str, status: str, details: str) -> bool:
    emoji = "✅" if status == "success" else "❌" if status == "error" else "⚠️"
    message = (
        f"{emoji} Agent Report: {agent_name}\n"
        f"----------------------------\n"
        f"Status: {status.upper()}\n"
        f"Details: {details}\n"
        f"----------------------------"
    )
    return send_notification(message)


if __name__ == "__main__":
    ok = send_notification("🚀 Test message from AI Agent Harness. If you see this, Telegram works!")
    print("RESULT:", ok)