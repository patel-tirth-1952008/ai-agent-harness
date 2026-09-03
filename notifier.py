import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_notification(message: str, silent: bool = False):
    """Send a Telegram notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"📢 [NO TELEGRAM] {message}")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_notification": silent
        }
        response = requests.post(url, data=payload, timeout=10)

        if response.status_code == 200:
            print(f"📱 Notification sent successfully.")
            return True
        else:
            print(f"⚠️ Telegram error: {response.text}")
            return False

    except Exception as e:
        print(f"⚠️ Failed to send notification: {e}")
        return False


def send_agent_report(agent_name: str, status: str, details: str):
    """Send a formatted agent status report."""
    emoji = "✅" if status == "success" else "❌" if status == "error" else "⚠️"
    message = (
        f"{emoji} *Agent Report: {agent_name}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Status:* {status.upper()}\n"
        f"*Details:* {details}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    return send_notification(message)