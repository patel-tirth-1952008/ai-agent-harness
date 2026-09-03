import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── API KEYS ───
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN", "")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME") or os.getenv("GH_USERNAME", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── YOUR PROFILE ───
YOUR_NAME = os.getenv("YOUR_NAME", "Developer")
YOUR_SKILLS = os.getenv("YOUR_SKILLS", "Python, JavaScript")
YOUR_EXPERIENCE_YEARS = os.getenv("YOUR_EXPERIENCE_YEARS", "1")
YOUR_RESUME_SUMMARY = os.getenv("YOUR_RESUME_SUMMARY", "Software Developer")

# ─── LINKEDIN ───
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL", "")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD", "")

# ─── SAFETY LIMITS ───
MAX_API_CALLS_PER_DAY = 200
MAX_API_CALLS_PER_RUN = 40
MAX_AGENT_ITERATIONS = 8
MAX_AGENT_RPM = 15

# ─── DYNAMIC GROQ MODEL SELECTOR ───
def get_best_groq_model():
    """Queries Groq API to automatically pick the best active model."""
    api_key = GROQ_API_KEY
    if not api_key:
        return "llama-3.1-8b-instant"
    
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            active_models = [m["id"] for m in response.json().get("data", [])]
            print(f"🔍 Found active Groq models: {active_models}")

            # Standard Groq chat models compatible with LiteLLM
            preferred_order = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]

            for model in preferred_order:
                if model in active_models:
                    print(f"✅ Selected optimal model: {model}")
                    return model
            
            # Fallback to any active non-whisper model
            chat_models = [m for m in active_models if "whisper" not in m and "guard" not in m]
            if chat_models:
                return chat_models[0]
    except Exception as e:
        print(f"⚠️ Could not fetch Groq models dynamically: {e}")
        
    return "llama-3.1-8b-instant"

# Auto-detect best model at startup
GROQ_MODEL = get_best_groq_model()

# ─── SCHEDULE TIMES ───
PORTFOLIO_BUILD_DAY = "every_3_days"
PORTFOLIO_BUILD_TIME = "03:00"
LEETCODE_SOLVE_TIME = "02:00"
FREELANCE_SEARCH_TIME = "08:00"
JOB_SEARCH_TIME = "09:00"

# ─── DATA FILE PATHS ───
LEETCODE_QUEUE_FILE = "data/leetcode_queue.json"
JOB_HISTORY_FILE = "data/job_history.json"
FREELANCE_HISTORY_FILE = "data/freelance_history.json"
PORTFOLIO_HISTORY_FILE = "data/portfolio_history.json"