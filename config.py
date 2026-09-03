import os
from dotenv import load_dotenv

load_dotenv()

# ─── API KEYS ───
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ─── YOUR PROFILE ───
YOUR_NAME = os.getenv("YOUR_NAME", "Developer")
YOUR_SKILLS = os.getenv("YOUR_SKILLS", "Python, JavaScript")
YOUR_EXPERIENCE_YEARS = os.getenv("YOUR_EXPERIENCE_YEARS", "1")
YOUR_RESUME_SUMMARY = os.getenv("YOUR_RESUME_SUMMARY", "Software Developer")

# ─── LINKEDIN ───
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL", "")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD", "")

# ─── SAFETY LIMITS ───
MAX_API_CALLS_PER_DAY = 100
MAX_API_CALLS_PER_RUN = 15
MAX_AGENT_ITERATIONS = 5
MAX_AGENT_RPM = 10

# ─── GROQ MODEL ───
GROQ_MODEL = "llama-3.1-8b-instant"

# ─── SCHEDULE TIMES ───
PORTFOLIO_BUILD_DAY = "monday"
PORTFOLIO_BUILD_TIME = "03:00"
LEETCODE_SOLVE_TIME = "02:00"
FREELANCE_SEARCH_TIME = "08:00"
JOB_SEARCH_TIME = "09:00"

# ─── DATA FILE PATHS ───
LEETCODE_QUEUE_FILE = "data/leetcode_queue.json"
JOB_HISTORY_FILE = "data/job_history.json"
FREELANCE_HISTORY_FILE = "data/freelance_history.json"
PORTFOLIO_HISTORY_FILE = "data/portfolio_history.json"