import os
import json
import time
import requests
import re
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Force LiteLLM configuration & automatic retries on Rate Limits
import litellm
litellm.drop_params = True
litellm.num_retries = 5
litellm.request_timeout = 120
os.environ["LITELLM_DROP_PARAMS"] = "true"

_orig_completion = litellm.completion
_orig_acompletion = litellm.acompletion


def _clean_messages(kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        cleaned = []
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                cleaned.append({k: v for k, v in msg.items() if k not in ["cache_breakpoint", "cache_control"]})
            else:
                cleaned.append(msg)
        kwargs["messages"] = cleaned


def _patched_completion(*args, **kwargs):
    _clean_messages(kwargs)
    return _orig_completion(*args, **kwargs)


async def _patched_acompletion(*args, **kwargs):
    _clean_messages(kwargs)
    return await _orig_acompletion(*args, **kwargs)


litellm.completion = _patched_completion
litellm.acompletion = _patched_acompletion

# Set env vars BEFORE importing tools & crewai
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Fallback-safe configuration imports
try:
    from limiter import limiter
except ImportError:
    class DummyLimiter:
        def check(self): pass
    limiter = DummyLimiter()

try:
    from notifier import send_notification, send_agent_report
except ImportError:
    def send_notification(msg): print(f"[Notification] {msg[:100]}...")
    def send_agent_report(agent, status, msg): print(f"[{agent} - {status}] {msg}")

try:
    from config import (
        GROQ_API_KEY, SERPER_API_KEY, GROQ_MODEL,
        YOUR_NAME, YOUR_SKILLS, YOUR_EXPERIENCE_YEARS,
        YOUR_RESUME_SUMMARY, MAX_AGENT_RPM, JOB_HISTORY_FILE
    )
except ImportError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
    GROQ_MODEL = "qwen/qwen3.8-27b"
    YOUR_NAME = "Full Stack Engineer"
    YOUR_SKILLS = "Python, FastAPI, Next.js, TypeScript, Docker"
    YOUR_EXPERIENCE_YEARS = "3"
    YOUR_RESUME_SUMMARY = "Full-Stack Software Engineer building scalable microservices and dynamic web apps."
    MAX_AGENT_RPM = 2
    JOB_HISTORY_FILE = "data/job_history.json"

CURRENT_YEAR = str(datetime.now().year)


# ─── COMPACT SEARCH TOOL (Saves Tokens to avoid Groq 7k limit) ───
@tool("Search Tech Jobs")
def search_tech_jobs(query: str) -> str:
    """Searches Google for live software engineering job postings and returns concise, clean summaries."""
    api_key = os.getenv("SERPER_API_KEY", "") or SERPER_API_KEY
    if not api_key:
        return "Error: SERPER_API_KEY not set."
    
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "num": 4})
    
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=15)
        if resp.status_code != 200:
            return f"Search error: HTTP {resp.status_code}"
        
        data = resp.json()
        organic = data.get("organic", [])
        if not organic:
            return "No recent job listings found for this query."
        
        results = []
        for item in organic[:4]:
            title = item.get("title", "No title")
            link = item.get("link", "")
            snippet = item.get("snippet", "")[:180]
            results.append(f"• Job: {title}\n  Link: {link}\n  Details: {snippet}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Search error: {str(e)}"


def get_llm():
    model = GROQ_MODEL or "qwen/qwen3.8-27b"
    model_name = model if model.startswith("groq/") else f"groq/{model}"
    
    return LLM(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY") or GROQ_API_KEY,
        temperature=0.2,
        max_tokens=750
    )


def load_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(JOB_HISTORY_FILE):
        try:
            with open(JOB_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "found_jobs" in data:
                    if "seen_urls" not in data:
                        data["seen_urls"] = []
                    return data
        except Exception as e:
            print(f"⚠️ Warning: Corrupted job history file ({e}). Resetting.")
    return {"found_jobs": [], "applied_jobs": [], "seen_urls": [], "last_search": None}


def save_history(history):
    os.makedirs("data", exist_ok=True)
    with open(JOB_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_applications_to_file(result_text):
    os.makedirs("data/applications", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filepath = f"data/applications/job_prep_{timestamp}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Job Application Package — {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write(result_text)
    print(f"💾 Application package saved to {filepath}")
    return filepath


def run_job_finder():
    print("\n" + "=" * 60)
    print("  AGENT 4: JOB FINDER & APPLICATION PREP")
    print("=" * 60)

    limiter.check()
    history = load_history()
    llm = get_llm()

    skills_list = [s.strip() for s in YOUR_SKILLS.split(",")]
    skill1 = skills_list[0] if len(skills_list) > 0 else "Python"
    skill2 = skills_list[1] if len(skills_list) > 1 else skill1

    job_scout = Agent(
        role="Senior Technical Recruiter",
        goal="Find active software engineering job openings matching candidate skills.",
        backstory="Technical talent sourcer scouting LinkedIn, Indeed, Glassdoor, and startup boards for remote developer positions.",
        tools=[search_tech_jobs],
        llm=llm,
        max_iter=2,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    app_prep = Agent(
        role="Career Coach and ATS Specialist",
        goal="Write tailored, high-converting cover letters and interview prep packages under 180 words each.",
        backstory="Expert at passing ATS filters and preparing candidates with high-impact talking points.",
        llm=llm,
        max_iter=2,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    search_task = Task(
        description=(
            f"Search for 4 active software developer jobs matching: {skill1} and {skill2}.\n"
            f"Run search: 'linkedin software engineer {skill1} remote {CURRENT_YEAR}'.\n\n"
            "For each job opening, extract:\n"
            "- Role Title & Company\n"
            "- Direct Link\n"
            "- Location / Remote status\n"
            "- Key Tech Stack\n"
            "- Match Score (1-10)"
        ),
        expected_output="List of 3-4 active software engineering jobs with title, company, URL, and match score.",
        agent=job_scout
    )

    app_task = Task(
        description=(
            f"For the top 2 jobs found, prepare complete application packages for {YOUR_NAME} "
            f"(Skills: {YOUR_SKILLS}, Experience: {YOUR_EXPERIENCE_YEARS} years, Summary: {YOUR_RESUME_SUMMARY}).\n\n"
            "Format each job package:\n"
            "## [Job Title] at [Company]\n"
            "**Link:** [URL]\n\n"
            "### 1. Tailored Cover Letter (under 150 words)\n"
            "- Hook addressing the role\n"
            "- Proof of skills ({skill1}, {skill2})\n"
            "- Confident call to action\n\n"
            "### 2. ATS Resume Keywords\n"
            "- [5 essential keywords]\n\n"
            "### 3. Interview Talking Points\n"
            "- 2 likely questions + key talking points\n\n"
            "Keep each package concise, punchy, and human-written."
        ),
        expected_output="2 complete application prep packages with cover letter, ATS keywords, and interview notes.",
        agent=app_prep
    )

    try:
        limiter.check()

        crew = Crew(
            agents=[job_scout, app_prep],
            tasks=[search_task, app_task],
            process=Process.sequential,
            verbose=True
        )

        result = None
        for attempt in range(1, 4):
            try:
                result = str(crew.kickoff())
                break
            except Exception as crew_err:
                err_msg = str(crew_err).lower()
                print(f"⚠️ Attempt {attempt}/3 error: {err_msg[:200]}")
                if "429" in err_msg or "rate" in err_msg or "limit" in err_msg:
                    wait_time = 65 * attempt
                    print(f"⏳ Rate limit hit. Backing off for {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(15)

        if not result or len(result) < 50:
            send_agent_report("Job Finder", "error", "Agent returned empty response after retries.")
            return None

        seen_urls = set(history.get("seen_urls", []))
        found_urls = re.findall(r'https?://[^\s\)]+', result)
        new_urls = [u for u in found_urls if u not in seen_urls]
        seen_urls.update(found_urls)
        history["seen_urls"] = list(seen_urls)[-300:]

        history["found_jobs"].append({
            "date": datetime.now().isoformat(),
            "new_jobs_count": len(new_urls),
            "results": result[:2000]
        })
        history["last_search"] = datetime.now().isoformat()
        if len(history["found_jobs"]) > 30:
            history["found_jobs"] = history["found_jobs"][-30:]
        save_history(history)

        prep_file = save_applications_to_file(result)

        short_result = result[:3000] if len(result) > 3000 else result
        notification = (
            "🎯 JOB MATCHES & APPLICATION PACKAGES\n"
            "========================\n\n"
            + short_result + "\n\n"
            "========================\n"
            f"📁 Saved to: {prep_file}\n"
            f"🔗 New Unique Openings: {len(new_urls)}"
        )
        send_notification(notification)

        send_agent_report(
            "Job Finder", "success",
            f"Found {len(new_urls)} new jobs and prepared application packages. Check Telegram!"
        )
        print("✅ Job Finder finished successfully.")
        return result

    except Exception as e:
        err = str(e)
        if "LIMIT REACHED" in err:
            send_agent_report("Job Finder", "warning", err)
        else:
            send_agent_report("Job Finder", "error", err)
        print(f"  Error: {e}")
        return None


main = run_job_finder

if __name__ == "__main__":
    run_job_finder()