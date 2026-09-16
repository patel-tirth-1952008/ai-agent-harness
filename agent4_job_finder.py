import os
import json
import time
import requests
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Set env vars BEFORE importing tools & crewai
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ─── 1. CRITICAL FIX FOR GROQ 'cache_breakpoint' & PARAMS ───
import litellm
litellm.drop_params = True
litellm.modify_params = True

_orig_completion = litellm.completion

def _safe_groq_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                msg.pop("cache_control", None)
    return _orig_completion(*args, **kwargs)

litellm.completion = _safe_groq_completion
# ─────────────────────────────────────────────────────────────

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from limiter import limiter
from notifier import send_notification, send_agent_report
from config import (
    GROQ_API_KEY, SERPER_API_KEY, GROQ_MODEL,
    YOUR_NAME, YOUR_SKILLS, YOUR_EXPERIENCE_YEARS,
    YOUR_RESUME_SUMMARY, MAX_AGENT_RPM, JOB_HISTORY_FILE
)

CURRENT_YEAR = str(datetime.now().year)


# ─── 2. COMPACT SEARCH TOOL (Saves 85% Tokens) ───
@tool("Search Tech Jobs")
def search_tech_jobs(query: str) -> str:
    """Searches Google for live software engineering job postings and returns concise, clean summaries."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY not set."
    
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "num": 4})
    
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=12)
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
    # Use gpt-oss-20b or gpt-oss-120b to avoid the 1k OTPM throttle on qwen
    model = GROQ_MODEL
    if "qwen" in model.lower():
        model = "openai/gpt-oss-20b"
    model_name = model if model.startswith("groq/") else f"groq/{model}"
    
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=750
    )


def load_history():
    """Self-healing history loader."""
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
    """Saves tailored cover letters and interview prep to disk."""
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

    # ─── AGENT 1: JOB SCOUT ───
    job_scout = Agent(
        role="Senior Technical Recruiter",
        goal="Find active software engineering job openings matching candidate skills.",
        backstory="Technical talent sourcer scouting LinkedIn, Indeed, Glassdoor, and startup boards for remote developer positions.",
        tools=[search_tech_jobs],
        llm=llm,
        max_iter=3,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── AGENT 2: APPLICATION PREP COACH ───
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

        # ─── 3. RETRY LOOP FOR KICKOFF ───
        result = None
        for attempt in range(3):
            try:
                result = str(crew.kickoff())
                break
            except Exception as crew_err:
                err_msg = str(crew_err).lower()
                if "429" in err_msg or "rate" in err_msg or "limit" in err_msg:
                    wait_time = 15 * (attempt + 1)
                    print(f"⏳ Rate limit hit. Retrying in {wait_time}s (Attempt {attempt+1}/3)...")
                    time.sleep(wait_time)
                else:
                    raise crew_err

        if not result or len(result) < 50:
            send_agent_report("Job Finder", "error", "Agent returned empty response after retries.")
            return None

        # ─── 4. UNIQUE JOB TRACKING (DEDUPLICATION) ───
        seen_urls = set(history.get("seen_urls", []))
        found_urls = re.findall(r'https?://[^\s\)]+', result)
        new_urls = [u for u in found_urls if u not in seen_urls]
        seen_urls.update(found_urls)
        history["seen_urls"] = list(seen_urls)[-300:]

        # Save to history & disk
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

        # Send Telegram notification
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


if __name__ == "__main__":
    run_job_finder()