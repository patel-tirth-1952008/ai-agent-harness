import os
import json
import time
import requests
import re
from datetime import datetime
import litellm
from dotenv import load_dotenv

load_dotenv()

# Force LiteLLM configuration & automatic retries on Rate Limits
litellm.drop_params = True
litellm.num_retries = 5
litellm.request_timeout = 120
os.environ["LITELLM_DROP_PARAMS"] = "true"


def _clean_messages(kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                msg.pop("cache_control", None)


_orig_completion = litellm.completion
_orig_acompletion = litellm.acompletion


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
from limiter import limiter
from notifier import send_notification, send_agent_report
from config import (
    GROQ_API_KEY, SERPER_API_KEY, GROQ_MODEL,
    YOUR_NAME, YOUR_SKILLS, YOUR_EXPERIENCE_YEARS,
    YOUR_RESUME_SUMMARY, MAX_AGENT_RPM, FREELANCE_HISTORY_FILE
)

CURRENT_YEAR = str(datetime.now().year)


# ─── COMPACT SEARCH TOOL (Saves 85% Tokens to avoid Groq 7k limit) ───
@tool("Search Freelance Gigs")
def search_freelance_gigs(query: str) -> str:
    """Searches Google for live freelance job postings and returns concise, clean summaries."""
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
            return "No recent gigs found for this query."
        
        results = []
        for item in organic[:4]:
            title = item.get("title", "No title")
            link = item.get("link", "")
            snippet = item.get("snippet", "")[:180]
            results.append(f"• Title: {title}\n  Link: {link}\n  Details: {snippet}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Search error: {str(e)}"


def get_llm():
    model_name = GROQ_MODEL if GROQ_MODEL.startswith("groq/") else f"groq/{GROQ_MODEL}"
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.2
    )


def load_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(FREELANCE_HISTORY_FILE):
        try:
            with open(FREELANCE_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "found_gigs" in data:
                    if "seen_urls" not in data:
                        data["seen_urls"] = []
                    return data
        except Exception:
            pass
    return {"found_gigs": [], "seen_urls": [], "last_search": None}


def save_history(history):
    os.makedirs("data", exist_ok=True)
    with open(FREELANCE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_proposals_to_file(result_text):
    os.makedirs("data/proposals", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filepath = f"data/proposals/proposals_{timestamp}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Freelance Proposals — {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write(result_text)
    print(f"💾 Proposals saved to {filepath}")
    return filepath


def run_freelance_finder():
    print("\n" + "=" * 60)
    print("  AGENT 3: FREELANCE FINDER")
    print("=" * 60)

    limiter.check()
    history = load_history()
    llm = get_llm()

    skills_list = [s.strip() for s in YOUR_SKILLS.split(",")]
    skill1 = skills_list[0] if len(skills_list) > 0 else "Python"
    skill2 = skills_list[1] if len(skills_list) > 1 else skill1

    gig_hunter = Agent(
        role="Freelance Opportunity Scout",
        goal="Identify active, high-match freelance listings using targeted searches.",
        backstory="Expert business developer finding remote contracts on Upwork, Freelancer, and LinkedIn.",
        tools=[search_freelance_gigs],
        llm=llm,
        max_iter=3,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    proposal_writer = Agent(
        role="Freelance Proposal Specialist",
        goal="Write concise, winning project proposals under 150 words each.",
        backstory="Crafts compelling, problem-first proposals that get high response rates from clients.",
        llm=llm,
        max_iter=2,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    search_task = Task(
        description=(
            f"Search for 4 active freelance gigs matching: {skill1} and {skill2}.\n"
            f"Run searches: 'upwork {skill1} developer {CURRENT_YEAR}' and 'freelance {skill2} contract remote'.\n\n"
            "For each gig, extract:\n"
            "- Title\n"
            "- Platform (Upwork, Freelancer, etc.)\n"
            "- Direct URL\n"
            "- 1-sentence summary\n"
            "- Match score (1-10)"
        ),
        expected_output="List of 3-4 active freelance opportunities with title, platform, link, and score.",
        agent=gig_hunter
    )

    proposal_task = Task(
        description=(
            f"From the opportunities found, write 2 tailored proposals for developer {YOUR_NAME} "
            f"(Skills: {YOUR_SKILLS}, Summary: {YOUR_RESUME_SUMMARY}).\n\n"
            "Format each proposal:\n"
            "## [Job Title]\n"
            "**Link:** [URL]\n"
            "**Pitch:** [2-3 sentences showing understanding of problem and solution]\n"
            "**Action Plan:**\n"
            "1. [Step 1]\n"
            "2. [Step 2]\n"
            "**Closing Question:** [1 relevant question to start conversation]\n\n"
            "Keep each proposal under 150 words."
        ),
        expected_output="2 concise, customized freelance proposals ready to send.",
        agent=proposal_writer
    )

    try:
        limiter.check()

        crew = Crew(
            agents=[gig_hunter, proposal_writer],
            tasks=[search_task, proposal_task],
            process=Process.sequential,
            verbose=True
        )

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
            send_agent_report("Freelance Finder", "error", "Agent returned empty response after retries.")
            return None

        seen_urls = set(history.get("seen_urls", []))
        found_urls = re.findall(r'https?://[^\s\)]+', result)
        new_urls = [u for u in found_urls if u not in seen_urls]
        seen_urls.update(found_urls)
        history["seen_urls"] = list(seen_urls)[-300:]

        history["found_gigs"].append({
            "date": datetime.now().isoformat(),
            "new_gigs_count": len(new_urls),
            "results": result[:2000]
        })
        history["last_search"] = datetime.now().isoformat()
        if len(history["found_gigs"]) > 30:
            history["found_gigs"] = history["found_gigs"][-30:]
        save_history(history)

        proposal_file = save_proposals_to_file(result)

        short_result = result[:3000] if len(result) > 3000 else result
        notification = (
            "💼 FREELANCE GIGS & PROPOSALS\n"
            "========================\n\n"
            + short_result + "\n\n"
            "========================\n"
            f"📁 Saved to: {proposal_file}\n"
            f"🔗 New Unique Gigs Found: {len(new_urls)}"
        )
        send_notification(notification)

        send_agent_report(
            "Freelance Finder", "success",
            f"Found {len(new_urls)} new gigs and generated proposals. Check Telegram!"
        )
        print("✅ Freelance Finder finished successfully.")
        return result

    except Exception as e:
        err = str(e)
        if "LIMIT REACHED" in err:
            send_agent_report("Freelance Finder", "warning", err)
        else:
            send_agent_report("Freelance Finder", "error", err)
        print(f"  Error: {e}")
        return None


if __name__ == "__main__":
    run_freelance_finder()