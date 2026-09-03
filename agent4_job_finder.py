import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Set env vars BEFORE importing tools
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from limiter import limiter
from notifier import send_notification, send_agent_report
from config import (
    GROQ_API_KEY, SERPER_API_KEY, GROQ_MODEL,
    YOUR_NAME, YOUR_SKILLS, YOUR_EXPERIENCE_YEARS,
    YOUR_RESUME_SUMMARY, MAX_AGENT_ITERATIONS,
    MAX_AGENT_RPM, JOB_HISTORY_FILE
)


def get_llm():
    return ChatGroq(
        temperature=0.3,
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY
    )


def load_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(JOB_HISTORY_FILE):
        with open(JOB_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"found_jobs": [], "applied_jobs": [], "last_search": None}


def save_history(history):
    with open(JOB_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def run_job_finder():
    print("\n" + "=" * 60)
    print("  AGENT 4: JOB FINDER")
    print("=" * 60)

    limiter.check()

    history = load_history()
    llm = get_llm()
    search_tool = SerperDevTool()

    # Extract skills safely OUTSIDE f-strings
    skills_list = [s.strip() for s in YOUR_SKILLS.split(",")]
    skill1 = skills_list[0] if len(skills_list) > 0 else "Python"
    skill2 = skills_list[1] if len(skills_list) > 1 else skill1
    skill3 = skills_list[2] if len(skills_list) > 2 else skill1

    # ─── AGENT 1: JOB SCOUT ───
    job_scout = Agent(
        role="Senior Technical Recruiter",
        goal="Find active software engineering jobs matching developer skills.",
        backstory=(
            "You have placed 1000+ candidates at startups and big tech. "
            "You check LinkedIn Jobs, Indeed, Glassdoor, Wellfound, and "
            "company career pages for fresh listings posted in the last 30 days."
        ),
        tools=[search_tool],
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── AGENT 2: APPLICATION PREP ───
    app_prep = Agent(
        role="Career Coach and Resume Specialist",
        goal="Prepare tailored cover letters and interview prep for each job.",
        backstory=(
            "You are a career coach who knows ATS systems inside out. "
            "You identify exact keywords from job descriptions and write "
            "cover letters that pass automated screening."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── BUILD SEARCH QUERIES SAFELY ───
    search_queries = (
        "Run these exact searches one by one:\n"
        "1. linkedin.com/jobs " + skill1 + " software engineer 2025\n"
        "2. indeed.com " + skill1 + " " + skill2 + " developer remote\n"
        "3. glassdoor.com software engineer " + skill2 + " hiring 2025\n"
        "4. wellfound.com " + skill1 + " startup engineer remote\n"
        "5. " + skill1 + " " + skill2 + " software engineer job opening remote 2025\n"
        "6. careers.google.com OR amazon.jobs " + skill1 + " engineer\n"
    )

    # ─── TASK 1: SEARCH JOBS ───
    search_task = Task(
        description=(
            "Find active full-time software engineering jobs.\n\n"
            "Developer skills: " + YOUR_SKILLS + "\n"
            "Experience level: " + YOUR_EXPERIENCE_YEARS + " years\n\n"
            + search_queries + "\n"
            "For each job provide:\n"
            "- Company name\n"
            "- Job title\n"
            "- Location (Remote / On-site / Hybrid)\n"
            "- Key requirements summarized\n"
            "- Salary range if available\n"
            "- Application link\n"
            "- Match score 1 to 10 based on skills\n"
            "- Missing skills the candidate should learn\n\n"
            "Find at least 8 jobs ranked by match score.\n"
            "Only include jobs posted in the last 30 days."
        ),
        expected_output="Ranked list of 8+ jobs with all details.",
        agent=job_scout
    )

    # ─── TASK 2: PREPARE APPLICATIONS ───
    app_task = Task(
        description=(
            "For the TOP 3 highest-matched jobs, prepare application materials.\n\n"
            "Developer profile:\n"
            "- Name: " + YOUR_NAME + "\n"
            "- Skills: " + YOUR_SKILLS + "\n"
            "- Experience: " + YOUR_EXPERIENCE_YEARS + " years\n"
            "- Summary: " + YOUR_RESUME_SUMMARY + "\n\n"
            "For EACH of the 3 jobs create:\n\n"
            "1. TAILORED COVER LETTER (under 250 words)\n"
            "   - Address the specific company and role\n"
            "   - Highlight 2 to 3 most relevant skills\n"
            "   - Show knowledge of company product or mission\n\n"
            "2. ATS KEYWORDS (5 to 10 keywords)\n"
            "   - Words from job description that must be in resume\n\n"
            "3. INTERVIEW PREP (3 likely questions)\n"
            "   - Questions specific to this role\n"
            "   - Suggested talking points for each\n\n"
            "4. ACTION STEPS\n"
            "   - Exact steps to apply\n"
            "   - Referral contacts to search on LinkedIn\n\n"
            "Use clear headers for each job."
        ),
        expected_output="3 complete application packages with cover letters, keywords, and prep.",
        agent=app_prep
    )

    # ─── EXECUTE ───
    try:
        limiter.check()

        crew = Crew(
            agents=[job_scout, app_prep],
            tasks=[search_task, app_task],
            process=Process.sequential,
            verbose=True
        )

        result = str(crew.kickoff())

        # Save to history
        history["found_jobs"].append({
            "date": datetime.now().isoformat(),
            "results": result[:3000]
        })
        history["last_search"] = datetime.now().isoformat()

        # Keep only last 30 entries
        if len(history["found_jobs"]) > 30:
            history["found_jobs"] = history["found_jobs"][-30:]

        save_history(history)

        # Send notification
        short_result = result[:3000] if len(result) > 3000 else result
        notification = (
            "JOB OPPORTUNITIES FOUND\n"
            "========================\n\n"
            + short_result + "\n\n"
            "========================\n"
            "Cover letters and prep materials ready!\n"
            "Click the links above to apply."
        )
        send_notification(notification)

        send_agent_report(
            "Job Finder", "success",
            "Found jobs and prepared 3 applications. Check Telegram!"
        )

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