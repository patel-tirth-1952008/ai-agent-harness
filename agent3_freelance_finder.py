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
    MAX_AGENT_RPM, FREELANCE_HISTORY_FILE
)


def get_llm():
    return ChatGroq(
        temperature=0.3,
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY
    )


def load_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(FREELANCE_HISTORY_FILE):
        with open(FREELANCE_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"found_gigs": [], "last_search": None}


def save_history(history):
    with open(FREELANCE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def run_freelance_finder():
    print("\n" + "=" * 60)
    print("  AGENT 3: FREELANCE FINDER")
    print("=" * 60)

    limiter.check()

    history = load_history()
    llm = get_llm()
    search_tool = SerperDevTool()

    # Extract skills safely OUTSIDE f-strings to avoid quote errors
    skills_list = [s.strip() for s in YOUR_SKILLS.split(",")]
    skill1 = skills_list[0] if len(skills_list) > 0 else "Python"
    skill2 = skills_list[1] if len(skills_list) > 1 else skill1
    skill3 = skills_list[2] if len(skills_list) > 2 else skill1

    # ─── AGENT 1: GIG HUNTER ───
    gig_hunter = Agent(
        role="Expert Freelance Business Development Manager",
        goal="Find active freelance projects matching the developer skills.",
        backstory=(
            "You have helped 500+ freelancers find projects on Upwork, "
            "Freelancer.com, LinkedIn, and Toptal. You focus on real, "
            "currently active project listings that match developer skills."
        ),
        tools=[search_tool],
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── AGENT 2: PROPOSAL WRITER ───
    proposal_writer = Agent(
        role="Freelance Proposal Specialist",
        goal="Write winning proposals for freelance projects.",
        backstory=(
            "You write concise, personalized proposals that win contracts. "
            "You address the client problem first, then show relevant skills."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── BUILD SEARCH QUERIES SAFELY (no nested quotes) ───
    search_queries = (
        "Run these exact searches one by one:\n"
        "1. upwork.com " + skill1 + " freelance project 2025\n"
        "2. freelancer.com " + skill2 + " project hiring remote\n"
        "3. linkedin.com freelance " + skill1 + " contract remote 2025\n"
        "4. guru.com OR toptal.com " + skill3 + " freelance developer\n"
        "5. freelance " + skill1 + " " + skill2 + " project remote 2025\n"
    )

    # ─── TASK 1: SEARCH ───
    search_task = Task(
        description=(
            "Find active freelance projects matching these developer skills: "
            + YOUR_SKILLS + ".\n\n"
            + search_queries + "\n"
            "The developer has " + YOUR_EXPERIENCE_YEARS + " years of experience.\n\n"
            "For each opportunity provide:\n"
            "- Platform (Upwork, Freelancer, LinkedIn, etc.)\n"
            "- Project title\n"
            "- Short description\n"
            "- Budget if visible\n"
            "- Required skills\n"
            "- Link to listing\n"
            "- Match score 1 to 10\n\n"
            "Find at least 5 opportunities ranked by match score."
        ),
        expected_output="Ranked list of 5+ freelance gigs with all details.",
        agent=gig_hunter
    )

    # ─── TASK 2: WRITE PROPOSALS ───
    proposal_task = Task(
        description=(
            "For the top 3 opportunities found, write a customized proposal for each.\n\n"
            "Developer profile:\n"
            "- Name: " + YOUR_NAME + "\n"
            "- Skills: " + YOUR_SKILLS + "\n"
            "- Experience: " + YOUR_EXPERIENCE_YEARS + " years\n"
            "- Summary: " + YOUR_RESUME_SUMMARY + "\n\n"
            "Each proposal must:\n"
            "1. Address the client specific problem first\n"
            "2. Show relevant experience briefly\n"
            "3. Give a 3-step action plan\n"
            "4. Include timeline estimate\n"
            "5. End with call to action\n"
            "6. Be under 200 words\n\n"
            "Use clear headers for each proposal."
        ),
        expected_output="3 customized proposals under 200 words each.",
        agent=proposal_writer
    )

    # ─── EXECUTE ───
    try:
        limiter.check()

        crew = Crew(
            agents=[gig_hunter, proposal_writer],
            tasks=[search_task, proposal_task],
            process=Process.sequential,
            verbose=True
        )

        result = str(crew.kickoff())

        # Save to history
        history["found_gigs"].append({
            "date": datetime.now().isoformat(),
            "results": result[:2000]
        })
        history["last_search"] = datetime.now().isoformat()

        # Keep only last 30 entries
        if len(history["found_gigs"]) > 30:
            history["found_gigs"] = history["found_gigs"][-30:]

        save_history(history)

        # Send notification (Telegram limit is 4096 chars)
        short_result = result[:3000] if len(result) > 3000 else result
        notification = (
            "FREELANCE OPPORTUNITIES FOUND\n"
            "========================\n\n"
            + short_result + "\n\n"
            "========================\n"
            "Full results saved to history."
        )
        send_notification(notification)

        send_agent_report(
            "Freelance Finder", "success",
            "Found gigs and wrote proposals. Check Telegram!"
        )

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