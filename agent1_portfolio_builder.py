import os
import re
import json
import time
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from config import (GROQ_API_KEY, SERPER_API_KEY, GROQ_MODEL,
                    MAX_AGENT_ITERATIONS, MAX_AGENT_RPM,
                    PORTFOLIO_HISTORY_FILE)
from limiter import limiter
from notifier import send_agent_report
from github_manager import github_mgr

os.environ["SERPER_API_KEY"] = SERPER_API_KEY


def get_groq_llm():
    return ChatGroq(
        temperature=0.2,
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY
    )


def load_portfolio_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"built_projects": []}


def save_portfolio_history(history):
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def run_portfolio_builder(target_repo: str = None):
    """
    Searches for trending resume projects, builds one, and pushes to GitHub.
    
    Args:
        target_repo: Optional specific repo name to push to.
                     If None, creates a new repo with the project name.
    """
    print("\n" + "=" * 60)
    print("🏗️  AGENT 1: PORTFOLIO BUILDER — Starting...")
    print("=" * 60)

    limiter.check()

    history = load_portfolio_history()
    already_built = ", ".join(history["built_projects"]) if history["built_projects"] else "none yet"

    llm = get_groq_llm()
    search_tool = SerperDevTool()

    # ─── AGENT: RECRUITER RESEARCHER ───
    recruiter = Agent(
        role="Senior Technical Recruiter at a FAANG Company",
        goal=(
            "Identify ONE specific, impressive backend or full-stack project "
            "that would make a software developer's GitHub stand out in 2025 interviews."
        ),
        backstory=(
            "You have reviewed over 10,000 developer resumes. You know exactly "
            "what projects make hiring managers excited. You focus on projects "
            "that demonstrate system design knowledge, API design, data structures, "
            "and real-world problem solving. You always avoid generic TODO apps "
            "or basic CRUD projects."
        ),
        tools=[search_tool],
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── AGENT: SENIOR DEVELOPER ───
    developer = Agent(
        role="Principal Software Engineer",
        goal="Build a complete, production-quality Python project with clean architecture.",
        backstory=(
            "You are a 15-year veteran engineer who writes clean, modular, "
            "well-documented code. Every project you create has: proper folder "
            "structure, comprehensive README with badges, setup instructions, "
            "architecture diagram in text, unit test examples, and a .gitignore. "
            "You follow PEP 8, include type hints, and write docstrings."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True
    )

    # ─── TASKS ───
    research_task = Task(
        description=(
            f"Search the web for 'most impressive GitHub projects for SDE resume 2025' "
            f"and 'projects that get software engineers hired at top companies'. "
            f"Also search for 'system design projects for junior developers GitHub'. "
            f"\n\nProjects already built (AVOID THESE): {already_built}"
            f"\n\nSelect ONE project that is:"
            f"\n- Technically impressive but buildable in a single session"
            f"\n- Demonstrates system design or algorithm knowledge"
            f"\n- NOT a basic TODO/CRUD/calculator app"
            f"\n- Examples: Rate Limiter, URL Shortener, In-Memory Cache (like Redis), "
            f"Log Aggregator, Task Queue, API Gateway, Web Crawler, Chat Server"
            f"\n\nOutput the project name (in slug format like 'rate-limiter-api'), "
            f"a one-line description, the tech stack, and 5 core features."
        ),
        expected_output=(
            "Project specification with: slug name, description, tech stack, "
            "and 5 feature bullet points."
        ),
        agent=recruiter
    )

    build_task = Task(
        description=(
            "Using the project specification from the recruiter, write the "
            "COMPLETE project code. You MUST output your response as a SINGLE "
            "valid JSON object with this EXACT structure:\n\n"
            "```json\n"
            "{\n"
            '  "repo_name": "project-slug-name",\n'
            '  "description": "One line project description for GitHub",\n'
            '  "files": {\n'
            '    "README.md": "Full markdown README content here...",\n'
            '    "src/main.py": "Main application code here...",\n'
            '    "src/__init__.py": "",\n'
            '    "tests/test_main.py": "Unit test code here...",\n'
            '    "requirements.txt": "package1\\npackage2",\n'
            '    ".gitignore": "__pycache__/\\n*.pyc\\n.env\\nvenv/",\n'
            '    "Makefile": "install:\\n\\tpip install -r requirements.txt"\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the JSON object. No text before or after it.\n"
            "2. The README must include: project title, description, features, "
            "architecture, setup instructions, usage examples, and tech stack.\n"
            "3. Include at least 4 source code files.\n"
            "4. Include at least 1 test file.\n"
            "5. All code must be syntactically correct Python.\n"
            "6. Use proper escaping for special characters in JSON strings."
        ),
        expected_output="A single valid JSON object containing repo_name, description, and files.",
        agent=developer
    )

    # ─── EXECUTE ───
    try:
        limiter.check()

        crew = Crew(
            agents=[recruiter, developer],
            tasks=[research_task, build_task],
            process=Process.sequential,
            verbose=True
        )

        result = str(crew.kickoff())

        # Parse JSON from the output
        json_match = re.search(r'\{[\s\S]*\}', result)

        if not json_match:
            send_agent_report("Portfolio Builder", "error",
                              "AI output did not contain valid JSON. Raw output saved to logs.")
            print(f"[-] Raw output:\n{result[:500]}")
            return None

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            cleaned = json_match.group(0)
            cleaned = cleaned.replace("\\n", "\n").replace('\\"', '"')
            try:
                data = json.loads(cleaned)
            except Exception:
                send_agent_report("Portfolio Builder", "error",
                                  "Failed to parse JSON from AI output.")
                return None

        repo_name = target_repo or data.get("repo_name", "ai-generated-project")
        description = data.get("description", "Built by AI Agent")
        files = data.get("files", {})

        if not files:
            send_agent_report("Portfolio Builder", "error", "No files were generated.")
            return None

        # Push to GitHub
        repo_url = github_mgr.push_files(repo_name, files, description)

        # Update history
        history["built_projects"].append(repo_name)
        save_portfolio_history(history)

        send_agent_report(
            "Portfolio Builder", "success",
            f"Project `{repo_name}` deployed!\n🔗 {repo_url}\n📄 Files: {len(files)}"
        )

        return repo_url

    except Exception as e:
        error_msg = str(e)
        if "LIMIT REACHED" in error_msg:
            send_agent_report("Portfolio Builder", "warning", error_msg)
        else:
            send_agent_report("Portfolio Builder", "error", f"Unexpected error: {error_msg}")
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Test run - specify your repo name or leave None for auto-naming
    run_portfolio_builder(target_repo=None)