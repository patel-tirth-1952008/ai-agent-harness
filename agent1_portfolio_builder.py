import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from config import (
    GROQ_API_KEY, GROQ_MODEL,
    MAX_AGENT_ITERATIONS, MAX_AGENT_RPM,
    PORTFOLIO_HISTORY_FILE
)
from limiter import limiter
from notifier import send_agent_report
from github_manager import github_mgr


def get_llm(temperature=0.2):
    """Uses CrewAI's native LLM wrapper for Groq."""
    model_name = GROQ_MODEL if GROQ_MODEL.startswith("groq/") else f"groq/{GROQ_MODEL}"
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=temperature
    )


def load_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"built_projects": []}


def save_history(history):
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─── HIGH-VALUE PROJECT IDEAS ───
INDUSTRY_PROJECTS = [
    {
        "slug": "realtime-collaborative-editor",
        "name": "Real-time Collaborative Document Editor",
        "why": "Google Docs clone. Tests WebSockets, CRDTs, real-time sync. Asked at Google, Notion, Figma.",
        "stack": "Next.js 14, TypeScript, Socket.io, Node.js, PostgreSQL, Redis, Docker"
    },
    {
        "slug": "url-shortener-analytics",
        "name": "URL Shortener with Analytics Dashboard",
        "why": "Bit.ly clone. Tests system design, caching, rate limiting, analytics. Classic Amazon/Meta interview project.",
        "stack": "FastAPI, PostgreSQL, Redis, React, TypeScript, Chart.js, Docker"
    },
    {
        "slug": "ecommerce-microservices",
        "name": "E-commerce Platform with Microservices",
        "why": "Amazon-scale architecture. Tests microservices, message queues, payment integration.",
        "stack": "Node.js, Express, MongoDB, RabbitMQ, Stripe, React, Docker"
    },
    {
        "slug": "job-board-with-search",
        "name": "Job Board with Advanced Search & Filters",
        "why": "LinkedIn clone. Tests full-text search, pagination, complex queries.",
        "stack": "Next.js, FastAPI, PostgreSQL, Elasticsearch, Redis, Docker, JWT Auth"
    },
    {
        "slug": "chat-application",
        "name": "Real-time Chat Application with Groups",
        "why": "WhatsApp/Slack clone. Tests WebSockets, message delivery, notifications, file uploads.",
        "stack": "Next.js, Socket.io, Node.js, PostgreSQL, Redis, AWS S3 mock, Docker"
    },
    {
        "slug": "task-management-saas",
        "name": "Task Management SaaS (Trello Clone)",
        "why": "Tests drag-and-drop UI, real-time collaboration, team features, permissions.",
        "stack": "React, TypeScript, DnD Kit, FastAPI, PostgreSQL, Redis, Docker"
    },
    {
        "slug": "video-streaming-platform",
        "name": "Video Streaming Platform",
        "why": "YouTube clone. Tests file uploads, video processing, streaming, CDN concepts.",
        "stack": "Next.js, Node.js, MongoDB, FFmpeg concepts, AWS S3 mock, Docker"
    },
    {
        "slug": "food-delivery-app",
        "name": "Food Delivery Application",
        "why": "Uber Eats clone. Tests geolocation, real-time tracking, payment flow, multi-role auth.",
        "stack": "Next.js, FastAPI, PostgreSQL, Redis, Mapbox, Stripe, Docker"
    },
    {
        "slug": "social-media-feed",
        "name": "Social Media Platform with Feed Algorithm",
        "why": "Twitter/Instagram clone. Tests feed generation, followers, likes, media uploads.",
        "stack": "Next.js, FastAPI, PostgreSQL, Redis, Celery, Docker"
    },
    {
        "slug": "banking-api-system",
        "name": "Banking API with Transaction System",
        "why": "Tests ACID transactions, security, audit logs. High-value for fintech interviews.",
        "stack": "FastAPI, PostgreSQL, Redis, JWT, bcrypt, pytest, Docker"
    },
]


def pick_next_project(history):
    built = set(history.get("built_projects", []))
    for project in INDUSTRY_PROJECTS:
        if project["slug"] not in built:
            return project
    return INDUSTRY_PROJECTS[len(built) % len(INDUSTRY_PROJECTS)]


def run_portfolio_builder():
    print("\n" + "=" * 60)
    print("  AGENT 1: INDUSTRY-GRADE PORTFOLIO BUILDER")
    print("=" * 60)

    limiter.check()

    history = load_history()
    project = pick_next_project(history)

    print(f"🎯 Building: {project['name']}")
    print(f"📚 Stack: {project['stack']}")

    llm = get_llm()

    # ─── AGENTS ───
    architect = Agent(
        role="Principal Software Architect",
        goal="Design production-grade full-stack systems with proper architecture and best practices.",
        backstory="You have 20 years of experience architecting scalable systems at Google, Amazon, and Meta.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    backend_dev = Agent(
        role="Senior Backend Engineer",
        goal="Write complete, production-ready backend code with authentication, database, tests, and error handling.",
        backstory="You write clean backend code handling auth, ORMs, validation, and unit tests.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    frontend_dev = Agent(
        role="Senior Frontend Engineer",
        goal="Build modern, responsive frontends with React/Next.js, state management, and clean UI.",
        backstory="You build modern UIs using Next.js, TypeScript, and Tailwind CSS.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    devops_engineer = Agent(
        role="DevOps Engineer and Technical Writer",
        goal="Create Docker setup, CI/CD pipelines, and professional documentation.",
        backstory="You create docker-compose setups, GitHub Actions workflows, and clear READMEs.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    # ─── TASKS ───
    architecture_task = Task(
        description=(
            f"Design the architecture for: {project['name']}\n"
            f"Tech stack: {project['stack']}\n\n"
            f"Output folder tree, database schema, and list of API endpoints."
        ),
        expected_output="Architecture specification with folder structure and API endpoints.",
        agent=architect,
    )

    backend_task = Task(
        description=(
            f"Write backend files for {project['name']} in JSON format:\n\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "backend/main.py": "code...",\n'
            f'    "backend/requirements.txt": "...",\n'
            f'    "backend/Dockerfile": "..."\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"Output ONLY valid JSON."
        ),
        expected_output="Valid JSON object containing backend files.",
        agent=backend_dev,
    )

    frontend_task = Task(
        description=(
            f"Write frontend files for {project['name']} in JSON format:\n\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "frontend/package.json": "...",\n'
            f'    "frontend/src/app/page.tsx": "...",\n'
            f'    "frontend/Dockerfile": "..."\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"Output ONLY valid JSON."
        ),
        expected_output="Valid JSON object containing frontend files.",
        agent=frontend_dev,
    )

    devops_task = Task(
        description=(
            f"Create README.md, docker-compose.yml, and deployment docs for {project['name']}.\n"
            f"Output JSON format:\n\n"
            f"```json\n"
            f"{{\n"
            f'  "repo_name": "{project["slug"]}",\n'
            f'  "description": "Full-stack {project["name"]}",\n'
            f'  "files": {{\n'
            f'    "README.md": "# {project["name"]}\\n\\n...",\n'
            f'    "docker-compose.yml": "..."\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"Output ONLY valid JSON."
        ),
        expected_output="Valid JSON with repo_name, description, and docs files.",
        agent=devops_engineer,
    )

    # ─── EXECUTE ───
    try:
        limiter.check()

        crew = Crew(
            agents=[architect, backend_dev, frontend_dev, devops_engineer],
            tasks=[architecture_task, backend_task, frontend_task, devops_task],
            process=Process.sequential,
            verbose=True,
        )

        result = str(crew.kickoff())

        # Extract JSON blocks
        all_files = {}
        repo_name = project["slug"]
        description = f"{project['name']} — {project['why']}"

        json_blocks = re.findall(r'\{[\s\S]*?\}', result)

        for block in json_blocks:
            try:
                data = json.loads(block)
                if "files" in data and isinstance(data["files"], dict):
                    all_files.update(data["files"])
                if "repo_name" in data:
                    repo_name = data["repo_name"]
                if "description" in data:
                    description = data["description"]
            except Exception:
                continue

        if not all_files:
            send_agent_report(
                "Portfolio Builder", "error",
                f"Failed to parse JSON output for {project['name']}."
            )
            return None

        print(f"\n📦 Generated {len(all_files)} files. Pushing to GitHub...")

        repo_url = github_mgr.push_files(repo_name, all_files, description)

        history["built_projects"].append(project["slug"])
        save_history(history)

        send_agent_report(
            "Portfolio Builder", "success",
            f"Built: {project['name']}\n"
            f"Files: {len(all_files)}\n"
            f"URL: {repo_url}"
        )

        return repo_url

    except Exception as e:
        err = str(e)
        if "LIMIT REACHED" in err:
            send_agent_report("Portfolio Builder", "warning", err)
        else:
            send_agent_report("Portfolio Builder", "error", err)
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    run_portfolio_builder()