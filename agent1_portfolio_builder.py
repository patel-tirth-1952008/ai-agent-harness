import os
import re
import json
import litellm
from dotenv import load_dotenv

load_dotenv()

# ─── MONKEYPATCH LITELLM TO STRIP UNSUPPORTED GROQ PARAMS ───
litellm.drop_params = True
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

# ─── ENV VARS ───
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

from crewai import Agent, Task, Crew, Process, LLM
from config import (
    GROQ_API_KEY, GROQ_MODEL,
    MAX_AGENT_ITERATIONS, MAX_AGENT_RPM,
    PORTFOLIO_HISTORY_FILE
)
from limiter import limiter
from notifier import send_agent_report
from github_manager import github_mgr


def get_llm(temperature=0.2):
    """Routes Groq safely through OpenAI-compatible endpoint format."""
    clean_model = GROQ_MODEL.replace("groq/", "").replace("openai/", "")
    return LLM(
        model=f"openai/{clean_model}",
        base_url="https://api.groq.com/openai/v1",
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


# ─── HIGH-VALUE INDUSTRY PROJECTS ───
INDUSTRY_PROJECTS = [
    {
        "slug": "realtime-collaborative-editor",
        "name": "Real-time Collaborative Document Editor",
        "why": "Google Docs clone. Tests WebSockets, real-time sync, and concurrency.",
        "stack": "Next.js 14, TypeScript, FastAPI, WebSockets, PostgreSQL, Redis, Docker"
    },
    {
        "slug": "url-shortener-analytics",
        "name": "URL Shortener with Analytics Dashboard",
        "why": "Bit.ly clone. Tests system design, caching, rate limiting, and analytics.",
        "stack": "FastAPI, PostgreSQL, Redis, React, TypeScript, Chart.js, Docker"
    },
    {
        "slug": "ecommerce-microservices",
        "name": "E-commerce Platform Architecture",
        "why": "Amazon-scale architecture. Tests services, authentication, and payments.",
        "stack": "Node.js, Express, MongoDB, Redis, Stripe API, React, Docker"
    },
    {
        "slug": "job-board-with-search",
        "name": "Job Board with Advanced Filters",
        "why": "LinkedIn clone. Tests complex queries, authentication, and UI state.",
        "stack": "Next.js, FastAPI, PostgreSQL, Redis, Docker, JWT Auth"
    },
    {
        "slug": "chat-application",
        "name": "Real-time Chat Platform",
        "why": "WhatsApp/Slack clone. Tests WebSockets, room management, and messaging.",
        "stack": "Next.js, Socket.io, Node.js, PostgreSQL, Redis, Docker"
    },
    {
        "slug": "task-management-saas",
        "name": "Task Management SaaS (Trello Clone)",
        "why": "Tests drag-and-drop UI, state management, and team permissions.",
        "stack": "React, TypeScript, FastAPI, PostgreSQL, Redis, Docker"
    },
    {
        "slug": "banking-api-system",
        "name": "Banking API with Transaction System",
        "why": "Tests ACID transactions, security, audit logs, and clean backend architecture.",
        "stack": "FastAPI, PostgreSQL, Redis, JWT, pytest, Docker"
    }
]


def pick_next_project(history):
    built = set(history.get("built_projects", []))
    for project in INDUSTRY_PROJECTS:
        if project["slug"] not in built:
            return project
    return INDUSTRY_PROJECTS[len(built) % len(INDUSTRY_PROJECTS)]


def parse_json_from_text(text):
    """Safely extracts JSON object from LLM response text."""
    try:
        # Match standard json block
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
        if match:
            return json.loads(match.group(1))
        
        # Fallback to any curly brace match
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"⚠️ JSON parsing error: {e}")
    return {}


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

    all_files = {}

    # ─── STEP 1: DEVOPS & DOCS ───
    print("\n📝 [1/3] Generating Architecture, README & Docker setup...")
    devops_agent = Agent(
        role="DevOps Architect and Technical Writer",
        goal="Create professional README.md, docker-compose.yml, and root files.",
        backstory="You write clean architecture docs, docker setup, and professional READMEs.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True
    )

    devops_task = Task(
        description=(
            f"Create the core documentation and Docker config for: {project['name']}\n"
            f"Tech Stack: {project['stack']}\n\n"
            f"Output strictly inside a JSON block with key 'files':\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "README.md": "# {project["name"]}\\n\\n## Architecture\\n...\\n\\n## Quick Start\\n```bash\\ndocker-compose up\\n```",\n'
            f'    "docker-compose.yml": "version: \'3.8\'\\nservices:\\n...",\n'
            f'    ".gitignore": "__pycache__/\\nnode_modules/\\n.env\\nvenv/",\n'
            f'    ".github/workflows/ci.yml": "name: CI\\non: [push]\\njobs:\\n..."\n'
            f'  }}\n'
            f"}}\n"
            f"```"
        ),
        expected_output="JSON object containing README.md, docker-compose.yml, and CI files.",
        agent=devops_agent
    )

    crew_1 = Crew(agents=[devops_agent], tasks=[devops_task], process=Process.sequential)
    res_1 = parse_json_from_text(str(crew_1.kickoff()))
    all_files.update(res_1.get("files", {}))

    # ─── STEP 2: BACKEND CODE ───
    print("\n⚙️ [2/3] Generating Production Backend Code...")
    backend_agent = Agent(
        role="Senior Backend Engineer",
        goal="Write full backend application code, models, routes, and Dockerfile.",
        backstory="You build complete Python/FastAPI or Node.js APIs with authentication, database ORM, and error handling.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True
    )

    backend_task = Task(
        description=(
            f"Write the backend service files for: {project['name']}\n"
            f"Tech Stack: {project['stack']}\n\n"
            f"Output strictly inside a JSON block with key 'files':\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "backend/main.py": "from fastapi import FastAPI\\n...",\n'
            f'    "backend/config.py": "...",\n'
            f'    "backend/database.py": "...",\n'
            f'    "backend/requirements.txt": "fastapi\\nuvicorn\\npydantic\\nsqlalchemy\\npsycopg2-binary\\npython-dotenv",\n'
            f'    "backend/Dockerfile": "FROM python:3.11-slim\\nWORKDIR /app\\nCOPY requirements.txt .\\nRUN pip install -r requirements.txt\\nCOPY . .\\nCMD [\\"uvicorn\\", \\"main:app\\", \\"--host\\", \\"0.0.0.0\\", \\"--port\\", \\"8000\\"]"\n'
            f'  }}\n'
            f"}}\n"
            f"```"
        ),
        expected_output="JSON object containing complete backend codebase.",
        agent=backend_agent
    )

    crew_2 = Crew(agents=[backend_agent], tasks=[backend_task], process=Process.sequential)
    res_2 = parse_json_from_text(str(crew_2.kickoff()))
    all_files.update(res_2.get("files", {}))

    # ─── STEP 3: FRONTEND CODE ───
    print("\n🎨 [3/3] Generating Modern Frontend Code...")
    frontend_agent = Agent(
        role="Senior Frontend Engineer",
        goal="Build modern Next.js/React UI code, pages, and components.",
        backstory="You write clean TypeScript and React components with responsive layout.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True
    )

    frontend_task = Task(
        description=(
            f"Write the frontend web application files for: {project['name']}\n"
            f"Tech Stack: {project['stack']}\n\n"
            f"Output strictly inside a JSON block with key 'files':\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "frontend/package.json": "{{\\n  \\"name\\": \\"frontend\\",\\n  \\"dependencies\\": {{\\n    \\"next\\": \\"14.0.0\\",\\n    \\"react\\": \\"^18.2.0\\"\\n  }}\\n}}",\n'
            f'    "frontend/src/app/page.tsx": "export default function Home() {{\\n  return <main><h1>{project["name"]}</h1></main>\\n}}",\n'
            f'    "frontend/src/lib/api.ts": "export const API_URL = process.env.NEXT_PUBLIC_API_URL || \'http://localhost:8000\';",\n'
            f'    "frontend/Dockerfile": "FROM node:20-alpine\\nWORKDIR /app\\nCOPY package.json .\\nRUN npm install\\nCOPY . .\\nCMD [\\"npm\\", \\"run\\", \\"dev\\"]"\n'
            f'  }}\n'
            f"}}\n"
            f"```"
        ),
        expected_output="JSON object containing complete frontend codebase.",
        agent=frontend_agent
    )

    crew_3 = Crew(agents=[frontend_agent], tasks=[frontend_task], process=Process.sequential)
    res_3 = parse_json_from_text(str(crew_3.kickoff()))
    all_files.update(res_3.get("files", {}))

    # ─── DEPLOY TO GITHUB ───
    if not all_files:
        send_agent_report(
            "Portfolio Builder", "error",
            f"Failed to generate project files for {project['name']}."
        )
        return None

    repo_name = project["slug"]
    description = f"{project['name']} — {project['why']}"

    print(f"\n📦 Generated {len(all_files)} total files across all services. Pushing to GitHub...")

    try:
        repo_url = github_mgr.push_files(repo_name, all_files, description)

        history["built_projects"].append(project["slug"])
        save_history(history)

        send_agent_report(
            "Portfolio Builder", "success",
            f"Built: *{project['name']}*\n"
            f"Files: {len(all_files)}\n"
            f"Stack: {project['stack']}\n"
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