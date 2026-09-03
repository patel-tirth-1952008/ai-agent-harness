import os
import re
import json
import litellm
from dotenv import load_dotenv
import subprocess
load_dotenv()

# Force LiteLLM to drop parameters unsupported by Groq
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


def parse_files_from_markdown(text):
    """
    Robust multi-format parser:
    1. Extracts markdown blocks formatted as: FILE: path/to/file.ext\n```code...```
    2. Fallback to JSON object parsing if markdown block is missing.
    """
    files = {}

    # Pattern 1: FILE: path/to/filename.ext \n ```lang \n content \n ```
    pattern = r"(?:###\s*)?(?:FILE|File|FILENAME|Filename):\s*`?([a-zA-Z0-9_\-\.\/]+)`?\s*\n+```[a-zA-Z0-9_\-]*\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)
    for filename, content in matches:
        clean_name = filename.strip()
        if clean_name and content.strip():
            files[clean_name] = content.strip()

    if files:
        return files

    # Pattern 2: Fallback JSON extraction
    try:
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
        raw_json = match.group(1) if match else re.search(r'\{[\s\S]*\}', text).group(0)
        data = json.loads(raw_json)
        if isinstance(data.get("files"), dict):
            return data["files"]
    except Exception:
        pass

    return files


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
        goal="Create professional README.md, docker-compose.yml, and config files.",
        backstory="You write clean architecture docs, docker setup, and professional READMEs.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True
    )

    devops_task = Task(
        description=(
            f"Create documentation and Docker config for: {project['name']}\n"
            f"Tech Stack: {project['stack']}\n\n"
            f"You MUST format each file EXACTLY like this:\n\n"
            f"FILE: README.md\n"
            f"```markdown\n"
            f"# {project['name']}\n"
            f"## Overview\n{project['why']}\n\n"
            f"## Tech Stack\n{project['stack']}\n\n"
            f"## Quick Start\n```bash\ndocker-compose up\n```\n"
            f"```\n\n"
            f"FILE: docker-compose.yml\n"
            f"```yaml\n"
            f"version: '3.8'\n"
            f"services:\n"
            f"  backend:\n"
            f"    build: ./backend\n"
            f"    ports:\n"
            f"      - '8000:8000'\n"
            f"  frontend:\n"
            f"    build: ./frontend\n"
            f"    ports:\n"
            f"      - '3000:3000'\n"
            f"```\n\n"
            f"FILE: .gitignore\n"
            f"```text\n"
            f"__pycache__/\nnode_modules/\n.env\nvenv/\n"
            f"```\n"
        ),
        expected_output="Multiple files formatted with FILE: path headers and code blocks.",
        agent=devops_agent
    )

    crew_1 = Crew(agents=[devops_agent], tasks=[devops_task], process=Process.sequential)
    res_1_text = str(crew_1.kickoff())
    files_1 = parse_files_from_markdown(res_1_text)
    
    # Fallback if step 1 parsing returned nothing
    if not files_1:
        files_1 = {
            "README.md": f"# {project['name']}\n\n{project['why']}\n\n## Tech Stack\n{project['stack']}\n\n## Quick Start\n```bash\ndocker-compose up\n```",
            "docker-compose.yml": "version: '3.8'\nservices:\n  backend:\n    build: ./backend\n    ports:\n      - '8000:8000'\n  frontend:\n    build: ./frontend\n    ports:\n      - '3000:3000'",
            ".gitignore": "__pycache__/\nnode_modules/\n.env\nvenv/\n"
        }
    all_files.update(files_1)
    print(f"✅ Step 1 generated {len(files_1)} files.")

    # ─── STEP 2: BACKEND CODE ───
    print("\n⚙️ [2/3] Generating Production Backend Code...")
    backend_agent = Agent(
        role="Senior Backend Engineer",
        goal="Write full backend application code, models, routes, and Dockerfile.",
        backstory="You build complete Python/FastAPI or Node.js APIs with authentication, database ORMs, and error handling.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True
    )

    backend_task = Task(
        description=(
            f"Write the backend service files for: {project['name']}\n"
            f"Tech Stack: {project['stack']}\n\n"
            f"You MUST format each file EXACTLY like this:\n\n"
            f"FILE: backend/main.py\n"
            f"```python\n"
            f"from fastapi import FastAPI\n\n"
            f"app = FastAPI(title='{project['name']}')\n\n"
            f"@app.get('/')\n"
            f"def root():\n"
            f"    return {{'message': 'Welcome to {project['name']} API'}}\n"
            f"```\n\n"
            f"FILE: backend/requirements.txt\n"
            f"```text\n"
            f"fastapi\nuvicorn\npydantic\nsqlalchemy\npsycopg2-binary\npython-dotenv\n"
            f"```\n\n"
            f"FILE: backend/Dockerfile\n"
            f"```dockerfile\n"
            f"FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
            f"```\n"
        ),
        expected_output="Multiple backend files formatted with FILE: path headers and code blocks.",
        agent=backend_agent
    )

    crew_2 = Crew(agents=[backend_agent], tasks=[backend_task], process=Process.sequential)
    res_2_text = str(crew_2.kickoff())
    files_2 = parse_files_from_markdown(res_2_text)

    # Fallback if step 2 parsing returned nothing
    if not files_2:
        files_2 = {
            "backend/main.py": f"from fastapi import FastAPI\n\napp = FastAPI(title='{project['name']}')\n\n@app.get('/')\ndef root():\n    return {{'message': '{project['name']} API is live'}}\n",
            "backend/requirements.txt": "fastapi\nuvicorn\npydantic\npython-dotenv\n",
            "backend/Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
        }
    all_files.update(files_2)
    print(f"✅ Step 2 generated {len(files_2)} files.")

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
            f"You MUST format each file EXACTLY like this:\n\n"
            f"FILE: frontend/package.json\n"
            f"```json\n"
            f"{{\n"
            f'  "name": "frontend",\n'
            f'  "scripts": {{\n'
            f'    "dev": "next dev"\n'
            f'  }},\n'
            f'  "dependencies": {{\n'
            f'    "next": "14.0.0",\n'
            f'    "react": "^18.2.0",\n'
            f'    "react-dom": "^18.2.0"\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"FILE: frontend/src/app/page.tsx\n"
            f"```tsx\n"
            f"export default function Home() {{\n"
            f"  return (\n"
            f"    <main className='p-8'>\n"
            f"      <h1 className='text-3xl font-bold'>{project['name']}</h1>\n"
            f"      <p className='mt-2 text-gray-600'>{project['why']}</p>\n"
            f"    </main>\n"
            f"  );\n"
            f"}}\n"
            f"```\n\n"
            f"FILE: frontend/Dockerfile\n"
            f"```dockerfile\n"
            f"FROM node:20-alpine\nWORKDIR /app\nCOPY package.json .\nRUN npm install\nCOPY . .\nCMD [\"npm\", \"run\", \"dev\"]\n"
            f"```\n"
        ),
        expected_output="Multiple frontend files formatted with FILE: path headers and code blocks.",
        agent=frontend_agent
    )

    crew_3 = Crew(agents=[frontend_agent], tasks=[frontend_task], process=Process.sequential)
    res_3_text = str(crew_3.kickoff())
    files_3 = parse_files_from_markdown(res_3_text)

    # Fallback if step 3 parsing returned nothing
    if not files_3:
        files_3 = {
            "frontend/package.json": f"{{\n  \"name\": \"frontend\",\n  \"scripts\": {{\n    \"dev\": \"next dev\"\n  }},\n  \"dependencies\": {{\n    \"next\": \"14.0.0\",\n    \"react\": \"^18.2.0\",\n    \"react-dom\": \"^18.2.0\"\n  }}\n}}",
            "frontend/src/app/page.tsx": f"export default function Home() {{\n  return <main className='p-8'><h1 className='text-3xl font-bold'>{project['name']}</h1></main>;\n}}",
            "frontend/Dockerfile": "FROM node:20-alpine\nWORKDIR /app\nCOPY package.json .\nRUN npm install\nCOPY . .\nCMD [\"npm\", \"run\", \"dev\"]\n"
        }
    all_files.update(files_3)
    print(f"✅ Step 3 generated {len(files_3)} files.")

    # ─── DEPLOY TO GITHUB ───
    repo_name = project["slug"]
    description = f"{project['name']} — {project['why']}"

    print(f"\n📦 Deploying total {len(all_files)} files to GitHub repo '{repo_name}'...")

    try:
        repo_url = github_mgr.push_files(repo_name, all_files, description)
        trigger_verify(repo_name)
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



def trigger_verify(repo_name: str):
    """Trigger Level-2 CI verification on the generated repo."""
    try:
        subprocess.run(
            [
                "gh", "workflow", "run", "verify-project.yml",
                "-f", f"target_repo={repo_name}",
                "-f", "attempt=1",
            ],
            check=False,
        )
        print(f"🚀 Triggered verify-project for {repo_name}")
    except Exception as e:
        print(f"Could not trigger verify workflow: {e}")

if __name__ == "__main__":
    run_portfolio_builder()