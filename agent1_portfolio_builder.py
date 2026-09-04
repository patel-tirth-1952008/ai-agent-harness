import os
import re
import json
import subprocess
from datetime import datetime, timezone
import litellm
from dotenv import load_dotenv

load_dotenv()

# Force LiteLLM to drop Groq-incompatible params
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
    MAX_AGENT_ITERATIONS,
    PORTFOLIO_HISTORY_FILE
)
from limiter import limiter
from github_manager import github_mgr

PORTFOLIO_EVERY_HOURS = 30


def get_llm(temperature=0.2):
    clean_model = GROQ_MODEL.replace("groq/", "").replace("openai/", "")
    return LLM(
        model=f"openai/{clean_model}",
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def hours_since(iso_ts):
    if not iso_ts:
        return 10**9
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return 10**9


def load_history():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("built_projects", [])
    data.setdefault("last_success_at", None)
    data.setdefault("in_progress", None)
    return data


def save_history(history):
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


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
        "why": "Amazon-scale architecture. Tests microservices, authentication, and payments.",
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
    },
    {
        "slug": "video-streaming-api",
        "name": "Video Streaming & Transcoding Service",
        "why": "YouTube clone backend. Tests chunked uploads, video processing, and HLS streaming.",
        "stack": "FastAPI, Node.js, FFmpeg, React, PostgreSQL, Docker"
    },
    {
        "slug": "ai-content-generator-saas",
        "name": "AI Content Generator SaaS with Billing",
        "why": "AI SaaS wrapper. Tests OpenAI/Groq API integration, usage limits, and Stripe subscriptions.",
        "stack": "Next.js, FastAPI, Groq API, Stripe, PostgreSQL, Docker"
    },
    {
        "slug": "distributed-rate-limiter",
        "name": "Distributed API Rate Limiter & Gateway",
        "why": "Cloudflare/Kong Gateway clone. Tests Sliding Window algorithm, Redis, and high-throughput routing.",
        "stack": "FastAPI, Redis, Docker, Locust Load Testing"
    },
    {
        "slug": "food-delivery-tracking",
        "name": "Food Delivery App with Live Driver Tracking",
        "why": "UberEats/Swiggy clone. Tests Geo-indexing, live map tracking, and order state machines.",
        "stack": "Next.js, FastAPI, PostgreSQL PostGIS, WebSockets, Docker"
    },
    {
        "slug": "devops-monitoring-dashboard",
        "name": "Server Health & Metrics Monitoring Dashboard",
        "why": "Datadog/Prometheus clone. Tests time-series data, agent pinging, and real-time alerts.",
        "stack": "React, FastAPI, TimescaleDB/PostgreSQL, Redis, Recharts, Docker"
    },
    {
        "slug": "event-ticketing-system",
        "name": "High-Concurrency Event Ticketing System",
        "why": "BookMyShow/Ticketmaster clone. Tests row locking, race conditions, and queue management.",
        "stack": "Next.js, FastAPI, PostgreSQL (ACID), Redis Queue, Docker"
    },
    {
        "slug": "notification-engine",
        "name": "Multi-Channel Notification Dispatch Engine",
        "why": "Novu/Courier clone. Tests async workers (Celery), template rendering, and fallback providers.",
        "stack": "FastAPI, Celery, Redis, PostgreSQL, React, Docker"
    },
    {
        "slug": "file-storage-cloud",
        "name": "Cloud File Storage & Sharing Platform",
        "why": "Dropbox/Google Drive clone. Tests pre-signed URLs, file encryption, and storage quotas.",
        "stack": "Next.js, FastAPI, AWS S3 / MinIO, PostgreSQL, Docker"
    }
]


def pick_next_project(history):
    built = set(history.get("built_projects", []))
    for project in INDUSTRY_PROJECTS:
        if project["slug"] not in built:
            return project
    return INDUSTRY_PROJECTS[len(built) % len(INDUSTRY_PROJECTS)]


def parse_files_from_markdown(text):
    files = {}
    pattern = r"(?:###\s*)?(?:FILE|File|FILENAME|Filename):\s*`?([a-zA-Z0-9_\-\.\/]+)`?\s*\n+```[a-zA-Z0-9_\-]*\n([\s\S]*?)\n```"
    for filename, content in re.findall(pattern, text):
        clean_name = filename.strip()
        if clean_name and content.strip():
            files[clean_name] = content.strip()

    if files:
        return files

    try:
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
        raw_json = match.group(1) if match else re.search(r'\{[\s\S]*\}', text).group(0)
        data = json.loads(raw_json)
        if isinstance(data.get("files"), dict):
            return data["files"]
    except Exception:
        pass

    return files


def verify_static(files):
    errors = []
    for path, content in files.items():
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as e:
                errors.append(f"SyntaxError {path}:{e.lineno} {e.msg}")
            if re.search(r"from\s+\.", content):
                errors.append(f"Relative import in {path}")

        if path.endswith("package.json"):
            try:
                pkg = json.loads(content)
                deps = pkg.get("dependencies", {})
                for bad in ["yjs-react", "react-yjs", "fastapi-websockets-sync"]:
                    if bad in deps:
                        errors.append(f"Hallucinated npm package: {bad}")
            except Exception as e:
                errors.append(f"Invalid package.json: {e}")
    return (len(errors) == 0), "\n".join(errors)


def trigger_verify(repo_name: str):
    try:
        subprocess.run(
            [
                "gh", "workflow", "run", "verify-until-green.yml",
                "-f", f"target_repo={repo_name}",
                "-f", "attempt=1",
            ],
            check=False,
        )
        print(f"🚀 Triggered verify-until-green for {repo_name}")
    except Exception as e:
        print(f"⚠️ Could not trigger verify workflow: {e}")


def run_portfolio_builder():
    print("\n" + "=" * 60)
    print("  AGENT 1: PORTFOLIO BUILDER (30h gate, silent draft)")
    print("=" * 60)

    limiter.check()
    history = load_history()

    elapsed = hours_since(history.get("last_success_at"))
    if elapsed < PORTFOLIO_EVERY_HOURS:
        print(f"⏳ Skip: only {elapsed:.1f}h since last successful project (need {PORTFOLIO_EVERY_HOURS}h).")
        return None

    if history.get("in_progress"):
        print(f"⏳ Skip: project already in progress: {history['in_progress']}")
        return None

    project = pick_next_project(history)
    print(f"🎯 Building: {project['name']}")
    print(f"📚 Stack: {project['stack']}")

    llm = get_llm()
    all_files = {}

    # ─── STEP 1: DEVOPS & DOCS ───
    print("\n📝 [1/3] Generating README & Docker setup...")
    devops_agent = Agent(
        role="DevOps Architect and Technical Writer",
        goal="Create professional README.md, docker-compose.yml, and config files.",
        backstory="You write clean docker setups, professional READMEs, and CI configs.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )
    devops_task = Task(
        description=(
            f"Create docs and Docker config for: {project['name']}\n"
            f"Tech Stack: {project['stack']}\n\n"
            "Format each file EXACTLY like this:\n\n"
            "FILE: README.md\n"
            "```markdown\n"
            f"# {project['name']}\n\n"
            f"## Overview\n{project['why']}\n\n"
            f"## Tech Stack\n{project['stack']}\n\n"
            "## Quick Start\n```bash\ndocker-compose up\n```\n"
            "```\n\n"
            "FILE: docker-compose.yml\n"
            "```yaml\n"
            "version: '3.8'\n"
            "services:\n"
            "  backend:\n    build: ./backend\n    ports: ['8000:8000']\n"
            "  frontend:\n    build: ./frontend\n    ports: ['3000:3000']\n"
            "```\n\n"
            "FILE: .gitignore\n"
            "```text\n"
            "__pycache__/\nnode_modules/\n.env\nvenv/\n"
            "```\n"
        ),
        expected_output="Files with FILE headers.",
        agent=devops_agent,
    )
    res_1 = str(Crew(agents=[devops_agent], tasks=[devops_task], process=Process.sequential).kickoff())
    files_1 = parse_files_from_markdown(res_1)
    if not files_1:
        files_1 = {
            "README.md": f"# {project['name']}\n\n{project['why']}\n\n## Stack\n{project['stack']}\n",
            "docker-compose.yml": "version: '3.8'\nservices:\n  backend:\n    build: ./backend\n    ports: ['8000:8000']\n  frontend:\n    build: ./frontend\n    ports: ['3000:3000']\n",
            ".gitignore": "__pycache__/\nnode_modules/\n.env\nvenv/\n",
        }
    all_files.update(files_1)
    print(f"✅ Stage 1 generated {len(files_1)} files.")

    # ─── STEP 2: BACKEND ───
    print("\n⚙️ [2/3] Generating Backend...")
    backend_agent = Agent(
        role="Senior Backend Engineer",
        goal="Write runnable FastAPI backend with /health, no relative imports, no required DB.",
        backstory="You produce production-simple backends that boot with uvicorn without external services.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )
    backend_task = Task(
        description=(
            f"Write backend files for {project['name']}.\n"
            "Rules:\n"
            "- Single backend/main.py must run with: python -m uvicorn main:app --app-dir backend\n"
            "- Absolutely no relative imports (no `from .something`)\n"
            "- Must include a /health endpoint returning {'status':'ok'}\n"
            "- No mandatory Postgres/Redis to boot\n\n"
            "Format:\n"
            "FILE: backend/main.py\n"
            "```python\n"
            "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n"
            "app = FastAPI()\n"
            "app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])\n"
            "@app.get('/')\ndef root():\n    return {'status':'online'}\n"
            "@app.get('/health')\ndef health():\n    return {'status':'ok'}\n"
            "```\n\n"
            "FILE: backend/requirements.txt\n"
            "```text\n"
            "fastapi\nuvicorn[standard]\npydantic\npython-dotenv\n"
            "```\n\n"
            "FILE: backend/Dockerfile\n"
            "```dockerfile\n"
            "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\n"
            "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
            "```\n"
        ),
        expected_output="Backend files with FILE headers.",
        agent=backend_agent,
    )
    res_2 = str(Crew(agents=[backend_agent], tasks=[backend_task], process=Process.sequential).kickoff())
    files_2 = parse_files_from_markdown(res_2)
    if not files_2:
        files_2 = {
            "backend/main.py": (
                "from fastapi import FastAPI\n"
                "from fastapi.middleware.cors import CORSMiddleware\n"
                "app = FastAPI()\n"
                "app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])\n"
                "@app.get('/')\ndef root():\n    return {'status':'online'}\n"
                "@app.get('/health')\ndef health():\n    return {'status':'ok'}\n"
            ),
            "backend/requirements.txt": "fastapi\nuvicorn[standard]\npydantic\npython-dotenv\n",
            "backend/Dockerfile": (
                "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\n"
                "RUN pip install -r requirements.txt\nCOPY . .\n"
                'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
            ),
        }
    all_files.update(files_2)
    print(f"✅ Stage 2 generated {len(files_2)} files.")

    # ─── STEP 3: FRONTEND ───
    print("\n🎨 [3/3] Generating Frontend...")
    frontend_agent = Agent(
        role="Senior Frontend Engineer",
        goal="Write buildable Next.js frontend using only real npm packages.",
        backstory="You avoid hallucinated packages and use next/react/react-dom only for base setup.",
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )
    frontend_task = Task(
        description=(
            f"Write frontend files for {project['name']}.\n"
            "Rules:\n"
            "- package.json must only use real packages: next, react, react-dom\n"
            "- No hallucinated packages like yjs-react\n"
            "- Must be buildable with: npm install && npm run build\n\n"
            "Format:\n"
            "FILE: frontend/package.json\n"
            "```json\n"
            "{\n"
            "  \"name\": \"frontend\",\n"
            "  \"private\": true,\n"
            "  \"scripts\": {\"dev\":\"next dev\",\"build\":\"next build\",\"start\":\"next start\"},\n"
            "  \"dependencies\": {\"next\":\"14.1.0\",\"react\":\"^18.2.0\",\"react-dom\":\"^18.2.0\"}\n"
            "}\n"
            "```\n\n"
            "FILE: frontend/src/app/layout.tsx\n"
            "```tsx\n"
            "export default function RootLayout({children}: {children: React.ReactNode}) {\n"
            "  return (<html><body>{children}</body></html>);\n"
            "}\n"
            "```\n\n"
            "FILE: frontend/src/app/page.tsx\n"
            "```tsx\n"
            "export default function Home() {\n"
            f"  return (<main style={{{{padding: 24}}}}><h1>{project['name']}</h1><p>{project['why']}</p></main>);\n"
            "}\n"
            "```\n\n"
            "FILE: frontend/next.config.js\n"
            "```js\n"
            "/** @type {import('next').NextConfig} */\n"
            "const nextConfig = { reactStrictMode: true };\n"
            "module.exports = nextConfig;\n"
            "```\n\n"
            "FILE: frontend/tsconfig.json\n"
            "```json\n"
            "{\n"
            "  \"compilerOptions\": {\n"
            "    \"target\": \"es5\",\n"
            "    \"lib\": [\"dom\", \"dom.iterable\", \"esnext\"],\n"
            "    \"allowJs\": true,\n"
            "    \"skipLibCheck\": true,\n"
            "    \"strict\": false,\n"
            "    \"forceConsistentCasingInFileNames\": true,\n"
            "    \"noEmit\": true,\n"
            "    \"esModuleInterop\": true,\n"
            "    \"module\": \"esnext\",\n"
            "    \"moduleResolution\": \"node\",\n"
            "    \"resolveJsonModule\": true,\n"
            "    \"isolatedModules\": true,\n"
            "    \"jsx\": \"preserve\",\n"
            "    \"incremental\": true,\n"
            "    \"plugins\": [{ \"name\": \"next\" }]\n"
            "  },\n"
            "  \"include\": [\"next-env.d.ts\", \"**/*.ts\", \"**/*.tsx\"],\n"
            "  \"exclude\": [\"node_modules\"]\n"
            "}\n"
            "```\n\n"
            "FILE: frontend/Dockerfile\n"
            "```dockerfile\n"
            "FROM node:20-alpine\nWORKDIR /app\nCOPY package.json .\nRUN npm install\nCOPY . .\n"
            'CMD ["npm", "run", "dev"]\n'
            "```\n"
        ),
        expected_output="Frontend files with FILE headers.",
        agent=frontend_agent,
    )
    res_3 = str(Crew(agents=[frontend_agent], tasks=[frontend_task], process=Process.sequential).kickoff())
    files_3 = parse_files_from_markdown(res_3)
    if not files_3:
        files_3 = {
            "frontend/package.json": (
                "{\n"
                "  \"name\": \"frontend\",\n"
                "  \"private\": true,\n"
                "  \"scripts\": {\"dev\":\"next dev\",\"build\":\"next build\",\"start\":\"next start\"},\n"
                "  \"dependencies\": {\"next\":\"14.1.0\",\"react\":\"^18.2.0\",\"react-dom\":\"^18.2.0\"}\n"
                "}\n"
            ),
            "frontend/src/app/layout.tsx": (
                "export default function RootLayout({children}: {children: React.ReactNode}) {\n"
                "  return (<html><body>{children}</body></html>);\n"
                "}\n"
            ),
            "frontend/src/app/page.tsx": (
                "export default function Home() {\n"
                f"  return (<main style={{{{padding: 24}}}}><h1>{project['name']}</h1></main>);\n"
                "}\n"
            ),
            "frontend/next.config.js": (
                "/** @type {import('next').NextConfig} */\n"
                "const nextConfig = { reactStrictMode: true };\n"
                "module.exports = nextConfig;\n"
            ),
            "frontend/tsconfig.json": (
                "{\n"
                "  \"compilerOptions\": {\n"
                "    \"target\": \"es5\",\n"
                "    \"lib\": [\"dom\", \"dom.iterable\", \"esnext\"],\n"
                "    \"allowJs\": true,\n"
                "    \"skipLibCheck\": true,\n"
                "    \"strict\": false,\n"
                "    \"forceConsistentCasingInFileNames\": true,\n"
                "    \"noEmit\": true,\n"
                "    \"esModuleInterop\": true,\n"
                "    \"module\": \"esnext\",\n"
                "    \"moduleResolution\": \"node\",\n"
                "    \"resolveJsonModule\": true,\n"
                "    \"isolatedModules\": true,\n"
                "    \"jsx\": \"preserve\",\n"
                "    \"incremental\": true,\n"
                "    \"plugins\": [{ \"name\": \"next\" }]\n"
                "  },\n"
                "  \"include\": [\"next-env.d.ts\", \"**/*.ts\", \"**/*.tsx\"],\n"
                "  \"exclude\": [\"node_modules\"]\n"
                "}\n"
            ),
            "frontend/Dockerfile": (
                "FROM node:20-alpine\nWORKDIR /app\nCOPY package.json .\nRUN npm install\nCOPY . .\n"
                'CMD ["npm", "run", "dev"]\n'
            ),
        }
    all_files.update(files_3)
    print(f"✅ Stage 3 generated {len(files_3)} files.")

    # ─── STATIC SELF-HEAL LOOP (bounded 5 tries) ───
    for i in range(5):
        ok, err = verify_static(all_files)
        if ok:
            break
        print(f"🩹 Static repair round {i+1}: {err}")
        fix_agent = Agent(
            role="Repair Engineer",
            goal="Fix broken generated files based on validation errors",
            backstory="You fix syntax errors, remove relative imports, remove fake packages.",
            llm=llm,
            max_iter=4,
            verbose=True,
        )
        fix_task = Task(
            description=(
                "Return ONLY the corrected full files that fix these errors:\n"
                f"{err}\n\n"
                "Current files:\n"
                + "\n".join([f"FILE: {p}\n```\n{c[:1500]}\n```" for p, c in list(all_files.items())[:8]])
            ),
            expected_output="Corrected files as FILE codeblocks",
            agent=fix_agent,
        )
        fixed = parse_files_from_markdown(
            str(Crew(agents=[fix_agent], tasks=[fix_task], process=Process.sequential).kickoff())
        )
        if fixed:
            all_files.update(fixed)
        else:
            break

    # ─── PUSH DRAFT TO GITHUB (SILENT, NO TELEGRAM) ───
    repo_name = project["slug"]
    description = f"{project['name']} — {project['why']}"
    print(f"\n📦 Pushing draft {len(all_files)} files to '{repo_name}' (no Telegram yet)...")

    try:
        repo_url = github_mgr.push_files(repo_name, all_files, description)

        # mark in-progress; success only after CI green
        history["in_progress"] = {
            "slug": repo_name,
            "started_at": now_iso(),
            "repo_url": repo_url,
        }
        save_history(history)

        # kick off closed-loop verify (no notify yet)
        trigger_verify(repo_name)
        print(f"⏳ Waiting for verify-until-green to confirm project before notifying.")
        return repo_url

    except Exception as e:
        print(f"❌ Push failed: {e}")
        return None


if __name__ == "__main__":
    run_portfolio_builder()