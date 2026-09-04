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
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_AGENT_ITERATIONS,
    PORTFOLIO_HISTORY_FILE,
)
from limiter import limiter
from github_manager import github_mgr

# 1 project per day
PORTFOLIO_EVERY_HOURS = 24


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
    data.setdefault("all_projects_completed_notified", False)
    return data


def save_history(history):
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─── 50 INDUSTRY-GRADE PORTFOLIO PROJECTS ───
INDUSTRY_PROJECTS = [
    {"slug": "url-shortener-analytics", "name": "URL Shortener with Analytics", "why": "Bitly-style system design: hashing, redirects, click analytics.", "stack": "FastAPI, Next.js, SQLite, Docker"},
    {"slug": "realtime-collaborative-editor", "name": "Real-time Collaborative Editor", "why": "Google Docs-style websockets collaboration.", "stack": "FastAPI, WebSockets, Next.js, Docker"},
    {"slug": "chat-app-rooms", "name": "Real-time Chat with Rooms", "why": "Slack/WhatsApp fundamentals: rooms, presence, messages.", "stack": "FastAPI, WebSockets, Next.js, Docker"},
    {"slug": "trello-task-board", "name": "Task Board SaaS (Trello Clone)", "why": "Kanban boards, auth, team permissions.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "job-board-filters", "name": "Job Board with Advanced Filters", "why": "LinkedIn-like search, filters, applications.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "ecommerce-storefront", "name": "E-commerce Store + Cart + Checkout", "why": "Catalog, cart, orders, checkout flow.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "banking-ledger-api", "name": "Banking Ledger & Transfers API", "why": "Transactions, auth, audit logs.", "stack": "FastAPI, SQLite, JWT, Docker"},
    {"slug": "expense-splitter", "name": "Expense Splitter (Splitwise Clone)", "why": "Groups, balances, settlements.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "blog-cms-markdown", "name": "Markdown Blog CMS", "why": "CRUD, auth, SEO pages, rich content.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "notes-app-sync", "name": "Notes App with Sync API", "why": "CRUD notes, tags, search, auth.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "file-storage-share", "name": "Cloud File Storage & Sharing", "why": "Upload, share links, expiry, quotas.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "image-gallery-cdn", "name": "Image Gallery with Processing", "why": "Uploads, thumbnails, albums.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "video-meta-platform", "name": "Video Metadata Platform", "why": "Upload metadata, player UI, comments.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "notification-center", "name": "Multi-channel Notification Center", "why": "In-app notifications + delivery logs.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "email-campaign-tool", "name": "Email Campaign Dashboard", "why": "Templates, audiences, campaign stats UI.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "feature-flag-service", "name": "Feature Flag Service", "why": "Flags, targeting rules, rollout percentages.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "api-rate-limiter-gateway", "name": "API Gateway + Rate Limiter", "why": "Token bucket/sliding window, API keys, routing.", "stack": "FastAPI, Docker"},
    {"slug": "auth-service-oauth", "name": "Auth Service (JWT + OAuth-ready)", "why": "Register/login/refresh tokens/roles.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "rbac-admin-panel", "name": "RBAC Admin Panel", "why": "Users, roles, permissions matrix.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "audit-log-platform", "name": "Audit Log Platform", "why": "Append-only event logs, filters, export.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "inventory-management", "name": "Inventory Management System", "why": "SKU, stock in/out, low-stock alerts.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "pos-billing-system", "name": "POS Billing System", "why": "Products, invoices, taxes, receipts.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "crm-pipeline", "name": "CRM Sales Pipeline", "why": "Leads, stages, activities, dashboard.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "helpdesk-ticketing", "name": "Helpdesk Ticketing System", "why": "Tickets, priorities, assignments, SLA timers.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "knowledge-base-search", "name": "Knowledge Base with Search", "why": "Articles, categories, full-text search UI.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "form-builder", "name": "Dynamic Form Builder", "why": "Create forms, collect responses, export CSV.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "survey-analytics", "name": "Survey & Analytics Platform", "why": "Surveys, charts, response insights.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "booking-scheduler", "name": "Appointment Booking Scheduler", "why": "Slots, bookings, cancellations, calendar UI.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "event-ticketing", "name": "Event Ticketing System", "why": "Events, tickets, orders, QR-ready flow.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "hotel-reservation", "name": "Hotel Reservation System", "why": "Rooms, availability, booking workflow.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "food-delivery-tracker", "name": "Food Delivery + Order Tracking", "why": "Restaurants, cart, order states, live status.", "stack": "Next.js, FastAPI, WebSockets, Docker"},
    {"slug": "ride-fare-estimator", "name": "Ride Fare Estimator + Trips API", "why": "Pricing rules, trip lifecycle, roles.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "workout-tracker", "name": "Fitness Workout Tracker", "why": "Plans, logs, progress charts.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "habit-tracker", "name": "Habit Tracker with Streaks", "why": "Daily habits, streaks, reminders API.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "personal-finance-dashboard", "name": "Personal Finance Dashboard", "why": "Accounts, transactions, budgets, charts.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "invoice-generator", "name": "Invoice Generator SaaS", "why": "Clients, invoices, PDF-ready UI/API.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "subscription-billing-demo", "name": "Subscription Billing Demo", "why": "Plans, trials, customer portal flow.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "saas-waitlist-referral", "name": "SaaS Waitlist + Referral System", "why": "Waitlist, referral codes, leaderboard.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "multi-tenant-notes", "name": "Multi-tenant Notes SaaS", "why": "Tenant isolation, members, roles.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "api-status-page", "name": "Status Page & Uptime Monitor", "why": "Service checks, incidents, public status page.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "log-ingestion-dashboard", "name": "Log Ingestion Dashboard", "why": "Ingest logs API, search/filter UI, severity charts.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "metrics-dashboard", "name": "DevOps Metrics Dashboard", "why": "Custom metrics API + realtime charts.", "stack": "Next.js, FastAPI, WebSockets, Docker"},
    {"slug": "ci-cd-mini-dashboard", "name": "Mini CI/CD Run Dashboard", "why": "Pipeline runs, statuses, logs viewer.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "secret-manager-lite", "name": "Secrets Manager Lite", "why": "Encrypted secrets CRUD, access policies.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "webhook-debugger", "name": "Webhook Debugger & Inspector", "why": "Receive webhooks, inspect payloads, replay.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "graphql-bff-shop", "name": "GraphQL BFF for Shop APIs", "why": "BFF pattern, schema, frontend integration.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "search-autocomplete-service", "name": "Search Autocomplete Service", "why": "Prefix search, ranking, debounced UI.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "recommendation-demo", "name": "Simple Recommendation Engine Demo", "why": "User-item interactions, top-N recommendations.", "stack": "FastAPI, Next.js, Docker"},
    {"slug": "chatbot-helpdesk", "name": "Helpdesk Chatbot + Ticket Escalation", "why": "Bot FAQ flow then human ticket creation.", "stack": "Next.js, FastAPI, Docker"},
    {"slug": "ai-support-summarizer", "name": "AI Support Ticket Summarizer", "why": "Tickets + summary endpoint + dashboard.", "stack": "Next.js, FastAPI, Docker"},
]


def pick_next_project(history):
    """Return next unbuilt project, or None if all 50 are done."""
    built = set(history.get("built_projects", []))
    remaining = [p for p in INDUSTRY_PROJECTS if p["slug"] not in built]

    if not remaining:
        if not history.get("all_projects_completed_notified"):
            try:
                from notifier import send_notification
                send_notification(
                    "✅ All 50 portfolio projects are completed!\n"
                    f"Finished: {len(built)}/{len(INDUSTRY_PROJECTS)}\n"
                    "Queue is empty. Add more projects when ready."
                )
            except Exception as e:
                print(f"⚠️ Completion notify failed: {e}")
            history["all_projects_completed_notified"] = True
            save_history(history)
        print("🎉 All 50 projects completed. No new project to build.")
        return None

    print(f"📊 Progress: {len(built)}/{len(INDUSTRY_PROJECTS)} completed | {len(remaining)} remaining")
    return remaining[0]


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
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
        raw_json = match.group(1) if match else re.search(r"\{[\s\S]*\}", text).group(0)
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
                "gh",
                "workflow",
                "run",
                "verify-until-green.yml",
                "-f",
                f"target_repo={repo_name}",
                "-f",
                "attempt=1",
            ],
            check=False,
        )
        print(f"🚀 Triggered verify-until-green for {repo_name}")
    except Exception as e:
        print(f"⚠️ Could not trigger verify workflow: {e}")


def run_portfolio_builder():
    print("\n" + "=" * 60)
    print("  AGENT 1: PORTFOLIO BUILDER (24h gate, 50 projects, silent draft)")
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
    if project is None:
        return None

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
            'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
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
            '  "name": "frontend",\n'
            '  "private": true,\n'
            '  "scripts": {"dev":"next dev","build":"next build","start":"next start"},\n'
            '  "dependencies": {"next":"14.1.0","react":"^18.2.0","react-dom":"^18.2.0"}\n'
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
            '  "compilerOptions": {\n'
            '    "target": "es5",\n'
            '    "lib": ["dom", "dom.iterable", "esnext"],\n'
            '    "allowJs": true,\n'
            '    "skipLibCheck": true,\n'
            '    "strict": false,\n'
            '    "forceConsistentCasingInFileNames": true,\n'
            '    "noEmit": true,\n'
            '    "esModuleInterop": true,\n'
            '    "module": "esnext",\n'
            '    "moduleResolution": "node",\n'
            '    "resolveJsonModule": true,\n'
            '    "isolatedModules": true,\n'
            '    "jsx": "preserve",\n'
            '    "incremental": true,\n'
            '    "plugins": [{ "name": "next" }]\n'
            "  },\n"
            '  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],\n'
            '  "exclude": ["node_modules"]\n'
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
                '  "name": "frontend",\n'
                '  "private": true,\n'
                '  "scripts": {"dev":"next dev","build":"next build","start":"next start"},\n'
                '  "dependencies": {"next":"14.1.0","react":"^18.2.0","react-dom":"^18.2.0"}\n'
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
                '  "compilerOptions": {\n'
                '    "target": "es5",\n'
                '    "lib": ["dom", "dom.iterable", "esnext"],\n'
                '    "allowJs": true,\n'
                '    "skipLibCheck": true,\n'
                '    "strict": false,\n'
                '    "forceConsistentCasingInFileNames": true,\n'
                '    "noEmit": true,\n'
                '    "esModuleInterop": true,\n'
                '    "module": "esnext",\n'
                '    "moduleResolution": "node",\n'
                '    "resolveJsonModule": true,\n'
                '    "isolatedModules": true,\n'
                '    "jsx": "preserve",\n'
                '    "incremental": true,\n'
                '    "plugins": [{ "name": "next" }]\n'
                "  },\n"
                '  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],\n'
                '  "exclude": ["node_modules"]\n'
                "}\n"
            ),
            "frontend/Dockerfile": (
                "FROM node:20-alpine\nWORKDIR /app\nCOPY package.json .\nRUN npm install\nCOPY . .\n"
                'CMD ["npm", "run", "dev"]\n'
            ),
        }
    all_files.update(files_3)
    print(f"✅ Stage 3 generated {len(files_3)} files.")

    # ─── STATIC SELF-HEAL LOOP ───
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

    # ─── PUSH DRAFT (NO TELEGRAM HERE) ───
    repo_name = project["slug"]
    description = f"{project['name']} — {project['why']}"
    print(f"\n📦 Pushing draft {len(all_files)} files to '{repo_name}' (Telegram only after CI green)...")

    try:
        repo_url = github_mgr.push_files(repo_name, all_files, description)

        history["in_progress"] = {
            "slug": repo_name,
            "started_at": now_iso(),
            "repo_url": repo_url,
        }
        save_history(history)

        trigger_verify(repo_name)
        print("⏳ Waiting for verify-until-green before final success notify.")
        return repo_url

    except Exception as e:
        print(f"❌ Push failed: {e}")
        return None


if __name__ == "__main__":
    run_portfolio_builder()