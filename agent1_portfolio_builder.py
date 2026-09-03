import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
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
    return ChatGroq(
        temperature=temperature,
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        max_tokens=8000,
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


# ─── HIGH-VALUE PROJECT IDEAS (Curated from real FAANG interview feedback) ───
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
        "stack": "Node.js, Express, MongoDB, RabbitMQ, Stripe, React, Docker, Kubernetes manifests"
    },
    {
        "slug": "job-board-with-search",
        "name": "Job Board with Advanced Search & Filters",
        "why": "LinkedIn clone. Tests full-text search (Elasticsearch), pagination, complex queries.",
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
    """Pick the next project not yet built."""
    built = set(history.get("built_projects", []))
    for project in INDUSTRY_PROJECTS:
        if project["slug"] not in built:
            return project
    # If all built, cycle back
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

    # ─── AGENT 1: SYSTEM ARCHITECT ───
    architect = Agent(
        role="Principal Software Architect at FAANG",
        goal="Design production-grade full-stack systems with proper architecture, folder structure, and industry best practices.",
        backstory=(
            "You have 20 years of experience architecting systems at Google, Amazon, and Meta. "
            "You design systems that are: scalable, testable, secure, and maintainable. "
            "You always include: database schemas, API contracts, authentication, error handling, "
            "logging, tests, Docker setup, CI/CD, and comprehensive documentation. "
            "You NEVER build toy projects or demos. Every project you design could run in production."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    # ─── AGENT 2: BACKEND ENGINEER ───
    backend_dev = Agent(
        role="Senior Backend Engineer",
        goal="Write complete, production-ready backend code with authentication, database, tests, and error handling.",
        backstory=(
            "You write backend code that handles: authentication (JWT), database ORM with migrations, "
            "input validation, error handling with proper HTTP codes, rate limiting, logging, "
            "unit tests, integration tests, environment configuration, and API documentation. "
            "Every endpoint you write has proper error handling and tests."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    # ─── AGENT 3: FRONTEND ENGINEER ───
    frontend_dev = Agent(
        role="Senior Frontend Engineer",
        goal="Build modern, responsive frontends with React/Next.js, proper state management, and clean UI.",
        backstory=(
            "You write frontend code using: TypeScript, Tailwind CSS, proper component structure, "
            "React hooks, state management, API integration with error handling, loading states, "
            "form validation, responsive design, and dark mode support. "
            "Your UIs look modern and professional."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    # ─── AGENT 4: DEVOPS + DOCS ENGINEER ───
    devops_engineer = Agent(
        role="DevOps Engineer and Technical Writer",
        goal="Create Docker setup, CI/CD pipelines, and professional documentation.",
        backstory=(
            "You create: Dockerfile for each service, docker-compose.yml for one-command setup, "
            "GitHub Actions workflows for testing, environment file templates, deployment guides "
            "for Render/Vercel, Postman collections, and comprehensive READMEs with architecture "
            "diagrams (in ASCII/mermaid), API documentation, and screenshots section."
        ),
        llm=llm,
        max_iter=MAX_AGENT_ITERATIONS,
        max_rpm=MAX_AGENT_RPM,
        verbose=True,
    )

    # ─── TASK 1: ARCHITECTURE DESIGN ───
    architecture_task = Task(
        description=(
            f"Design the complete architecture for: {project['name']}\n\n"
            f"Tech stack to use: {project['stack']}\n\n"
            f"You MUST output a detailed architecture specification including:\n\n"
            f"1. **Complete folder structure** (as a tree diagram)\n"
            f"   - Monorepo with `/backend`, `/frontend`, `/docker`, `/docs`\n"
            f"   - All subfolders (models, routes, controllers, services, tests, etc.)\n\n"
            f"2. **Database schema** (all tables/collections with fields and relationships)\n\n"
            f"3. **API endpoints** (list ALL endpoints with method, path, auth requirement)\n"
            f"   - Minimum 15 endpoints covering CRUD, auth, business logic\n\n"
            f"4. **Authentication flow** (JWT with refresh tokens)\n\n"
            f"5. **Frontend pages/routes** (list all pages needed)\n\n"
            f"6. **Docker services** (backend, frontend, database, cache)\n\n"
            f"7. **Environment variables** needed\n\n"
            f"Be extremely detailed. This design will be used to generate ALL the code."
        ),
        expected_output="Complete architecture specification with folder tree, DB schema, API endpoints, auth flow, and Docker setup.",
        agent=architect,
    )

    # ─── TASK 2: BACKEND CODE ───
    backend_task = Task(
        description=(
            f"Based on the architecture, write the COMPLETE backend code for: {project['name']}\n\n"
            f"You MUST output a valid JSON object with this exact structure:\n\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "backend/README.md": "...",\n'
            f'    "backend/requirements.txt or package.json": "...",\n'
            f'    "backend/main.py or index.js": "...",\n'
            f'    "backend/config.py or config.js": "...",\n'
            f'    "backend/models/user.py": "...",\n'
            f'    "backend/models/[entity].py": "...",\n'
            f'    "backend/routes/auth.py": "...",\n'
            f'    "backend/routes/[feature].py": "...",\n'
            f'    "backend/services/[service].py": "...",\n'
            f'    "backend/middleware/auth_middleware.py": "...",\n'
            f'    "backend/utils/security.py": "...",\n'
            f'    "backend/database/connection.py": "...",\n'
            f'    "backend/database/migrations/001_initial.sql": "...",\n'
            f'    "backend/tests/test_auth.py": "...",\n'
            f'    "backend/tests/test_[feature].py": "...",\n'
            f'    "backend/.env.example": "...",\n'
            f'    "backend/Dockerfile": "..."\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"REQUIREMENTS:\n"
            f"1. Include AT LEAST 15 backend files\n"
            f"2. All code must be syntactically correct and runnable\n"
            f"3. Include JWT authentication with proper password hashing\n"
            f"4. Include input validation using Pydantic (Python) or Joi (Node.js)\n"
            f"5. Include proper error handling with HTTP status codes\n"
            f"6. Include at least 3 test files with real tests\n"
            f"7. Include database migrations\n"
            f"8. Use environment variables for all secrets\n"
            f"9. Include a Dockerfile\n\n"
            f"Output ONLY valid JSON. No text before or after."
        ),
        expected_output="Valid JSON with at least 15 backend files, all production-quality.",
        agent=backend_dev,
    )

    # ─── TASK 3: FRONTEND CODE ───
    frontend_task = Task(
        description=(
            f"Based on the architecture, write the COMPLETE frontend code for: {project['name']}\n\n"
            f"You MUST output a valid JSON object with this exact structure:\n\n"
            f"```json\n"
            f"{{\n"
            f'  "files": {{\n'
            f'    "frontend/package.json": "...",\n'
            f'    "frontend/tsconfig.json": "...",\n'
            f'    "frontend/next.config.js or vite.config.ts": "...",\n'
            f'    "frontend/tailwind.config.js": "...",\n'
            f'    "frontend/src/app/layout.tsx": "...",\n'
            f'    "frontend/src/app/page.tsx": "...",\n'
            f'    "frontend/src/app/login/page.tsx": "...",\n'
            f'    "frontend/src/app/register/page.tsx": "...",\n'
            f'    "frontend/src/app/dashboard/page.tsx": "...",\n'
            f'    "frontend/src/components/Navbar.tsx": "...",\n'
            f'    "frontend/src/components/[Component].tsx": "...",\n'
            f'    "frontend/src/lib/api.ts": "...",\n'
            f'    "frontend/src/lib/auth.ts": "...",\n'
            f'    "frontend/src/hooks/useAuth.ts": "...",\n'
            f'    "frontend/src/types/index.ts": "...",\n'
            f'    "frontend/.env.example": "...",\n'
            f'    "frontend/Dockerfile": "..."\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"REQUIREMENTS:\n"
            f"1. Use Next.js 14 App Router with TypeScript\n"
            f"2. Use Tailwind CSS for styling\n"
            f"3. Include AT LEAST 15 frontend files\n"
            f"4. Include proper authentication (login/register/protected routes)\n"
            f"5. Include API integration with error handling and loading states\n"
            f"6. Include form validation\n"
            f"7. Include responsive design (mobile-first)\n"
            f"8. Include a Dockerfile\n\n"
            f"Output ONLY valid JSON. No text before or after."
        ),
        expected_output="Valid JSON with at least 15 frontend files, all modern and production-quality.",
        agent=frontend_dev,
    )

    # ─── TASK 4: DEVOPS + DOCS ───
    devops_task = Task(
        description=(
            f"Create the DevOps files and documentation for: {project['name']}\n\n"
            f"You MUST output a valid JSON object with this exact structure:\n\n"
            f"```json\n"
            f"{{\n"
            f'  "repo_name": "{project["slug"]}",\n'
            f'  "description": "One-line project description",\n'
            f'  "files": {{\n'
            f'    "README.md": "Comprehensive README with all sections",\n'
            f'    "docker-compose.yml": "...",\n'
            f'    ".gitignore": "...",\n'
            f'    ".github/workflows/ci.yml": "...",\n'
            f'    "docs/ARCHITECTURE.md": "Architecture with mermaid diagram",\n'
            f'    "docs/API.md": "Full API documentation",\n'
            f'    "docs/DEPLOYMENT.md": "Deployment guide for Render/Vercel",\n'
            f'    "docs/SETUP.md": "Local setup guide",\n'
            f'    "postman_collection.json": "Postman collection for all APIs",\n'
            f'    "Makefile": "make install, make test, make dev, make prod"\n'
            f'  }}\n'
            f"}}\n"
            f"```\n\n"
            f"REQUIREMENTS FOR README.md:\n"
            f"- Project title with badges (build status, license, etc.)\n"
            f"- Live demo link section (placeholder)\n"
            f"- Screenshots section (placeholder markdown for images)\n"
            f"- Features list (10+ features)\n"
            f"- Tech stack with icons/badges\n"
            f"- Architecture diagram (ASCII or mermaid)\n"
            f"- Quick start (docker-compose up)\n"
            f"- Detailed setup for local development\n"
            f"- API documentation summary\n"
            f"- Testing instructions\n"
            f"- Deployment guide\n"
            f"- Contributing guidelines\n"
            f"- License\n\n"
            f"REQUIREMENTS FOR docker-compose.yml:\n"
            f"- Backend service\n"
            f"- Frontend service\n"
            f"- Database (PostgreSQL/MongoDB)\n"
            f"- Redis cache\n"
            f"- Proper networking and volumes\n"
            f"- Environment variables from .env\n\n"
            f"REQUIREMENTS FOR .github/workflows/ci.yml:\n"
            f"- Run backend tests\n"
            f"- Run frontend build check\n"
            f"- Run linters\n\n"
            f"Output ONLY valid JSON. No text before or after."
        ),
        expected_output="Valid JSON with repo_name, description, and all devops/docs files.",
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

        # Extract JSON blocks from each task output
        all_files = {}
        repo_name = project["slug"]
        description = f"{project['name']} — {project['why']}"

        # Find all JSON blocks in the combined result
        json_blocks = re.findall(r'\{[\s\S]*?\}(?=\s*(?:\{|\Z))', result)

        for block in json_blocks:
            try:
                data = json.loads(block)
                if "files" in data:
                    all_files.update(data["files"])
                if "repo_name" in data:
                    repo_name = data["repo_name"]
                if "description" in data:
                    description = data["description"]
            except json.JSONDecodeError:
                continue

        # Fallback: try to find one big JSON
        if not all_files:
            match = re.search(r'\{[\s\S]*"files"[\s\S]*\}', result)
            if match:
                try:
                    data = json.loads(match.group(0))
                    all_files = data.get("files", {})
                    repo_name = data.get("repo_name", repo_name)
                    description = data.get("description", description)
                except Exception:
                    pass

        if not all_files:
            send_agent_report(
                "Portfolio Builder", "error",
                f"Failed to extract files from AI output for {project['name']}. Try again next cycle."
            )
            return None

        print(f"\n📦 Extracted {len(all_files)} files. Pushing to GitHub...")

        # Push to GitHub
        repo_url = github_mgr.push_files(repo_name, all_files, description)

        # Update history
        history["built_projects"].append(project["slug"])
        save_history(history)

        send_agent_report(
            "Portfolio Builder", "success",
            f"Built: {project['name']}\n"
            f"Files: {len(all_files)}\n"
            f"Stack: {project['stack']}\n"
            f"URL: {repo_url}\n\n"
            f"⚠️ IMPORTANT: Clone locally, run 'docker-compose up', "
            f"fix any small bugs, then it is interview-ready!"
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