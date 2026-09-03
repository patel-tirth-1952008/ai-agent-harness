"""
The AI Agent Harness Scheduler.
Runs all 4 agents on a schedule in the background.

Usage:
    python scheduler.py
    
Keep this running (or set it up as a system service).
"""

import time
import schedule
from datetime import datetime
from limiter import limiter
from notifier import send_notification
from config import (PORTFOLIO_BUILD_DAY, PORTFOLIO_BUILD_TIME,
                    LEETCODE_SOLVE_TIME, FREELANCE_SEARCH_TIME,
                    JOB_SEARCH_TIME)


def safe_run(agent_name: str, agent_function, *args, **kwargs):
    """Wrapper that catches all errors to prevent scheduler from crashing."""
    try:
        print(f"\n{'=' * 60}")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Scheduled run: {agent_name}")
        print(f"📊 Limiter status: {limiter.get_status()}")
        print(f"{'=' * 60}")

        limiter.reset_run()
        agent_function(*args, **kwargs)

        print(f"✅ {agent_name} completed successfully.")

    except Exception as e:
        error_msg = f"❌ {agent_name} failed: {str(e)}"
        print(error_msg)
        try:
            send_notification(f"⚠️ *Agent Error*\n{error_msg}")
        except Exception:
            pass


# ─── SCHEDULED JOBS ───
def portfolio_job():
    from agent1_portfolio_builder import run_portfolio_builder
    safe_run("Portfolio Builder", run_portfolio_builder)


def leetcode_job():
    from agent2_leetcode_solver import solve_leetcode_problem
    safe_run("LeetCode Solver", solve_leetcode_problem)


def freelance_job():
    from agent3_freelance_finder import run_freelance_finder
    safe_run("Freelance Finder", run_freelance_finder)


def job_finder_job():
    from agent4_job_finder import run_job_finder
    safe_run("Job Finder", run_job_finder)


# ─── SETUP SCHEDULE ───
def setup_schedule():
    # Agent 1: Portfolio Builder — Once per week
    getattr(schedule.every(), PORTFOLIO_BUILD_DAY).at(PORTFOLIO_BUILD_TIME).do(portfolio_job)

    # Agent 2: LeetCode Solver — Every day
    schedule.every().day.at(LEETCODE_SOLVE_TIME).do(leetcode_job)

    # Agent 3: Freelance Finder — Twice per day
    schedule.every().day.at(FREELANCE_SEARCH_TIME).do(freelance_job)
    schedule.every().day.at("18:00").do(freelance_job)

    # Agent 4: Job Finder — Once per day
    schedule.every().day.at(JOB_SEARCH_TIME).do(job_finder_job)

    print("\n" + "=" * 60)
    print("🤖 AI AGENT HARNESS — ACTIVE")
    print("=" * 60)
    print(f"\n📅 Schedule:")
    print(f"   🏗️  Portfolio Builder : Every {PORTFOLIO_BUILD_DAY.capitalize()} at {PORTFOLIO_BUILD_TIME}")
    print(f"   🧠 LeetCode Solver   : Every day at {LEETCODE_SOLVE_TIME}")
    print(f"   💼 Freelance Finder  : Every day at {FREELANCE_SEARCH_TIME} & 18:00")
    print(f"   🎯 Job Finder        : Every day at {JOB_SEARCH_TIME}")
    print(f"\n📊 API Limits: {limiter.max_per_day} calls/day, {limiter.max_per_run} calls/run")
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Press Ctrl+C to stop the harness.")
    print("=" * 60)

    # Send startup notification
    send_notification(
        "🤖 *AI Agent Harness Started!*\n\n"
        f"📅 Portfolio: {PORTFOLIO_BUILD_DAY.capitalize()} at {PORTFOLIO_BUILD_TIME}\n"
        f"🧠 LeetCode: Daily at {LEETCODE_SOLVE_TIME}\n"
        f"💼 Freelance: Daily at {FREELANCE_SEARCH_TIME} & 18:00\n"
        f"🎯 Jobs: Daily at {JOB_SEARCH_TIME}\n\n"
        f"All systems operational. ✅"
    )


# ─── MAIN LOOP ───
if __name__ == "__main__":
    setup_schedule()

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("\n\n🛑 Harness stopped by user.")
            send_notification("🛑 AI Agent Harness has been stopped.")
            break
        except Exception as e:
            print(f"⚠️ Scheduler error: {e}")
            time.sleep(60)  # Wait a minute and retry