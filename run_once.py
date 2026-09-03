"""
Run any agent manually for testing.
Usage:
    python run_once.py portfolio
    python run_once.py leetcode
    python run_once.py freelance
    python run_once.py jobs
    python run_once.py all
"""

import sys
from limiter import limiter


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_once.py [portfolio|leetcode|freelance|jobs|all]")
        print("\nExamples:")
        print("  python run_once.py portfolio     → Build 1 project and push to GitHub")
        print("  python run_once.py leetcode      → Solve 1 LeetCode problem")
        print("  python run_once.py freelance     → Search for freelance gigs")
        print("  python run_once.py jobs          → Search for full-time jobs")
        print("  python run_once.py all           → Run all 4 agents")
        return

    agent = sys.argv[1].lower()

    print(f"\n📊 Current API Limiter Status: {limiter.get_status()}")

    if agent in ("portfolio", "all"):
        print("\n🏗️ Running Portfolio Builder...")
        from agent1_portfolio_builder import run_portfolio_builder
        run_portfolio_builder()
        limiter.reset_run()

    if agent in ("leetcode", "all"):
        print("\n🧠 Running LeetCode Solver...")
        from agent2_leetcode_solver import solve_leetcode_problem
        solve_leetcode_problem()
        limiter.reset_run()

    if agent in ("freelance", "all"):
        print("\n💼 Running Freelance Finder...")
        from agent3_freelance_finder import run_freelance_finder
        run_freelance_finder()
        limiter.reset_run()

    if agent in ("jobs", "all"):
        print("\n🎯 Running Job Finder...")
        from agent4_job_finder import run_job_finder
        run_job_finder()
        limiter.reset_run()

    print(f"\n📊 Final API Limiter Status: {limiter.get_status()}")
    print("✅ Done!")


if __name__ == "__main__":
    main()