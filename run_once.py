# run_once.py
import os
import sys
import traceback

print("🤖 Orchestrating AI Agent Runner...")

# --- RESILIENT IMPORTS WITH FALLBACKS ---

# 1. Agent 1: Portfolio Builder
try:
    from agent1_portfolio_builder import run_portfolio_builder
except ImportError as e:
    print(f"⚠️ Warning: Could not import run_portfolio_builder from agent1_portfolio_builder: {e}")
    def run_portfolio_builder():
        print("❌ Portfolio Builder is currently unavailable.")

# 2. Agent 2: LeetCode Solver (with fallback to multiple naming patterns)
run_leetcode_solver = None
try:
    from agent2_leetcode_solver import solve_leetcode_problem as run_leetcode_solver
except ImportError:
    try:
        from agent2_leetcode_solver import solve_single_problem as run_leetcode_solver
    except ImportError:
        try:
            from agent2_leetcode_solver import run_leetcode_solver as run_leetcode_solver
        except ImportError:
            try:
                from agent2_leetcode_solver import main as run_leetcode_solver
            except ImportError as e:
                print(f"⚠️ Warning: Could not import LeetCode solver from agent2_leetcode_solver: {e}")

if run_leetcode_solver is None:
    def run_leetcode_solver(*args, **kwargs):
        print("❌ LeetCode Solver is currently unavailable.")

# 3. Agent 3: Freelance Finder
try:
    from agent3_freelance_finder import run_freelance_finder
except ImportError:
    try:
        from agent3_freelance_finder import main as run_freelance_finder
    except ImportError:
        def run_freelance_finder():
            print("❌ Freelance Finder is currently unavailable.")

# 4. Agent 4: Job Finder
try:
    from agent4_job_finder import run_job_finder
except ImportError:
    try:
        from agent4_job_finder import main as run_job_finder
    except ImportError:
        def run_job_finder():
            print("❌ Job Finder is currently unavailable.")


def main():
    # Determine which agent to run from command-line args or environment variables
    agent_to_run = os.environ.get("AGENT_TO_RUN", "").lower()
    leetcode_batch = os.environ.get("LEETCODE_BATCH", "4")

    print(f"📋 Configuration - Agent to Run: '{agent_to_run}' | LeetCode Batch size: {leetcode_batch}")

    if not agent_to_run:
        # Fallback parsing command line arguments if workflow dispatch is manual
        import argparse
        parser = argparse.ArgumentParser(description="AI Agent Runner Trigger")
        parser.add_argument("--agent", type=str, default="all", help="Agent to run")
        parser.add_argument("--leetcode-batch", type=str, default="4", help="Leetcode batch count")
        args = parser.parse_args()
        agent_to_run = args.agent.lower()
        leetcode_batch = args.leetcode_batch

    try:
        if agent_to_run in ["portfolio", "all"]:
            print("\n🏗️ Running Portfolio Builder...")
            run_portfolio_builder()

        if agent_to_run in ["leetcode", "all"]:
            print("\n🧠 Running LeetCode Solver...")
            try:
                # Try running with batch count argument
                run_leetcode_solver(batch_count=int(leetcode_batch))
            except TypeError:
                # Fallback to zero-argument call if agent2 has a different signature
                run_leetcode_solver()

        if agent_to_run in ["freelance", "all"]:
            print("\n💼 Running Freelance Finder...")
            run_freelance_finder()

        if agent_to_run in ["jobs", "all"]:
            print("\n🔍 Running Job Finder...")
            run_job_finder()

        print("\n✨ All triggered tasks executed successfully.")

    except Exception as e:
        print(f"\n❌ Error encountered during execution flow:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()