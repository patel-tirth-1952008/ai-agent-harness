"""
Agent 2: Autonomous LeetCode Problem Solver & Streak Keeper
Multi-Problem Batching & Rate Limit Shield
"""

import os
import sys
import json
import time
import random
import subprocess
from datetime import datetime
from typing import Dict, Any, List

# Suppress litellm cache breakpoint warnings that cause Groq 400 errors
os.environ["LITELLM_DROP_PARAMS"] = "True"

from crewai import Agent, Task, Crew, Process, LLM

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
LEETCODE_BATCH_SIZE = int(os.getenv("LEETCODE_BATCH_SIZE", "1"))
DELAY_BETWEEN_PROBLEMS = int(os.getenv("LEETCODE_DELAY_SECONDS", "90"))
QUEUE_FILE = "leetcode_queue.json"
SOLUTIONS_DIR = "leetcode_solutions"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("PAT_TOKEN")

# Resilient model configuration with strict token bounds
llm = LLM(
    model="groq/openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=750,
)

# ==========================================
# PROBLEM QUEUE UTILITIES
# ==========================================
DEFAULT_PROBLEMS = [
    {"id": 1, "title": "Two Sum", "difficulty": "Easy", "topic": "Arrays & Hash Table"},
    {"id": 20, "title": "Valid Parentheses", "difficulty": "Easy", "topic": "Stack"},
    {"id": 21, "title": "Merge Two Sorted Lists", "difficulty": "Easy", "topic": "Linked List"},
    {"id": 53, "title": "Maximum Subarray", "difficulty": "Medium", "topic": "Dynamic Programming"},
    {"id": 70, "title": "Climbing Stairs", "difficulty": "Easy", "topic": "Dynamic Programming"},
    {"id": 121, "title": "Best Time to Buy and Sell Stock", "difficulty": "Easy", "topic": "Sliding Window"},
    {"id": 125, "title": "Valid Palindrome", "difficulty": "Easy", "topic": "Two Pointers"},
    {"id": 141, "title": "Linked List Cycle", "difficulty": "Easy", "topic": "Two Pointers"},
    {"id": 206, "title": "Reverse Linked List", "difficulty": "Easy", "topic": "Linked List"},
    {"id": 217, "title": "Contains Duplicate", "difficulty": "Easy", "topic": "Arrays & Hash Table"},
    {"id": 226, "title": "Invert Binary Tree", "difficulty": "Easy", "topic": "Trees"},
    {"id": 242, "title": "Valid Anagram", "difficulty": "Easy", "topic": "Hash Table"},
    {"id": 3, "title": "Longest Substring Without Repeating Characters", "difficulty": "Medium", "topic": "Sliding Window"},
    {"id": 11, "title": "Container With Most Water", "difficulty": "Medium", "topic": "Two Pointers"},
    {"id": 15, "title": "3Sum", "difficulty": "Medium", "topic": "Two Pointers"},
    {"id": 33, "title": "Search in Rotated Sorted Array", "difficulty": "Medium", "topic": "Binary Search"},
    {"id": 98, "title": "Validate Binary Search Tree", "difficulty": "Medium", "topic": "Trees"},
    {"id": 102, "title": "Binary Tree Level Order Traversal", "difficulty": "Medium", "topic": "BFS"},
    {"id": 133, "title": "Clone Graph", "difficulty": "Medium", "topic": "Graphs"},
    {"id": 198, "title": "House Robber", "difficulty": "Medium", "topic": "Dynamic Programming"},
    {"id": 200, "title": "Number of Islands", "difficulty": "Medium", "topic": "Graphs / DFS"},
    {"id": 238, "title": "Product of Array Except Self", "difficulty": "Medium", "topic": "Arrays"},
    {"id": 300, "title": "Longest Increasing Subsequence", "difficulty": "Medium", "topic": "Dynamic Programming"},
    {"id": 322, "title": "Coin Change", "difficulty": "Medium", "topic": "Dynamic Programming"},
    {"id": 417, "title": "Pacific Atlantic Water Flow", "difficulty": "Medium", "topic": "Graphs"}
]

def load_or_init_queue() -> List[Dict[str, Any]]:
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            print(f"⚠️ Warning: Could not read {QUEUE_FILE} ({e}). Resetting queue.")
    
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_PROBLEMS, f, indent=2)
    return DEFAULT_PROBLEMS.copy()

def pop_next_problem() -> Dict[str, Any]:
    queue = load_or_init_queue()
    if not queue:
        queue = DEFAULT_PROBLEMS.copy()
    
    problem = queue.pop(0)
    # Re-append solved problem to the back to maintain continuous cycle
    queue.append(problem)
    
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)
    return problem

# ==========================================
# AGENTS & TASK CREATION
# ==========================================
def solve_single_problem(problem: Dict[str, Any], max_retries: int = 3) -> bool:
    print(f"\n==================================================")
    print(f"🧩 Solving Problem #{problem['id']}: {problem['title']} [{problem['difficulty']}]")
    print(f"Topic: {problem['topic']}")
    print(f"==================================================")

    algo_specialist = Agent(
        role="Senior Algorithm & Data Structures Specialist",
        goal=f"Provide optimal Python solution, unit tests, and $O(N)$ complexity analysis for {problem['title']}.",
        backstory=(
            "You are an expert competitive programmer and algorithm engineer. "
            "You write clean, strictly optimal Python 3 code with complete complexity explanations."
        ),
        llm=llm,
        verbose=False,
        memory=False
    )

    task = Task(
        description=(
            f"Solve LeetCode Problem #{problem['id']}: '{problem['title']}' ({problem['difficulty']}).\n"
            f"Topic: {problem['topic']}.\n\n"
            "Produce output formatted strictly as:\n"
            "```python\n"
            "# Production-grade Python solution with class Solution\n"
            "```\n"
            "### Complexity Analysis:\n"
            "- **Time Complexity:** O(...)\n"
            "- **Space Complexity:** O(...)\n\n"
            "### Explanation:\n"
            "Concise breakdown of optimal strategy.\n\n"
            "### Test Cases:\n"
            "Example assert statements validating edge cases."
        ),
        expected_output="Python solution with complexity analysis, explanation, and test cases.",
        agent=algo_specialist
    )

    crew = Crew(
        agents=[algo_specialist],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )

    for attempt in range(1, max_retries + 1):
        try:
            result = crew.kickoff()
            solution_text = str(result)

            # Persist solution
            os.makedirs(SOLUTIONS_DIR, exist_ok=True)
            slug = problem['title'].lower().replace(" ", "_").replace("-", "_")
            file_name = f"{problem['id']:04d}_{slug}.md"
            file_path = os.path.join(SOLUTIONS_DIR, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# LeetCode #{problem['id']}: {problem['title']}\n")
                f.write(f"**Difficulty:** {problem['difficulty']} | **Topic:** {problem['topic']}\n\n")
                f.write(f"**Solved Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
                f.write(solution_text)

            print(f"✅ Solution saved: {file_path}")
            return True

        except Exception as e:
            err_msg = str(e)
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {err_msg}")
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                wait_time = 60 * attempt
                print(f"⏳ Rate limited. Backing off for {wait_time}s...")
                time.sleep(wait_time)
            else:
                time.sleep(15)

    print(f"❌ Failed to solve #{problem['id']} after {max_retries} attempts.")
    return False

# ==========================================
# GIT COMMIT & PUSH
# ==========================================
def commit_and_push_solutions(solved_count: int):
    if solved_count <= 0:
        print("ℹ️ No new solutions to commit.")
        return

    try:
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", SOLUTIONS_DIR, QUEUE_FILE], check=True)

        commit_msg = f"feat(leetcode): auto-solve {solved_count} problem(s) [skip ci]"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # Pull rebase before push to avoid remote conflict
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"🚀 Successfully pushed {solved_count} solved problem(s) to GitHub!")
    except Exception as e:
        print(f"⚠️ Git push failed or nothing to commit: {e}")

# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================
def main():
    print(f"🎯 Starting LeetCode Batch Runner: Target = {LEETCODE_BATCH_SIZE} problem(s)")
    solved_count = 0

    for i in range(1, LEETCODE_BATCH_SIZE + 1):
        print(f"\n--- Processing Batch Item {i}/{LEETCODE_BATCH_SIZE} ---")
        problem = pop_next_problem()
        
        success = solve_single_problem(problem)
        if success:
            solved_count += 1

        # Pause between problems to safeguard Groq OTPM/TPM limits
        if i < LEETCODE_BATCH_SIZE:
            print(f"🛡️ Rate-limit shield: Sleeping {DELAY_BETWEEN_PROBLEMS}s before next problem...")
            time.sleep(DELAY_BETWEEN_PROBLEMS)

    commit_and_push_solutions(solved_count)
    print(f"\n🎉 Batch run complete: {solved_count}/{LEETCODE_BATCH_SIZE} problems solved successfully.")

if __name__ == "__main__":
    main()