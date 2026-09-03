import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

# Set env vars BEFORE importing crewai/langchain
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from langchain_groq import ChatGroq
from limiter import limiter
from notifier import send_agent_report
from github_manager import github_mgr
from config import GROQ_API_KEY, GROQ_MODEL, LEETCODE_QUEUE_FILE


def get_llm():
    return ChatGroq(
        temperature=0.1,
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY
    )


def load_queue():
    if os.path.exists(LEETCODE_QUEUE_FILE):
        with open(LEETCODE_QUEUE_FILE, "r") as f:
            return json.load(f)
    return {"problems": [], "solved": []}


def save_queue(data):
    os.makedirs("data", exist_ok=True)
    with open(LEETCODE_QUEUE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def sanitize_filename(text):
    """Remove all characters that break file paths."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def solve_leetcode_problem(problem=None, repo_name="leetcode-solutions"):
    print("\n" + "=" * 60)
    print("  AGENT 2: LEETCODE SOLVER")
    print("=" * 60)

    limiter.check()

    queue = load_queue()

    # Pick next unsolved problem
    if problem is None:
        unsolved = [
            p for p in queue["problems"]
            if p["title"] not in queue["solved"]
        ]
        if not unsolved:
            msg = "All problems in queue are solved!"
            print(msg)
            send_agent_report("LeetCode Solver", "success", msg)
            return None
        problem = unsolved[0]

    title = problem["title"]
    difficulty = problem["difficulty"]
    number = problem.get("number", 0)

    print(f"  Solving: #{number} {title} ({difficulty})")

    llm = get_llm()

    prompt = (
        "You are a competitive programming grandmaster.\n\n"
        f"Solve LeetCode Problem #{number}: {title} (Difficulty: {difficulty}).\n\n"
        "Respond in EXACTLY this format:\n\n"
        f"# {number}. {title}\n\n"
        "## Difficulty\n"
        f"{difficulty}\n\n"
        "## Problem Description\n"
        "[Clear description of the problem]\n\n"
        "## Approach\n"
        "- Algorithm used and why\n"
        "- Time Complexity: O(?)\n"
        "- Space Complexity: O(?)\n\n"
        "## Solution\n\n"
        "```python\n"
        "class Solution:\n"
        "    def solve(self, ...):\n"
        "        # Your optimal solution with comments\n"
        "        pass\n"
        "```\n\n"
        "## Edge Cases Handled\n"
        "- [List edge cases]\n\n"
        "RULES:\n"
        "1. Solution must be optimal time complexity.\n"
        "2. Code must be valid Python 3.\n"
        "3. Include detailed comments.\n"
        "4. Handle all edge cases.\n"
        "5. Use only Python standard library.\n"
    )

    try:
        limiter.check()
        response = llm.invoke(prompt)

        # Extract text content safely
        if hasattr(response, "content"):
            solution_text = str(response.content)
        else:
            solution_text = str(response)

        if len(solution_text) < 100:
            send_agent_report(
                "LeetCode Solver", "error",
                f"Solution for #{number} {title} was too short."
            )
            return None

        # Build safe file path
        folder = difficulty.lower()
        safe_name = sanitize_filename(title)
        file_path = f"{folder}/{number:04d}_{safe_name}.md"

        # Push to GitHub
        repo_url = github_mgr.push_single_file(
            repo_name=repo_name,
            file_path=file_path,
            content=solution_text,
            description="LeetCode Solutions Archive"
        )

        # Mark as solved
        if title not in queue["solved"]:
            queue["solved"].append(title)
            save_queue(queue)

        total = len(queue["problems"])
        solved = len(queue["solved"])

        send_agent_report(
            "LeetCode Solver", "success",
            f"Solved #{number} *{title}* ({difficulty})\n"
            f"Progress: {solved}/{total}\n"
            f"File: {file_path}"
        )

        return repo_url

    except Exception as e:
        err = str(e)
        if "LIMIT REACHED" in err:
            send_agent_report("LeetCode Solver", "warning", err)
        else:
            send_agent_report("LeetCode Solver", "error", err)
        print(f"  Error: {e}")
        return None


if __name__ == "__main__":
    solve_leetcode_problem()