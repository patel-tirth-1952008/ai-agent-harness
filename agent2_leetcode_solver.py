import os
import re
import json
import time
import litellm
from dotenv import load_dotenv

load_dotenv()

# Force LiteLLM configuration & automatic retries on Rate Limits
litellm.drop_params = True
litellm.num_retries = 5
litellm.request_timeout = 120
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

# Set env vars BEFORE importing crewai
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from crewai import LLM
from limiter import limiter
from notifier import send_agent_report
from github_manager import github_mgr
from config import GROQ_API_KEY, GROQ_MODEL, LEETCODE_QUEUE_FILE

DEFAULT_QUEUE = {
    "problems": [
        {"number": 1, "title": "Two Sum", "difficulty": "Easy"},
        {"number": 20, "title": "Valid Parentheses", "difficulty": "Easy"},
        {"number": 21, "title": "Merge Two Sorted Lists", "difficulty": "Easy"},
        {"number": 121, "title": "Best Time to Buy and Sell Stock", "difficulty": "Easy"},
        {"number": 125, "title": "Valid Palindrome", "difficulty": "Easy"},
        {"number": 226, "title": "Invert Binary Tree", "difficulty": "Easy"},
        {"number": 242, "title": "Valid Anagram", "difficulty": "Easy"},
        {"number": 704, "title": "Binary Search", "difficulty": "Easy"},
        {"number": 3, "title": "Longest Substring Without Repeating Characters", "difficulty": "Medium"},
        {"number": 11, "title": "Container With Most Water", "difficulty": "Medium"},
        {"number": 15, "title": "3Sum", "difficulty": "Medium"},
        {"number": 33, "title": "Search in Rotated Sorted Array", "difficulty": "Medium"},
        {"number": 49, "title": "Group Anagrams", "difficulty": "Medium"},
        {"number": 53, "title": "Maximum Subarray", "difficulty": "Medium"},
        {"number": 198, "title": "House Robber", "difficulty": "Medium"},
        {"number": 200, "title": "Number of Islands", "difficulty": "Medium"},
        {"number": 4, "title": "Median of Two Sorted Arrays", "difficulty": "Hard"},
        {"number": 23, "title": "Merge k Sorted Lists", "difficulty": "Hard"},
        {"number": 42, "title": "Trapping Rain Water", "difficulty": "Hard"},
        {"number": 295, "title": "Find Median from Data Stream", "difficulty": "Hard"}
    ],
    "solved": []
}


def get_llm():
    model_name = GROQ_MODEL if GROQ_MODEL.startswith("groq/") else f"groq/{GROQ_MODEL}"
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.1
    )


def load_queue():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LEETCODE_QUEUE_FILE):
        try:
            with open(LEETCODE_QUEUE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "problems" in data and "solved" in data:
                    return data
        except Exception as e:
            print(f"⚠️ Warning: Invalid {LEETCODE_QUEUE_FILE} ({e}). Resetting with clean queue.")
    
    save_queue(DEFAULT_QUEUE)
    return DEFAULT_QUEUE


def save_queue(data):
    os.makedirs("data", exist_ok=True)
    with open(LEETCODE_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sanitize_filename(text):
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
        
        solution_text = None
        for attempt in range(3):
            try:
                response = llm.call([{"role": "user", "content": prompt}])
                solution_text = str(response)
                break
            except Exception as call_err:
                if "429" in str(call_err) or "rate" in str(call_err).lower():
                    print(f"⏳ Rate limit hit on attempt {attempt+1}. Retrying in 12s...")
                    time.sleep(12)
                else:
                    raise call_err

        if not solution_text or len(solution_text) < 100:
            send_agent_report(
                "LeetCode Solver", "error",
                f"Solution for #{number} {title} was too short or empty."
            )
            return None

        folder = difficulty.lower()
        safe_name = sanitize_filename(title)
        file_path = f"{folder}/{number:04d}_{safe_name}.md"

        repo_url = github_mgr.push_single_file(
            repo_name=repo_name,
            file_path=file_path,
            content=solution_text,
            description="LeetCode Solutions Archive"
        )

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
        print(f"✅ Successfully solved #{number} {title} and pushed to {file_path}")

        return repo_url

    except Exception as e:
        err = str(e)
        if "LIMIT REACHED" in err:
            send_agent_report("LeetCode Solver", "warning", err)
        else:
            send_agent_report("LeetCode Solver", "error", err)
        print(f"  Error: {e}")
        return None


def solve_multiple(count: int = 3, repo_name: str = "leetcode-solutions"):
    print(f"\n🔄 Solving {count} LeetCode problems in this shift...")
    results = []
    for i in range(count):
        print(f"\n--- [Problem {i + 1}/{count}] ---")
        try:
            res = solve_leetcode_problem(repo_name=repo_name)
            if res:
                results.append(res)
            if i < count - 1:
                print("⏳ Sleeping 8s before next problem...")
                time.sleep(8)
        except Exception as e:
            print(f"⚠️ Batch stopped early due to error or limit: {e}")
            break
    return results


if __name__ == "__main__":
    solve_leetcode_problem()