import time
import json
import os
from config import MAX_API_CALLS_PER_DAY, MAX_API_CALLS_PER_RUN

LIMITER_STATE_FILE = "data/limiter_state.json"


class APILimiter:
    def __init__(self):
        self.max_per_day = MAX_API_CALLS_PER_DAY
        self.max_per_run = MAX_API_CALLS_PER_RUN
        self.calls_this_run = 0
        self.load_state()

    def load_state(self):
        if os.path.exists(LIMITER_STATE_FILE):
            try:
                with open(LIMITER_STATE_FILE, "r") as f:
                    state = json.load(f)
                self.calls_today = state.get("calls_today", 0)
                self.day_start = state.get("day_start", time.time())
            except Exception:
                self.calls_today = 0
                self.day_start = time.time()
        else:
            self.calls_today = 0
            self.day_start = time.time()

    def save_state(self):
        os.makedirs("data", exist_ok=True)
        with open(LIMITER_STATE_FILE, "w") as f:
            json.dump({
                "calls_today": self.calls_today,
                "day_start": self.day_start
            }, f)

    def check(self):
        # Reset daily counter if 24 hours have passed
        if time.time() - self.day_start > 86400:
            self.calls_today = 0
            self.day_start = time.time()
            print("🔄 Daily API counter reset.")

        if self.calls_today >= self.max_per_day:
            self.save_state()
            raise Exception(
                f"🛑 DAILY LIMIT REACHED ({self.calls_today}/{self.max_per_day}). "
                f"Agent stopped. Will reset in {int(86400 - (time.time() - self.day_start))} seconds."
            )

        if self.calls_this_run >= self.max_per_run:
            self.save_state()
            raise Exception(
                f"🛑 PER-RUN LIMIT REACHED ({self.calls_this_run}/{self.max_per_run}). "
                f"This agent run is stopped."
            )

        self.calls_today += 1
        self.calls_this_run += 1
        self.save_state()

        remaining_today = self.max_per_day - self.calls_today
        remaining_run = self.max_per_run - self.calls_this_run
        print(f"📊 API Call #{self.calls_this_run} this run | "
              f"#{self.calls_today} today | "
              f"Remaining: {remaining_run} (run) / {remaining_today} (day)")

    def reset_run(self):
        self.calls_this_run = 0

    def get_status(self):
        return {
            "calls_today": self.calls_today,
            "max_per_day": self.max_per_day,
            "calls_this_run": self.calls_this_run,
            "max_per_run": self.max_per_run
        }


# Global limiter instance
limiter = APILimiter()