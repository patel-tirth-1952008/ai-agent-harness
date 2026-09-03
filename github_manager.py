import os
import time
from github import Github, GithubException
from config import GITHUB_TOKEN, GITHUB_USERNAME


class GitHubManager:
    def __init__(self):
        token = (
            GITHUB_TOKEN
            or os.getenv("GH_TOKEN")
            or os.getenv("GITHUB_TOKEN")
            or ""
        ).strip()

        if not token:
            raise ValueError("Missing GitHub token. Set GH_TOKEN secret.")

        self.gh = Github(token)
        self.user = self.gh.get_user()
        self.username = (
            GITHUB_USERNAME
            or os.getenv("GH_USERNAME")
            or os.getenv("GITHUB_USERNAME")
            or self.user.login
        )

    def create_or_get_repo(self, repo_name: str, description: str = ""):
        # 1) Try existing repo first
        try:
            repo = self.user.get_repo(repo_name)
            print(f"📁 Using existing repo: {repo.html_url}")
            return repo
        except GithubException:
            pass

        # 2) Try create only if missing
        try:
            repo = self.user.create_repo(
                name=repo_name,
                description=description or "Created by AI Agent Harness",
                auto_init=True,
                private=False,
            )
            print(f"✅ Created repo: {repo.html_url}")
            time.sleep(2)
            return repo
        except GithubException as e:
            raise Exception(
                "GitHub token cannot create repositories (403). "
                "Use a CLASSIC PAT with 'repo' scope, and/or pre-create the repo. "
                f"Original error: {e}"
            )

    def push_files(self, repo_name: str, files: dict, description: str = "") -> str:
        repo = self.create_or_get_repo(repo_name, description)
        ok, fail = 0, 0

        for path, content in files.items():
            try:
                content = content if isinstance(content, str) else str(content)
                try:
                    existing = repo.get_contents(path)
                    repo.update_file(path, f"Update {path}", content, existing.sha)
                    print(f"  📝 Updated: {path}")
                except GithubException:
                    repo.create_file(path, f"Add {path}", content)
                    print(f"  ✅ Created: {path}")
                ok += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"  ❌ Failed {path}: {e}")
                fail += 1

        url = f"https://github.com/{self.username}/{repo_name}"
        print(f"📊 Push done. success={ok}, failed={fail}")
        print(f"🔗 {url}")
        return url

    def push_single_file(self, repo_name: str, file_path: str, content: str, description: str = "") -> str:
        return self.push_files(repo_name, {file_path: content}, description)


github_mgr = GitHubManager()