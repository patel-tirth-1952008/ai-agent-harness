import os
import time
from github import Github, GithubException
from config import GITHUB_TOKEN, GITHUB_USERNAME


class GitHubManager:
    def __init__(self):
        # Fallback to GH_TOKEN or GITHUB_TOKEN from env
        token = GITHUB_TOKEN or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub Token is missing! Please set GITHUB_TOKEN or GH_TOKEN.")
        self.gh = Github(token)
        self.user = self.gh.get_user()

    def create_or_get_repo(self, repo_name: str, description: str = "") -> object:
        """Create a new repo or return existing one."""
        try:
            repo = self.user.get_repo(repo_name)
            print(f"📁 Repository '{repo_name}' already exists. Updating...")
            return repo
        except GithubException:
            repo = self.user.create_repo(
                name=repo_name,
                description=description,
                auto_init=True,
                private=False
            )
            print(f"✅ Created new repository: {repo.html_url}")
            time.sleep(3)
            return repo

    def push_files(self, repo_name: str, files: dict, description: str = "") -> str:
        """Push multiple files to a repository."""
        repo = self.create_or_get_repo(repo_name, description)
        pushed_count = 0
        failed_count = 0

        for file_path, content in files.items():
            try:
                try:
                    existing = repo.get_contents(file_path)
                    repo.update_file(
                        path=existing.path,
                        message=f"Update {file_path}",
                        content=content,
                        sha=existing.sha
                    )
                    print(f"  📝 Updated: {file_path}")
                except GithubException:
                    repo.create_file(
                        path=file_path,
                        message=f"Add {file_path}",
                        content=content
                    )
                    print(f"  ✅ Created: {file_path}")
                pushed_count += 1
                time.sleep(1)
            except Exception as e:
                print(f"  ❌ Failed to push {file_path}: {e}")
                failed_count += 1

        username = GITHUB_USERNAME or os.getenv("GH_USERNAME") or os.getenv("GITHUB_USERNAME")
        repo_url = f"https://github.com/{username}/{repo_name}"
        print(f"\n📊 Push complete: {pushed_count} succeeded, {failed_count} failed")
        print(f"🔗 Repository: {repo_url}")
        return repo_url

    def push_single_file(self, repo_name: str, file_path: str, content: str, description: str = "") -> str:
        """Push a single file to a repository."""
        return self.push_files(repo_name, {file_path: content}, description)


# Global instance
github_mgr = GitHubManager()