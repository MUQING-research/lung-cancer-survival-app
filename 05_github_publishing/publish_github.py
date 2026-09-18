# Module guide:
# - Role: Publish selected survival project files safely.
# - Workflow: Validate repository scope and push only explicitly allowed files.
# - Design note: Keep publishing operations auditable and narrowly scoped.
"""Safely commit and push selected project files to GitHub.

Usage:
    python 05_github_publishing/publish_github.py
    python 05_github_publishing/publish_github.py README.md -m "Update project README"
    python 05_github_publishing/publish_github.py README.md 05_github_publishing/publish_github.py --base origin/main
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_FILES = ("README.md",)
DEFAULT_MESSAGE = "Update project README"
PUSH_TIMEOUT_SECONDS = 60


# Class guide: PublishError is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class PublishError(RuntimeError):
    """Raised when a safe publish precondition or Git command fails."""


# Function guide: run_git is responsible for run git.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def run_git(
    repo: Path,
    *args: str,
    check: bool = True,
    capture_output: bool = True,
    timeout: int | None = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one Git command from the repository root."""
    command = ["git", *args]
    try:
        result = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        raise PublishError("Git is not installed or is not available on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        rendered = " ".join(command)
        raise PublishError(f"Command timed out: {rendered}") from exc

    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "Unknown Git error").strip()
        rendered = " ".join(command)
        raise PublishError(f"Command failed: {rendered}\n{detail}")
    return result


# Function guide: output_lines is responsible for perform lines.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def output_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Return non-empty output lines from a completed command."""
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# Function guide: find_repository is responsible for find repository.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def find_repository(script_dir: Path) -> Path:
    """Resolve the Git repository that contains this script."""
    result = run_git(script_dir, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


# Function guide: normalize_files is responsible for normalize files.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def normalize_files(repo: Path, raw_files: Sequence[str]) -> list[str]:
    """Validate files and return unique repository-relative paths."""
    normalized: list[str] = []
    for raw_file in raw_files:
        candidate = Path(raw_file)
        if not candidate.is_absolute():
            candidate = repo / candidate
        candidate = candidate.resolve()

        try:
            relative = candidate.relative_to(repo)
        except ValueError as exc:
            raise PublishError(f"File is outside the repository: {candidate}") from exc

        if not candidate.is_file():
            raise PublishError(f"File does not exist: {candidate}")

        relative_text = relative.as_posix()
        if relative_text not in normalized:
            normalized.append(relative_text)

    return normalized


# Function guide: current_branch is responsible for perform branch.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def current_branch(repo: Path) -> str:
    """Return the checked-out branch and reject detached HEAD state."""
    result = run_git(repo, "branch", "--show-current")
    branch = result.stdout.strip()
    if not branch:
        raise PublishError("Detached HEAD is not supported. Check out a branch first.")
    return branch


# Function guide: tracking_ref is responsible for perform ref.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def tracking_ref(repo: Path) -> str | None:
    """Return the configured upstream ref, if the current branch has one."""
    result = run_git(
        repo,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# Function guide: validate_staged_scope is responsible for validate staged scope.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def validate_staged_scope(repo: Path, allowed_files: set[str]) -> None:
    """Reject staged files that were not explicitly selected."""
    staged = set(output_lines(run_git(repo, "diff", "--cached", "--name-only")))
    unexpected = sorted(staged - allowed_files)
    if unexpected:
        rendered = ", ".join(unexpected)
        raise PublishError(f"Unrelated files are already staged: {rendered}")


# Function guide: validate_ahead_scope is responsible for validate ahead scope.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def validate_ahead_scope(repo: Path, base_ref: str, allowed_files: set[str]) -> int:
    """Ensure local-only commits contain only explicitly selected files."""
    run_git(repo, "rev-parse", "--verify", base_ref)
    ahead_count = int(
        run_git(repo, "rev-list", "--count", f"{base_ref}..HEAD").stdout.strip()
    )
    ahead_files = set(
        output_lines(run_git(repo, "diff", "--name-only", f"{base_ref}..HEAD"))
    )
    unexpected = sorted(ahead_files - allowed_files)
    if unexpected:
        rendered = ", ".join(unexpected)
        raise PublishError(
            f"Local-only commits include files outside the publish scope: {rendered}"
        )
    return ahead_count


# Function guide: parse_args is responsible for parse args.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Commit and push only explicitly selected files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Repository-relative files to publish (default: README.md).",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=DEFAULT_MESSAGE,
        help=f"Commit message (default: {DEFAULT_MESSAGE!r}).",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote to push (default: origin).",
    )
    parser.add_argument(
        "--base",
        help="Base ref used to validate local-only commits when no upstream exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the planned publish without changing Git state.",
    )
    return parser.parse_args()


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main() -> int:
    """Validate, commit, push, and verify the selected files."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()

    if shutil.which("git") is None:
        raise PublishError("Git is not installed or is not available on PATH.")

    script_dir = Path(__file__).resolve().parent
    repo = find_repository(script_dir)
    selected_files = normalize_files(repo, args.files or DEFAULT_FILES)
    allowed_files = set(selected_files)
    branch = current_branch(repo)
    remote_url = run_git(repo, "remote", "get-url", args.remote).stdout.strip()
    upstream = tracking_ref(repo)
    base_ref = args.base or upstream

    validate_staged_scope(repo, allowed_files)
    existing_ahead = 0
    if base_ref:
        existing_ahead = validate_ahead_scope(repo, base_ref, allowed_files)

    print(f"Repository: {repo}")
    print(f"Remote:     {args.remote} ({remote_url})")
    print(f"Branch:     {branch}")
    print(f"Files:      {', '.join(selected_files)}")
    if base_ref:
        print(f"Base:       {base_ref} ({existing_ahead} local commit(s) ahead)")
    else:
        print("Base:       none (a new upstream will be created)")

    status = output_lines(run_git(repo, "status", "--short", "--", *selected_files))
    if args.dry_run:
        print("Selected changes:")
        if status:
            for line in status:
                print(f"  {line}")
        else:
            print("  No uncommitted changes in selected files.")
        print("Dry run complete. No Git state was changed.")
        return 0

    run_git(repo, "add", "--", *selected_files)
    validate_staged_scope(repo, allowed_files)

    staged = output_lines(run_git(repo, "diff", "--cached", "--name-only"))
    if staged:
        run_git(repo, "commit", "-m", args.message)
        print(f"Created commit: {args.message}")
    else:
        print("No new commit was needed for the selected files.")

    upstream = tracking_ref(repo)
    base_ref = args.base or upstream
    if base_ref:
        ahead_count = validate_ahead_scope(repo, base_ref, allowed_files)
        if ahead_count == 0:
            print("Nothing to push. The branch already matches its base ref.")
            return 0

    push_args = ["push"]
    if upstream is None:
        push_args.extend(["-u", args.remote, branch])
    else:
        push_args.extend([args.remote, branch])

    print(f"Pushing {branch} to {args.remote}...")
    push_env = os.environ.copy()
    push_env["GCM_INTERACTIVE"] = "Never"
    push_env["GIT_TERMINAL_PROMPT"] = "0"
    result = run_git(
        repo,
        *push_args,
        check=False,
        capture_output=False,
        timeout=PUSH_TIMEOUT_SECONDS,
        env=push_env,
    )
    if result.returncode != 0:
        raise PublishError(
            "Push failed. Check network access and authenticate with Git Credential "
            "Manager using: git credential-manager github login --browser"
        )

    upstream = tracking_ref(repo)
    if upstream is None:
        raise PublishError("Push completed, but no upstream branch was configured.")

    local_sha = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream_sha = run_git(repo, "rev-parse", upstream).stdout.strip()
    if local_sha != upstream_sha:
        raise PublishError("Push returned successfully, but upstream verification failed.")

    print(f"Published commit {local_sha[:12]} to {upstream}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublishError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
