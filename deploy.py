"""
deploy.py -- Deploy survival app to shinyapps.io
Usage: python deploy.py
"""
import os
import subprocess
import sys
from pathlib import Path

ACCOUNT = "medictio"
TITLE   = "nsclc-survival"
DIR     = str(Path(__file__).parent)
# Files/folders that are only needed for local development, HuggingFace, or
# documentation — never required by the deployed Shiny runtime.
EXCLUDE = [
    "Dockerfile", "upload.py", "deploy.py", "README.md",
    "generate_readme_figures.py",
    "__pycache__", ".cache", ".git", ".github", "output", ".playwright-cli",
    "rsconnect-python", "assets",
]

cmd = (
    ["rsconnect", "deploy", "shiny", DIR, "--name", ACCOUNT, "--title", TITLE]
    + [arg for f in EXCLUDE for arg in ("--exclude", f)]
)

for variable in ("SUPABASE_URL", "SUPABASE_KEY", "IPINFO_TOKEN"):
    if os.environ.get(variable):
        cmd.extend(("--environment", variable))

print(f"Deploying '{TITLE}' to shinyapps.io ({ACCOUNT})...")
result = subprocess.run(cmd)

if result.returncode == 0:
    print(f"\nDone -- https://{ACCOUNT}.shinyapps.io/{TITLE}/")
else:
    print("\nDeployment failed.", file=sys.stderr)
    sys.exit(result.returncode)
