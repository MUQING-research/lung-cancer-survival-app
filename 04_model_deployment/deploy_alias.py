# Module guide:
# - Role: Validate and publish the survival Shiny application.
# - Workflow: Check required runtime files before delegating to the deployment command.
# - Design note: Fail early when package contents or runtime compatibility are invalid.
"""Backward-compatible entry point for the shinyapps.io deployment helper."""

from deploy import main


if __name__ == "__main__":
    raise SystemExit(main())
