# Module guide:
# - Role: Provide the lightweight survival application entry point.
# - Workflow: Expose the configured app object from the neighboring application module.
# - Design note: Keep this wrapper side-effect free.
from survival_app import app as app  # noqa: PLC0414
