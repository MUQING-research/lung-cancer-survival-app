"""Compatibility imports from the canonical hosted survival model code."""

# Module guide:
# - Role: Preserve legacy imports for the canonical survival model code.
# - Workflow: Re-export or delegate to the active survival implementation.
# - Design note: Do not add new modelling logic to this archived compatibility layer.
import importlib.util
from pathlib import Path


_CANONICAL = Path(__file__).resolve().parents[2] / "04_model_deployment" / "survival_core.py"
_SPEC = importlib.util.spec_from_file_location("canonical_survival_core", _CANONICAL)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load canonical survival core: {_CANONICAL}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
globals().update({name: value for name, value in vars(_MODULE).items() if not name.startswith("__")})
