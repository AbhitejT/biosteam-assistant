"""Configuration and environment loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_LOG_DIR = PROJECT_ROOT / "run_logs"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CLAUDE_MODEL = os.environ.get("BIOSTEAM_AI_MODEL", "claude-sonnet-4-5")

# Conversion: ethanol density 0.789 kg/L * 3.78541 L/gal = 2.98668 kg/gal.
KG_ETHANOL_PER_GAL = 2.98668849


def get_api_key() -> str | None:
    """Return the Anthropic API key, or None if not configured."""
    return os.environ.get("ANTHROPIC_API_KEY")
