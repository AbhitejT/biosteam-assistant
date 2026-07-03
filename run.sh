#!/usr/bin/env bash
# Quickstart launcher for the BioSTEAM Assistant web UI.
#
# Usage:  ./run.sh
#
# This checks that a .env exists (with your ANTHROPIC_API_KEY) and then starts
# the Streamlit app. It does NOT install dependencies; see the README for the
# one-time conda setup.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and add your ANTHROPIC_API_KEY:"
  echo "  cp .env.example .env"
  exit 1
fi

if ! grep -q "ANTHROPIC_API_KEY=..*" .env; then
  echo "Warning: ANTHROPIC_API_KEY does not appear to be set in .env."
fi

echo "Starting BioSTEAM Assistant at http://localhost:8501 ..."
exec streamlit run app.py
