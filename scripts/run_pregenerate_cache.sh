#!/bin/sh

# Wrapper script to run pregenerate_doc_cache.py using the backend venv

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the directory where this script is located.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Construct the absolute path to the project root (one level up from scripts)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Construct the absolute path to the virtual environment's Python interpreter
VENV_PYTHON="$PROJECT_ROOT/backend/venv/bin/python"

# Construct the absolute path to the target Python script
TARGET_SCRIPT="$SCRIPT_DIR/pregenerate_doc_cache.py"

# Check if the Python interpreter exists
if [ ! -f "$VENV_PYTHON" ]; then
  echo "Error: Python interpreter not found at $VENV_PYTHON" >&2
  echo "Please ensure the backend virtual environment exists (run ./scripts/dev-env_init.sh)." >&2
  exit 1
fi

# Check if the target script exists
if [ ! -f "$TARGET_SCRIPT" ]; then
  echo "Error: Target script not found at $TARGET_SCRIPT" >&2
  exit 1
fi

# Execute the target Python script using the virtual environment's Python,
# passing all arguments received by this wrapper script ("$@").
echo "Executing $TARGET_SCRIPT using $VENV_PYTHON..."
"$VENV_PYTHON" "$TARGET_SCRIPT" "$@"

echo "Cache pre-generation wrapper script finished."
