#!/bin/sh

# Wrapper script to run process_project_docs.py using the backend venv
# This script combines and replaces functionality from:
# - run_calculate_tokens.sh
# - run_pregenerate_cache.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the directory where this script is located.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Construct the absolute path to the project root (two levels up from backend/scripts)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Define potential Python interpreter paths
CONTAINER_VENV_PYTHON="/opt/venv/bin/python"
HOST_VENV_PYTHON="$PROJECT_ROOT/backend/venv/bin/python"

# Determine the correct Python interpreter path
if [ -f "$CONTAINER_VENV_PYTHON" ]; then
  # Use container path if it exists
  VENV_PYTHON="$CONTAINER_VENV_PYTHON"
  echo "Running inside container, using interpreter: $VENV_PYTHON"
elif [ -f "$HOST_VENV_PYTHON" ]; then
  # Use host path if it exists
  VENV_PYTHON="$HOST_VENV_PYTHON"
  echo "Running on host, using interpreter: $VENV_PYTHON"
else
  # Neither found, report error referencing both possibilities
  echo "Error: Python interpreter not found." >&2
  echo "Looked for container path: $CONTAINER_VENV_PYTHON" >&2
  echo "Looked for host path: $HOST_VENV_PYTHON" >&2
  echo "Please ensure the backend virtual environment exists (run ./scripts/dev-env_init.sh on host, or check container build)." >&2
  exit 1
fi

# Construct the absolute path to the target Python script
TARGET_SCRIPT="$SCRIPT_DIR/process_project_docs.py"

# The check is now implicitly handled by the logic above that sets VENV_PYTHON or exits

# Check if the target script exists

# Check if the target script exists
if [ ! -f "$TARGET_SCRIPT" ]; then
  echo "Error: Target script not found at $TARGET_SCRIPT" >&2
  exit 1
fi

# Make the script executable if it isn't already
chmod +x "$TARGET_SCRIPT"

# Print help information if no arguments provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <project_directory> [options]"
  echo ""
  echo "Options:"
  echo "  --token-counts-only     Only update token counts in filelist.csv (no cache generation)"
  echo "  --cache-only            Only generate cache (no filelist.csv update)"
  echo "  --parallel              Use parallel processing for document parsing"
  echo "  --all-projects          Process all valid projects in the projects directory"
  echo "  --regenerate            Force regeneration of filelist.csv and cache even if they exist"
  echo ""
  echo "Example: $0 projects/my_project"
  echo "Example: $0 projects/my_project --token-counts-only"
  echo "Example: $0 --all-projects"
  echo "Example: $0 projects/my_project --regenerate  # To force complete regeneration"
  exit 0
fi

# Execute the target Python script using the virtual environment's Python,
# passing all arguments received by this wrapper script ("$@").

# Check if --all-projects is in the arguments
all_projects_flag=false
for arg in "$@"; do
  if [ "$arg" = "--all-projects" ]; then
    all_projects_flag=true
    break
  fi
done

if [ "$all_projects_flag" = true ]; then
  echo "Executing $TARGET_SCRIPT with --all-projects (adding dummy project dir)..."
  "$VENV_PYTHON" "$TARGET_SCRIPT" "." "$@" # Add "." before original args
else
  echo "Executing $TARGET_SCRIPT using $VENV_PYTHON..."
  "$VENV_PYTHON" "$TARGET_SCRIPT" "$@" # Normal execution
fi

echo "Document processing wrapper script finished."
