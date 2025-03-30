#!/bin/bash

# Script to update token counts in filelist.csv for all projects.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PROJECT_ROOT=$( cd -- "$(dirname "$SCRIPT_DIR")" &> /dev/null && pwd )
readonly PROJECTS_BASE_DIR="$PROJECT_ROOT/projects"
readonly BACKEND_VENV_ACTIVATE="$PROJECT_ROOT/backend/venv/bin/activate"
readonly TOKEN_CALC_SCRIPT="$PROJECT_ROOT/scripts/calculate_token_counts.py"

# --- Check Dependencies ---
if [ ! -f "$BACKEND_VENV_ACTIVATE" ]; then
    echo "ERROR: Backend virtual environment activation script not found at $BACKEND_VENV_ACTIVATE" >&2
    echo "Please run './scripts/dev-env_init.sh' first." >&2
    exit 1
fi
if [ ! -f "$TOKEN_CALC_SCRIPT" ]; then
    echo "ERROR: Token calculation script not found at $TOKEN_CALC_SCRIPT" >&2
    exit 1
fi
if [ ! -d "$PROJECTS_BASE_DIR" ]; then
     echo "ERROR: Projects base directory not found at $PROJECTS_BASE_DIR" >&2
     exit 1
fi

echo "Starting token count update for all projects in $PROJECTS_BASE_DIR..."
echo "(This may take a while depending on the number and size of documents)"

# --- Activate Venv ---
# Activate once before the loop
echo "Activating backend virtual environment..."
source "$BACKEND_VENV_ACTIVATE"

# --- Loop Through Projects ---
shopt -s nullglob # Prevent loop from running if no matches
found_projects=0
for project_path in "$PROJECTS_BASE_DIR"/*/; do
    project_name=$(basename "$project_path")

    # Skip directories starting with underscore
    if [[ "$project_name" == _* ]]; then
        echo "Skipping template/hidden directory: $project_name"
        continue
    fi

    # Check if it's actually a directory
    if [ -d "$project_path" ]; then
        found_projects=$((found_projects + 1))
        echo "--------------------------------------------------"
        echo "Processing project: $project_name"
        echo "--------------------------------------------------"

        # Run the python script for this project directory
        # Ensure python uses the activated venv
        python "$TOKEN_CALC_SCRIPT" "$project_path"

        echo "Finished processing project: $project_name"
    else
         echo "Skipping non-directory item: $project_name"
    fi
done
shopt -u nullglob # Reset nullglob option

# --- Deactivate Venv ---
echo "--------------------------------------------------"
echo "Deactivating virtual environment..."
deactivate

echo "--------------------------------------------------"
if [ "$found_projects" -eq 0 ]; then
    echo "No project directories found to process in $PROJECTS_BASE_DIR (excluding those starting with '_')."
else
    echo "Token count update process finished for $found_projects project(s)."
fi
exit 0
