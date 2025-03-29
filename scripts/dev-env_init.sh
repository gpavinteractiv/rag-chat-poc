#!/bin/bash

# This script initializes the local development environment by:
# 1. Checking prerequisites (python3, podman).
# 2. Creating Python virtual environments if they don't exist.
# 3. Installing initial dependencies into the venvs from requirements.txt.
# 4. Checking for essential config files (.env).

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
# Temporarily disable during 'source activate' using set +u / set -u
# set -u
# Prevent errors in pipelines from being masked
set -o pipefail

# --- Configuration ---
# Determine project root directory (assuming script is in PROJECT_ROOT/scripts)
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PROJECT_ROOT=$( cd -- "$(dirname "$SCRIPT_DIR")" &> /dev/null && pwd )

readonly BACKEND_DIR="$PROJECT_ROOT/backend"
readonly FRONTEND_DIR="$PROJECT_ROOT/frontend"
readonly BACKEND_VENV_DIR="$BACKEND_DIR/venv"
readonly FRONTEND_VENV_DIR="$FRONTEND_DIR/venv_streamlit"
readonly BACKEND_REQ_FILE="$BACKEND_DIR/requirements.txt"
readonly FRONTEND_REQ_FILE="$FRONTEND_DIR/requirements.txt"
readonly ENV_FILE="$BACKEND_DIR/.env"
readonly TEMPLATE_ENV_FILE="$BACKEND_DIR/template.env"


# --- Helper Functions ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"
}

log_success() {
     echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS - $1"
}

log_warn() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARN - $1"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
}

# --- Main Initialization Logic ---
log "Starting Local Development Environment Initialization..."

# 1. Check Prerequisites
log "Checking for python3..."
PYTHON3=$(command -v python3)
if [ -z "$PYTHON3" ]; then
    log_error "python3 command not found in PATH. Please install Python 3."
    exit 1
fi
log "Found python3: $PYTHON3"

log "Checking for podman..."
PODMAN=$(command -v podman)
if [ -z "$PODMAN" ]; then
    log_error "podman command not found in PATH. Please install Podman."
    exit 1
fi
log "Found podman: $PODMAN"

# 2. Check Essential Files
log "Checking for requirements files..."
if [ ! -f "$BACKEND_REQ_FILE" ]; then log_error "Backend requirements file not found: $BACKEND_REQ_FILE"; exit 1; fi
if [ ! -f "$FRONTEND_REQ_FILE" ]; then log_error "Frontend requirements file not found: $FRONTEND_REQ_FILE"; exit 1; fi
log "Requirements files found."

log "Checking for backend .env file..."
if [ ! -f "$ENV_FILE" ]; then
    log_warn "Backend .env file not found at: $ENV_FILE"
    if [ -f "$TEMPLATE_ENV_FILE" ]; then
        log "Copying template environment file to $ENV_FILE..."
        cp "$TEMPLATE_ENV_FILE" "$ENV_FILE"
        log_success "Copied $TEMPLATE_ENV_FILE to $ENV_FILE."
        log_warn "IMPORTANT: Please edit $ENV_FILE and replace the placeholder GOOGLE_API_KEY with your actual key."
    else
        log_error "Template environment file $TEMPLATE_ENV_FILE not found. Cannot create $ENV_FILE."
        log_error "Please create $ENV_FILE manually with your GOOGLE_API_KEY."
        exit 1 # Make it fatal if template is missing and .env doesn't exist
    fi
else
    log ".env file found."
fi

# 3. Create Backend Virtual Environment and Install Dependencies
log "Checking backend virtual environment: $BACKEND_VENV_DIR"
if [ ! -d "$BACKEND_VENV_DIR" ]; then
    log "Backend venv not found. Creating..."
    if "$PYTHON3" -m venv "$BACKEND_VENV_DIR"; then
        log "Backend venv created successfully."
        log "Installing initial backend dependencies..."
        # Use subshell for activation and installation
        (
            set +u # Allow unset variables during source
            source "$BACKEND_VENV_DIR/bin/activate"
            set -u

            pip install --upgrade pip
            pip install -r "$BACKEND_REQ_FILE"
            # Optionally freeze right after initial install? Or let rebuild handle freeze. Let's skip freeze here.
        ) || { log_error "Failed during initial backend dependency installation."; exit 1; }
        log_success "Initial backend dependencies installed."
    else
        log_error "Failed to create backend virtual environment."
        exit 1
    fi
else
    log "Backend venv already exists. Skipping creation."
    log "Run './scripts/rebuild_poc.sh -u' to update dependencies if needed."
fi

# 4. Create Frontend Virtual Environment and Install Dependencies
log "Checking frontend virtual environment: $FRONTEND_VENV_DIR"
if [ ! -d "$FRONTEND_VENV_DIR" ]; then
    log "Frontend venv not found. Creating..."
    if "$PYTHON3" -m venv "$FRONTEND_VENV_DIR"; then
        log "Frontend venv created successfully."
        log "Installing initial frontend dependencies..."
        # Use subshell for activation and installation
        (
            set +u # Allow unset variables during source
            source "$FRONTEND_VENV_DIR/bin/activate"
            set -u

            pip install --upgrade pip
            pip install -r "$FRONTEND_REQ_FILE"
        ) || { log_error "Failed during initial frontend dependency installation."; exit 1; }
        log_success "Initial frontend dependencies installed."
    else
        log_error "Failed to create frontend virtual environment."
        exit 1
    fi
else
    log "Frontend venv already exists. Skipping creation."
    log "Run './scripts/rebuild_poc.sh -u' to update dependencies if needed."
fi

log_success "--------------------------------------------------------"
log_success "Local Development Environment Initialization Complete!"
log_success "Virtual environments are set up."
if [ -f "$ENV_FILE" ]; then
    # Check if placeholders are still there
    google_key_configured=true
    openrouter_key_configured=true
    if grep -q "YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE" "$ENV_FILE"; then
        log_warn "Reminder: Ensure '$ENV_FILE' contains your actual GOOGLE_API_KEY (placeholder detected)."
        google_key_configured=false
    fi
     if grep -q "YOUR_OPENROUTER_API_KEY_HERE" "$ENV_FILE"; then
        log_warn "Reminder: Ensure '$ENV_FILE' contains your actual OPENROUTER_API_KEY (placeholder detected)."
        openrouter_key_configured=false
    fi

    if $google_key_configured && $openrouter_key_configured; then
         log_success "'$ENV_FILE' seems to be configured with API keys."
    elif $google_key_configured; then
        log_success "GOOGLE_API_KEY seems configured in '$ENV_FILE'."
    elif $openrouter_key_configured; then
         log_success "OPENROUTER_API_KEY seems configured in '$ENV_FILE'."
    fi
else
     log_error "'$ENV_FILE' is missing. Backend may not function correctly." # Should not happen due to earlier check, but good to have.
fi
log_success "You can now build the container images using './scripts/rebuild_poc.sh'."
log_success "--------------------------------------------------------"

exit 0
