#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in pipelines from being masked
set -o pipefail

# --- Configuration (UPDATED WITH RAG-* NAMES) ---
readonly POD_NAME="rag-poc-pod"
readonly BACKEND_CONTAINER_NAME="rag-backend-container"
readonly FRONTEND_CONTAINER_NAME="rag-frontend-container"
readonly BACKEND_IMAGE="localhost/rag-chat-backend:0.1"
readonly FRONTEND_IMAGE="localhost/rag-chat-frontend:0.1"

# Determine project root directory (assuming script is in PROJECT_ROOT/scripts)
# Assuming the root dir name also changed, e.g., /mnt/LAB/rag-chat-poc
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PROJECT_ROOT=$( cd -- "$(dirname "$SCRIPT_DIR")" &> /dev/null && pwd )

readonly BACKEND_DIR="$PROJECT_ROOT/backend"
readonly FRONTEND_DIR="$PROJECT_ROOT/frontend"
readonly BACKEND_VENV_DIR="$BACKEND_DIR/venv"
readonly FRONTEND_VENV_DIR="$FRONTEND_DIR/venv_streamlit"

# Flag for dependency update
UPDATE_DEPS=false

# Find podman executable
PODMAN=$(command -v podman)
if [ -z "$PODMAN" ]; then
    echo "Error: podman command not found in PATH." >&2
    exit 1
fi

# Find python3 executable
PYTHON3=$(command -v python3)
if [ -z "$PYTHON3" ]; then
    echo "Error: python3 command not found in PATH." >&2
    exit 1
fi

# --- Argument Parsing ---
for arg in "$@"
do
    case $arg in
        -u|--update-deps)
        UPDATE_DEPS=true
        shift # Remove --update-deps from processing
        ;;
        -h|--help)
        echo "Usage: $0 [-u|--update-deps]"
        echo "  -u, --update-deps: Force remove/recreate local venvs and install dependencies from requirements.txt before rebuilding."
        exit 0
        ;;
        *)
        # Unknown option
        ;;
    esac
done


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

# Function to REMOVE AND RECREATE venv and install dependencies
update_venv_deps() {
    local target_dir="$1"
    local venv_dir="$2"
    local req_file="$target_dir/requirements.txt"
    local component_name="$3" # e.g., "backend", "frontend"

    log "Recreating virtual environment and updating dependencies for $component_name in $target_dir"

     if [ ! -f "$req_file" ]; then
        log_error "Requirements file not found: $req_file. Cannot proceed with $component_name dependency update."
        exit 1
    fi

    # --- Remove existing venv ---
    if [ -d "$venv_dir" ]; then
        log "Removing existing virtual environment: $venv_dir"
        if ! rm -rf "$venv_dir"; then
            log_error "Failed to remove existing virtual environment at $venv_dir."
            exit 1
        fi
        log "Existing venv removed."
    else
        log "Virtual environment directory not found: $venv_dir. Will create new one."
    fi

    # --- Create new venv ---
    log "Creating new virtual environment: $venv_dir"
    if ! "$PYTHON3" -m venv "$venv_dir"; then
        log_error "Failed to create new virtual environment at $venv_dir."
        exit 1
    fi
    log "$component_name venv created successfully."

    # --- Install Dependencies into new venv ---
    log "Installing dependencies from $req_file into new $component_name venv..."
    # Use subshell to avoid polluting current environment and needing deactivate
    (
        set +u # Temporarily allow unset variables during source
        source "$venv_dir/bin/activate"
        set -u

        log "Upgrading pip in $component_name venv..."
        pip install --upgrade pip

        log "Installing packages from $req_file..."
        pip install -r "$req_file"

        log "Freezing installed dependencies back to $req_file..."
        pip freeze > "$req_file"

        log "$component_name dependency update complete."
    ) || { log_error "Failed during $component_name dependency installation process."; exit 1; }

}


# --- Main Rebuild Logic ---
log "Starting RAG PoC rebuild process..."
log "This script stops/removes the existing pod and containers, then rebuilds the container images."
log "It does NOT restart the application. Use manage_pod.sh or the systemd service for that."
if [ "$UPDATE_DEPS" = true ]; then log "Local dependency update requested (force recreating venvs)."; fi

# 1. Stop existing pod (if running) - Necessary for standalone execution.
# Note: If called by update_and_restart.sh, the systemd service stop already issued this command.
# Running it again here is generally harmless (podman handles 'already stopped' state).
log "Checking for existing pod '$POD_NAME'..." # Uses updated POD_NAME variable
if $PODMAN pod exists "$POD_NAME"; then
    log "Pod '$POD_NAME' exists. Attempting to stop..."
    if $PODMAN pod stop "$POD_NAME"; then log "Pod '$POD_NAME' stopped."; else log_warn "Failed to stop pod '$POD_NAME'."; fi
else
    log "Pod '$POD_NAME' does not exist. Skipping stop."
fi

# 2. Remove existing containers
log "Removing container '$BACKEND_CONTAINER_NAME' if it exists..." # Uses updated variable
$PODMAN rm "$BACKEND_CONTAINER_NAME" --force --ignore || log_warn "Container '$BACKEND_CONTAINER_NAME' not found or removal failed."
log "Removing container '$FRONTEND_CONTAINER_NAME' if it exists..." # Uses updated variable
$PODMAN rm "$FRONTEND_CONTAINER_NAME" --force --ignore || log_warn "Container '$FRONTEND_CONTAINER_NAME' not found or removal failed."

# 3. Remove existing pod
log "Removing pod '$POD_NAME' if it exists..." # Uses updated variable
if $PODMAN pod exists "$POD_NAME"; then
     if $PODMAN pod rm "$POD_NAME" --force; then log "Pod '$POD_NAME' removed."; else log_warn "Failed to remove pod '$POD_NAME'."; fi
else
     log "Pod '$POD_NAME' did not exist. Skipping pod removal."
fi

# 4. (MODIFIED) Update Local Dependencies if Requested (Now Recreates Venvs)
if [ "$UPDATE_DEPS" = true ]; then
    log "--- Recreating Venvs & Updating Local Python Dependencies ---"
    # Update Backend
    pushd "$BACKEND_DIR" > /dev/null # Change dir quietly
    update_venv_deps "$BACKEND_DIR" "$BACKEND_VENV_DIR" "backend"
    popd > /dev/null # Return to previous dir quietly

    # Update Frontend
    pushd "$FRONTEND_DIR" > /dev/null
    update_venv_deps "$FRONTEND_DIR" "$FRONTEND_VENV_DIR" "frontend"
    popd > /dev/null
    log "--- Local Venv Recreation & Dependency Update Finished ---"
else
    log "Skipping local venv recreation/dependency update (use -u or --update-deps flag to enable)."
fi


# 5. Rebuild Backend Image
log "Navigating to backend directory: $BACKEND_DIR"
cd "$BACKEND_DIR"
log "Building backend image: $BACKEND_IMAGE" # Uses updated variable
if $PODMAN build --tag "$BACKEND_IMAGE" -f Containerfile .; then
    log_success "Backend image '$BACKEND_IMAGE' built successfully."
else
    log_error "Backend image build failed."
    exit 1
fi

# 6. Rebuild Frontend Image
log "Navigating to frontend directory: $FRONTEND_DIR"
cd "$FRONTEND_DIR"
log "Building frontend image: $FRONTEND_IMAGE" # Uses updated variable
if $PODMAN build --tag "$FRONTEND_IMAGE" -f Containerfile .; then
     log_success "Frontend image '$FRONTEND_IMAGE' built successfully."
else
     log_error "Frontend image build failed."
     exit 1
fi

log "Navigating back to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

log_success "------------------------------------------"
log_success "RAG PoC Rebuild Process Completed." # Updated log message
log_success "You can now start the application using:"
log_success "  ./scripts/manage_pod.sh start"
log_success "  OR"
log_success "  systemctl --user start rag-poc.service" # Updated service name
log_success "------------------------------------------"

exit 0
