#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in pipelines from being masked
set -o pipefail

# --- Configuration (Should match manage_pod.sh) ---
readonly POD_NAME="gemini-poc-pod"
readonly BACKEND_CONTAINER_NAME="gemini-backend-container"
readonly FRONTEND_CONTAINER_NAME="gemini-frontend-container"
readonly BACKEND_IMAGE="localhost/gemini-chat-backend:0.1"
readonly FRONTEND_IMAGE="localhost/gemini-chat-frontend:0.1"

# Determine project root directory (assuming script is in PROJECT_ROOT/scripts)
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
        echo "  -u, --update-deps: Update local venvs from requirements.txt before rebuilding."
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

# Function to update dependencies in a specific venv
update_venv_deps() {
    local target_dir="$1"
    local venv_dir="$2"
    local req_file="$target_dir/requirements.txt"
    local component_name="$3" # e.g., "backend", "frontend"

    log "Updating dependencies for $component_name in $target_dir"

    if [ ! -d "$venv_dir" ]; then
        log_warn "Virtual environment directory not found: $venv_dir. Skipping $component_name dependency update."
        # Optional: could offer to create it: python3 -m venv "$venv_dir"
        return
    fi
     if [ ! -f "$req_file" ]; then
        log_warn "Requirements file not found: $req_file. Skipping $component_name dependency update."
        return
    fi

    log "Activating venv: $venv_dir/bin/activate"
    # Use subshell to avoid polluting current environment and needing deactivate
    (
        set +u # Temporarily allow unset variables during source
        source "$venv_dir/bin/activate"
        set -u

        log "Upgrading pip in $component_name venv..."
        pip install --upgrade pip

        log "Installing/updating packages from $req_file..."
        # Use --upgrade to ensure packages listed are upgraded if needed
        pip install -r "$req_file" --upgrade

        log "Freezing updated dependencies back to $req_file..."
        pip freeze > "$req_file"

        log "$component_name dependency update complete."
    ) || { log_error "Failed during $component_name dependency update process."; exit 1; } # Catch errors within subshell

     # Check if subshell succeeded by checking pip execution$? (complicated)
     # Simpler to rely on set -e within the subshell

}


# --- Main Rebuild Logic ---
log "Starting PoC rebuild process..."
if [ "$UPDATE_DEPS" = true ]; then log "Local dependency update requested."; fi

# 1. Stop existing pod (if running)
# (Cleanup logic remains the same)
log "Checking for existing pod '$POD_NAME'..."
if $PODMAN pod exists "$POD_NAME"; then
    log "Pod '$POD_NAME' exists. Attempting to stop..."
    if $PODMAN pod stop "$POD_NAME"; then
        log "Pod '$POD_NAME' stopped."
    else
        log_warn "Failed to stop pod '$POD_NAME'. It might already be stopped."
    fi
else
    log "Pod '$POD_NAME' does not exist. Skipping stop."
fi

# 2. Remove existing containers
log "Removing container '$BACKEND_CONTAINER_NAME' if it exists..."
$PODMAN rm "$BACKEND_CONTAINER_NAME" --force --ignore || log_warn "Container '$BACKEND_CONTAINER_NAME' not found or removal failed."
log "Removing container '$FRONTEND_CONTAINER_NAME' if it exists..."
$PODMAN rm "$FRONTEND_CONTAINER_NAME" --force --ignore || log_warn "Container '$FRONTEND_CONTAINER_NAME' not found or removal failed."

# 3. Remove existing pod
log "Removing pod '$POD_NAME' if it exists..."
if $PODMAN pod exists "$POD_NAME"; then
     if $PODMAN pod rm "$POD_NAME" --force; then
          log "Pod '$POD_NAME' removed."
     else
          log_warn "Failed to remove pod '$POD_NAME'."
     fi
else
     log "Pod '$POD_NAME' did not exist. Skipping pod removal."
fi

# 4. (NEW) Update Local Dependencies if Requested
if [ "$UPDATE_DEPS" = true ]; then
    log "--- Updating Local Python Dependencies ---"
    # Update Backend
    pushd "$BACKEND_DIR" > /dev/null # Change dir quietly
    update_venv_deps "$BACKEND_DIR" "$BACKEND_VENV_DIR" "backend"
    popd > /dev/null # Return to previous dir quietly

    # Update Frontend
    pushd "$FRONTEND_DIR" > /dev/null
    update_venv_deps "$FRONTEND_DIR" "$FRONTEND_VENV_DIR" "frontend"
    popd > /dev/null
    log "--- Local Dependency Update Finished ---"
else
    log "Skipping local dependency update (use -u or --update-deps flag to enable)."
fi


# 5. Rebuild Backend Image
log "Navigating to backend directory: $BACKEND_DIR"
cd "$BACKEND_DIR"
log "Building backend image: $BACKEND_IMAGE"
if $PODMAN build --tag "$BACKEND_IMAGE" -f Containerfile .; then
    log_success "Backend image '$BACKEND_IMAGE' built successfully."
else
    log_error "Backend image build failed."
    exit 1
fi

# 6. Rebuild Frontend Image
log "Navigating to frontend directory: $FRONTEND_DIR"
cd "$FRONTEND_DIR"
log "Building frontend image: $FRONTEND_IMAGE"
if $PODMAN build --tag "$FRONTEND_IMAGE" -f Containerfile .; then
     log_success "Frontend image '$FRONTEND_IMAGE' built successfully."
else
     log_error "Frontend image build failed."
     exit 1
fi

log "Navigating back to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

log_success "------------------------------------------"
log_success "PoC Rebuild Process Completed."
log_success "You can now start the application using:"
log_success "  ./scripts/manage_pod.sh start"
log_success "  OR"
log_success "  systemctl --user start gemini-poc.service"
log_success "------------------------------------------"

exit 0
