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

# Find podman executable
PODMAN=$(command -v podman)
if [ -z "$PODMAN" ]; then
    echo "Error: podman command not found in PATH." >&2
    exit 1
fi

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


# --- Main Rebuild Logic ---
log "Starting PoC rebuild process..."

# 1. Stop existing pod (if running)
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

# 2. Remove existing containers (associated with the pod or potentially standalone)
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


# 4. Rebuild Backend Image
log "Navigating to backend directory: $BACKEND_DIR"
cd "$BACKEND_DIR"
log "Building backend image: $BACKEND_IMAGE"
if $PODMAN build --tag "$BACKEND_IMAGE" -f Containerfile .; then
    log_success "Backend image '$BACKEND_IMAGE' built successfully."
else
    log_error "Backend image build failed."
    exit 1 # Exit script on build failure
fi

# 5. Rebuild Frontend Image
log "Navigating to frontend directory: $FRONTEND_DIR"
cd "$FRONTEND_DIR"
log "Building frontend image: $FRONTEND_IMAGE"
if $PODMAN build --tag "$FRONTEND_IMAGE" -f Containerfile .; then
     log_success "Frontend image '$FRONTEND_IMAGE' built successfully."
else
     log_error "Frontend image build failed."
     exit 1 # Exit script on build failure
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
