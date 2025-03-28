#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in pipelines from being masked
set -o pipefail

# --- Configuration ---
readonly POD_NAME="rag-poc-pod"
readonly BACKEND_CONTAINER_NAME="rag-backend-container" # The --name we assign
readonly FRONTEND_CONTAINER_NAME="rag-frontend-container" # The --name we assign
readonly BACKEND_IMAGE="localhost/rag-chat-backend:0.1"
readonly FRONTEND_IMAGE="localhost/rag-chat-frontend:0.1"

# Determine project root directory (assuming script is in PROJECT_ROOT/scripts)
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PROJECT_ROOT=$( cd -- "$(dirname "$SCRIPT_DIR")" &> /dev/null && pwd )

readonly BACKEND_DIR="$PROJECT_ROOT/backend"
readonly FRONTEND_DIR="$PROJECT_ROOT/frontend"
readonly ENV_FILE="$BACKEND_DIR/.env"
readonly PROJECTS_DIR="$PROJECT_ROOT/projects"

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

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
}

check_image_exists() {
    local image_name="$1"
    if ! $PODMAN image exists "$image_name"; then
        log_error "Required image '$image_name' not found. Please build it first."
        exit 1
    fi
}

get_pod_status() {
    local pod_name="$1"
    $PODMAN pod ps --filter name="^${pod_name}$" --format "{{.Status}}" | head -n 1
}

# Checks if a container with the given name is running *within the specific pod*
is_container_running_in_pod() {
    local target_pod_name="$1"
    local target_container_name="$2"
    # List containers in the pod, filter by exact name, check if status includes "Up"
    local status
    status=$($PODMAN ps --pod --filter pod="$target_pod_name" --filter name="^${target_container_name}$" --format "{{.Status}}" | head -n 1)
    if [[ "$status" == *"Up"* ]]; then
        return 0 # 0 means success (true in bash conditional)
    else
        return 1 # 1 means failure (false in bash conditional)
    fi
}

# --- Start Script Logic ---
start_all() {
    log "Starting Gemini PoC Pod management script..."

    # Check required files/dirs exist
    if [ ! -f "$ENV_FILE" ]; then log_error "Required environment file not found: $ENV_FILE"; exit 1; fi
    if [ ! -d "$PROJECTS_DIR" ]; then log_error "Required projects directory not found: $PROJECTS_DIR"; exit 1; fi

    # Check required images exist
    check_image_exists "$BACKEND_IMAGE"
    check_image_exists "$FRONTEND_IMAGE"

    # --- Pod Management ---
    local pod_needs_start=false
    if $PODMAN pod exists "$POD_NAME"; then
        log "Pod '$POD_NAME' already exists."
        local pod_status
        pod_status=$(get_pod_status "$POD_NAME")
        if [[ "$pod_status" == *"Running"* || "$pod_status" == *"Degraded"* || "$pod_status" == *"Up"* ]]; then
            log "Pod '$POD_NAME' status is '$pod_status' (considered running)."
        elif [ -z "$pod_status" ]; then
             log_error "Pod '$POD_NAME' exists but couldn't determine status via 'podman pod ps'. Check manually."
             exit 1
        else
            log "Pod status is '$pod_status'. Will attempt to start pod."
            pod_needs_start=true
        fi
    else
        log "Pod '$POD_NAME' does not exist. Creating..."
        if ! $PODMAN pod create --name "$POD_NAME" -p 8000:8000 -p 8501:8501; then
            log_error "Failed to create pod '$POD_NAME'."
            exit 1
        fi
        log "Pod '$POD_NAME' created successfully."
        pod_needs_start=true # Newly created pod also needs starting (or containers added)
    fi

    # Start the pod if needed (before adding/starting containers)
    if [ "$pod_needs_start" = true ]; then
         if ! $PODMAN pod start "$POD_NAME"; then
             log_error "Failed to start pod '$POD_NAME'."
             exit 1
         fi
         log "Pod '$POD_NAME' started/ready."
         sleep 3 # Wait a moment after starting pod
    fi

    # --- Backend Container Management (within the pod) ---
    log "Checking backend container '$BACKEND_CONTAINER_NAME' in pod '$POD_NAME'..."
    if is_container_running_in_pod "$POD_NAME" "$BACKEND_CONTAINER_NAME"; then
        log "Backend container '$BACKEND_CONTAINER_NAME' is running in pod '$POD_NAME'."
    else
        log "Backend container '$BACKEND_CONTAINER_NAME' not found running in pod '$POD_NAME'."
        # Check if container *name* exists globally but isn't running in the pod (ambiguous state)
        if $PODMAN container exists "$BACKEND_CONTAINER_NAME"; then
            log_error "Container '$BACKEND_CONTAINER_NAME' exists but is not running correctly within pod '$POD_NAME'. Manual cleanup required ('podman rm $BACKEND_CONTAINER_NAME', then restart this script)."
            exit 1
        fi
        # Container doesn't exist globally, safe to create within the pod
        log "Creating backend container '$BACKEND_CONTAINER_NAME' in pod '$POD_NAME'..."
        if ! $PODMAN run --detach --pod "$POD_NAME" --name "$BACKEND_CONTAINER_NAME" \
                   --volume "$ENV_FILE:/app/.env:ro,Z" \
                   --volume "$PROJECTS_DIR:/app/../projects:ro,Z" \
                   "$BACKEND_IMAGE"; then
            log_error "Failed to create backend container '$BACKEND_CONTAINER_NAME'."
            # Attempt to check pod logs maybe?
            # $PODMAN logs "$POD_NAME" might show infra logs
            exit 1
        fi
        log "Backend container '$BACKEND_CONTAINER_NAME' created successfully."
    fi

    # --- Frontend Container Management (within the pod) ---
    log "Checking frontend container '$FRONTEND_CONTAINER_NAME' in pod '$POD_NAME'..."
     if is_container_running_in_pod "$POD_NAME" "$FRONTEND_CONTAINER_NAME"; then
        log "Frontend container '$FRONTEND_CONTAINER_NAME' is running in pod '$POD_NAME'."
    else
        log "Frontend container '$FRONTEND_CONTAINER_NAME' not found running in pod '$POD_NAME'."
         if $PODMAN container exists "$FRONTEND_CONTAINER_NAME"; then
             log_error "Container '$FRONTEND_CONTAINER_NAME' exists but is not running correctly within pod '$POD_NAME'. Manual cleanup required ('podman rm $FRONTEND_CONTAINER_NAME', then restart this script)."
             exit 1
         fi
        log "Creating frontend container '$FRONTEND_CONTAINER_NAME' in pod '$POD_NAME'..."
         if ! $PODMAN run --detach --pod "$POD_NAME" --name "$FRONTEND_CONTAINER_NAME" \
                   "$FRONTEND_IMAGE"; then
             log_error "Failed to create frontend container '$FRONTEND_CONTAINER_NAME'."
             exit 1
         fi
         log "Frontend container '$FRONTEND_CONTAINER_NAME' created successfully."
    fi

    log "Gemini PoC Pod startup script completed successfully."
    log "You should be able to access the frontend at http://localhost:8501"
    $PODMAN ps --pod --filter name="$POD_NAME"
}

# --- Stop Script Logic ---
stop_all() {
    log "Stopping Gemini PoC Pod '$POD_NAME'..."
    if $PODMAN pod exists "$POD_NAME"; then
        if ! $PODMAN pod stop "$POD_NAME"; then
            log_error "Failed to stop pod '$POD_NAME'. It might have already been stopped."
        else
             log "Pod '$POD_NAME' stopped successfully."
        fi
    else
        log "Pod '$POD_NAME' does not exist, nothing to stop."
    fi
}


# --- Main Execution Logic ---
case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac

exit 0
