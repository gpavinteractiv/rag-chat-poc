#!/bin/bash

# This script stops the systemd service, runs the rebuild script,
# and then starts the systemd service again.

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Prevent errors in pipelines from being masked
set -o pipefail

# --- Configuration ---
readonly SERVICE_NAME="rag-poc.service"

# Determine project root directory (assuming script is in PROJECT_ROOT/scripts)
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PROJECT_ROOT=$( cd -- "$(dirname "$SCRIPT_DIR")" &> /dev/null && pwd )
readonly REBUILD_SCRIPT="$PROJECT_ROOT/scripts/rebuild_poc.sh"

# --- Helper Functions ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
}

log_success() {
     echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS - $1"
}


# --- Main Logic ---
log "Starting Update and Restart process..."

# 1. Check if rebuild script exists and is executable
if [ ! -x "$REBUILD_SCRIPT" ]; then
    log_error "Rebuild script not found or not executable at: $REBUILD_SCRIPT"
    exit 1
fi

# 2. Stop the systemd user service
log "Attempting to stop systemd user service: $SERVICE_NAME..."
# Use || true to prevent script exit if service is already stopped or doesn't exist
systemctl --user stop "$SERVICE_NAME" || true
log "Stop command issued for service $SERVICE_NAME."
# Add a small delay to allow resources to be released before rebuild tries removing things
sleep 2

# 3. Run the rebuild script
log "Executing rebuild script: $REBUILD_SCRIPT"
# The rebuild script itself has error handling and will exit if it fails (due to set -e)
"$REBUILD_SCRIPT"
log "Rebuild script completed."

# 4. Start the systemd user service
log "Attempting to start systemd user service: $SERVICE_NAME..."
if systemctl --user start "$SERVICE_NAME"; then
    log_success "Service $SERVICE_NAME started successfully."
else
    log_error "Failed to start service $SERVICE_NAME."
    log_error "Please check the service status and logs:"
    log_error "  systemctl --user status $SERVICE_NAME"
    log_error "  journalctl --user -u $SERVICE_NAME -n 50 --no-pager"
    exit 1 # Exit with error if start fails
fi

log_success "------------------------------------------"
log_success "Update and Restart process completed successfully!"
log_success "Application should be running and accessible."
log_success "------------------------------------------"

exit 0
