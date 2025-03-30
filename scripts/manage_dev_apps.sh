#!/bin/bash

# Script to manage starting and stopping local development servers (backend & frontend)
# without containers.

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
# Temporarily disable during 'source activate' using set +u / set -u
# set -u
# Prevent errors in pipelines from being masked
set -o pipefail

# --- Configuration ---
readonly SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly PROJECT_ROOT=$( cd -- "$(dirname "$SCRIPT_DIR")" &> /dev/null && pwd )

readonly BACKEND_DIR="$PROJECT_ROOT/backend"
readonly FRONTEND_DIR="$PROJECT_ROOT/frontend"
readonly BACKEND_VENV_DIR="$BACKEND_DIR/venv"
readonly FRONTEND_VENV_DIR="$FRONTEND_DIR/venv_streamlit"
readonly PID_FILE="$PROJECT_ROOT/.dev_pids" # File to store PIDs

# --- Helper Functions ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"
}
log_warn() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARN - $1"
}
log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1" >&2
}
log_success() {
     echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS - $1"
}

# --- Check Virtual Environments ---
check_venvs() {
    if [ ! -d "$BACKEND_VENV_DIR" ] || [ ! -f "$BACKEND_VENV_DIR/bin/activate" ]; then
        log_error "Backend virtual environment not found or incomplete: $BACKEND_VENV_DIR"
        log_error "Please run './scripts/dev-env_init.sh' first."
        exit 1
    fi
     if [ ! -d "$FRONTEND_VENV_DIR" ] || [ ! -f "$FRONTEND_VENV_DIR/bin/activate" ]; then
        log_error "Frontend virtual environment not found or incomplete: $FRONTEND_VENV_DIR"
        log_error "Please run './scripts/dev-env_init.sh' first."
        exit 1
    fi
    log "Virtual environments found."
}

# --- Start Function ---
start_apps() {
    log "Starting development servers..."
    check_venvs

    if [ -f "$PID_FILE" ]; then
        log_warn "PID file ($PID_FILE) already exists. Are servers already running?"
        log_warn "Attempting to stop existing servers first..."
        stop_apps # Attempt cleanup before starting again
    fi

    # Activate backend venv and start uvicorn in background
    log "Starting backend (uvicorn)..."
    ( # Run in subshell to isolate environment activation
        cd "$BACKEND_DIR" || exit 1 # Go into backend dir for uvicorn
        set +u # Disable strict unset variable check for 'source'
        source "$BACKEND_VENV_DIR/bin/activate"
        set -u
        # Start uvicorn, redirect stdout/stderr to logs, run in background
        nohup uvicorn main:app --host 0.0.0.0 --port 8000 --reload &> uvicorn.log &
        backend_pid=$!
        echo "backend_pid=$backend_pid" > "$PID_FILE" # Store PID
        log "Backend (uvicorn) started with PID $backend_pid. Logs: $BACKEND_DIR/uvicorn.log"
        # Allow some time for startup
        sleep 2
    ) || { log_error "Failed to start backend."; exit 1; }


    # Activate frontend venv and start streamlit in background
    log "Starting frontend (streamlit)..."
     ( # Run in subshell
        cd "$FRONTEND_DIR" || exit 1 # Go into frontend dir for streamlit
        set +u
        source "$FRONTEND_VENV_DIR/bin/activate"
        set -u
        # Start streamlit, redirect stdout/stderr, run in background
        # Streamlit should automatically detect .streamlit/config.toml
        nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &> streamlit.log &
        frontend_pid=$!
        echo "frontend_pid=$frontend_pid" >> "$PID_FILE" # Append PID
        log "Frontend (streamlit) started with PID $frontend_pid. Logs: $FRONTEND_DIR/streamlit.log"
    ) || { log_error "Failed to start frontend."; exit 1; }

    log_success "Development servers started."
    log "Backend API should be available at http://localhost:8000"
    log "Frontend UI should be available at http://localhost:8501"
    log "Use './scripts/manage_dev_apps.sh stop' to stop them."
}

# --- Stop Function ---
stop_apps() {
    log "Stopping development servers..."
    killed_count=0
    backend_pid=""
    frontend_pid=""

    # Attempt to read PIDs from file if it exists (for logging context)
    if [ -f "$PID_FILE" ]; then
        log "PID file found ($PID_FILE). Reading PIDs..."
        backend_pid=$(grep 'backend_pid=' "$PID_FILE" | cut -d'=' -f2 || true) # Use || true to avoid exit on grep fail
        frontend_pid=$(grep 'frontend_pid=' "$PID_FILE" | cut -d'=' -f2 || true)
        log "Read backend PID: ${backend_pid:-'Not Found'}, frontend PID: ${frontend_pid:-'Not Found'}"
    else
        log_warn "PID file ($PID_FILE) not found. Will attempt to stop based on process pattern."
    fi

    # --- Backend Stop Logic ---
    # Use pkill -f to find processes matching the uvicorn command for port 8000
    # This is potentially more robust than relying on the PID file, especially for --reload
    backend_pattern="uvicorn main:app.*--port 8000"
    log "Attempting to stop backend processes matching pattern: '$backend_pattern'"

    # Check if any matching processes exist before trying to kill
    if pgrep -f "$backend_pattern" > /dev/null; then
        log "Found running backend processes. Sending TERM signal..."
        # Send TERM signal first (graceful shutdown)
        # Use || true to prevent script exit if no process is found (e.g., already stopped)
        pkill -f "$backend_pattern" || true
        sleep 2 # Allow time for graceful shutdown

        # Check again if processes still exist
        if pgrep -f "$backend_pattern" > /dev/null; then
            log_warn "Backend processes still running after TERM signal. Sending KILL signal..."
            # Send KILL signal (forceful shutdown)
            pkill -9 -f "$backend_pattern" || true
            sleep 1 # Give OS time
            if pgrep -f "$backend_pattern" > /dev/null; then
                 log_error "Backend processes failed to stop even after KILL signal."
                 # Optionally, add more drastic measures or just report the error
            else
                 log "Backend processes stopped via KILL signal."
                 killed_count=$((killed_count + 1))
            fi
        else
            log "Backend processes stopped gracefully via TERM signal."
            killed_count=$((killed_count + 1))
        fi
    else
        log_warn "No running backend processes found matching pattern '$backend_pattern'."
        # If PID file exists but process doesn't, log the PID from file for info
        if [ -n "$backend_pid" ]; then
             log_warn "(PID $backend_pid was listed in $PID_FILE but process not found matching pattern)."
        fi
    fi

    # --- Frontend Stop Logic ---
    # Still use PID for frontend as it's less likely to have complex process trees
    if [ -n "$frontend_pid" ]; then
        log "Attempting to stop frontend (PID: $frontend_pid)..."
        if ps -p "$frontend_pid" > /dev/null; then
            kill "$frontend_pid" || true # Send TERM, ignore error if already gone
            sleep 1
            if ps -p "$frontend_pid" > /dev/null; then
                 log_warn "Frontend (PID: $frontend_pid) did not stop gracefully, sending KILL signal..."
                 kill -9 "$frontend_pid" || true # Send KILL, ignore error
            fi
            # Check one last time
            if ! ps -p "$frontend_pid" > /dev/null; then
                 log "Frontend stopped."
                 killed_count=$((killed_count + 1))
            else
                 log_error "Frontend (PID: $frontend_pid) failed to stop."
            fi
        else
            log_warn "Frontend process (PID: $frontend_pid from $PID_FILE) not found."
        fi
    else
         log_warn "Frontend PID not found in $PID_FILE (or file missing)."
         # Optionally, could add pkill -f for streamlit pattern here too if needed
    fi

    # Clean up PID file
    rm -f "$PID_FILE"
    log "PID file removed."

    if [ "$killed_count" -gt 0 ]; then
        log_success "Development servers stopped."
    else
         log_warn "No running processes associated with PIDs found in $PID_FILE were stopped."
    fi
}

# --- Restart Function ---
restart_apps() {
    log "Restarting development servers..."
    stop_apps
    log "Waiting a moment before starting again..."
    sleep 2 # Give a brief pause
    start_apps
    log_success "Development servers restarted."
}

# --- Usage Instructions ---
usage() {
    echo "Usage: $0 {start|stop|restart}"
    echo "  start:   Starts the backend (uvicorn) and frontend (streamlit) servers in the background."
    echo "  stop:    Stops the background servers using PIDs stored in $PID_FILE."
    echo "  restart: Stops the servers if running, then starts them again."
    exit 1
}

# --- Main Logic ---
COMMAND=$1

if [ -z "$COMMAND" ]; then
    usage
fi

case "$COMMAND" in
    start)
        start_apps
        ;;
    stop)
        stop_apps
        ;;
    restart)
        restart_apps
        ;;
    *)
        log_error "Invalid command: $COMMAND"
        usage
        ;;
esac

exit 0
