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
    if [ ! -f "$PID_FILE" ]; then
        log_warn "PID file ($PID_FILE) not found. Servers might not be running or were stopped manually."
        return 0 # Not an error if already stopped
    fi

    backend_pid=$(grep 'backend_pid=' "$PID_FILE" | cut -d'=' -f2)
    frontend_pid=$(grep 'frontend_pid=' "$PID_FILE" | cut -d'=' -f2)

    killed_count=0

    if [ -n "$backend_pid" ]; then
        log "Attempting to stop backend (PID: $backend_pid)..."
        # Use kill and check if process exists
        if ps -p "$backend_pid" > /dev/null; then
            kill "$backend_pid"
            sleep 1 # Give it a moment to terminate
            if ps -p "$backend_pid" > /dev/null; then
                 log_warn "Backend (PID: $backend_pid) did not stop gracefully, sending KILL signal..."
                 kill -9 "$backend_pid"
            fi
            log "Backend stopped."
            killed_count=$((killed_count + 1))
        else
            log_warn "Backend process (PID: $backend_pid) not found."
        fi
    else
         log_warn "Backend PID not found in $PID_FILE."
    fi

     if [ -n "$frontend_pid" ]; then
        log "Attempting to stop frontend (PID: $frontend_pid)..."
        if ps -p "$frontend_pid" > /dev/null; then
            kill "$frontend_pid"
            sleep 1
            if ps -p "$frontend_pid" > /dev/null; then
                 log_warn "Frontend (PID: $frontend_pid) did not stop gracefully, sending KILL signal..."
                 kill -9 "$frontend_pid"
            fi
            log "Frontend stopped."
            killed_count=$((killed_count + 1))
        else
            log_warn "Frontend process (PID: $frontend_pid) not found."
        fi
    else
         log_warn "Frontend PID not found in $PID_FILE."
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
