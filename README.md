
## Prerequisites

*   **Podman:** Installed and running. (Tested with Podman >= 5.0).
*   **Git:** For cloning the repository.
*   **Python 3:** Primarily for creating virtual environments locally if modifying dependencies (Container uses Python 3.11).
*   **Google API Key:** A valid API key for Google AI Studio or Google Cloud AI Platform with access to the desired Gemini model.
*   **Operating System:** Developed on Fedora 42, compatible with CentOS Stream 9 / RHEL 9 and other Linux distributions with Podman support. SELinux enabled is assumed (uses `:Z` flag on volumes).

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd gemini-chat-poc
    ```

2.  **Initialize Local Development Environment:**
    Run the initialization script. This checks prerequisites, creates local Python virtual environments (`backend/venv`, `frontend/venv_streamlit`), and installs initial dependencies.
    ```bash
    ./scripts/dev-env_init.sh
    ```
    *Note: This script is safe to run multiple times but only creates venvs if they don't exist.*

3.  **Configure API Key (`.env` file):**
    The initialization script (`dev-env_init.sh`) checks for the `backend/.env` file.
    *   If `backend/.env` **does not exist**, the script will automatically copy the template file `backend/template.env` to `backend/.env`.
    *   You **must** then edit the newly created `backend/.env` file and replace the placeholder `YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE` with your actual Google API key.
    ```dotenv
    # backend/.env (Edit this file after it's copied)
    GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_API_KEY
    ```
    *   If `backend/.env` **already exists**, the script will leave it untouched. Ensure it contains your valid API key.

    **IMPORTANT:** The `backend/.env` file is gitignored and contains sensitive credentials. Do not commit it to version control.

4.  **Add Project Data:**
    *   Create subdirectories inside the `projects/` directory for each project... *(rest of this step remains the same)*

5.  **Build Initial Container Images:**
    Run the rebuild script once initially to build both backend and frontend images:
    ```bash
    ./scripts/rebuild_poc.sh
    ```
    *(Optional: If you modified requirements.txt after running the init script, use `./scripts/rebuild_poc.sh -u` to sync your venvs before the first build).*

6.  **Setup Systemd Service (Optional but Recommended):**
    *   Copy the example service file... *(rest of this step remains the same)*emctl --user enable rag-poc.service

## Running the Application

There are two primary ways to run the application:

**Method 1: Using the Systemd User Service (Recommended)**

*   **Start:**
    ```bash
    systemctl --user start rag-poc.service
    ```
*   **Stop:**
    ```bash
    systemctl --user stop rag-poc.service
    ```
*   **Check Status:**
    ```bash
    systemctl --user status rag-poc.service
    ```
*   **View Logs:**
    ```bash
    journalctl --user -u rag-poc.service -f
    ```

**Method 2: Using the Management Script Directly**

*   **Start:**
    ```bash
    ./scripts/manage_pod.sh start
    ```
*   **Stop:**
    ```bash
    ./scripts/manage_pod.sh stop
    ```

**Accessing the UI:**

Once the application is running (using either method), open your web browser and navigate to:

`http://localhost:8501`

## Development

**Rebuilding Container Images:**

If you make changes to the backend (`backend/`) or frontend (`frontend/`) code or their dependencies (`requirements.txt`), you need to rebuild the container images.

The `rebuild_poc.sh` script handles stopping the application, cleaning up old containers/pods, and building new images.

*   **Option 1: Rebuild Only**
    Use this if you only changed code and not dependencies, or if you manually managed your local virtual environments.
    ```bash
    # Make sure the application is stopped first (e.g., systemctl --user stop rag-poc.service)
    ./scripts/rebuild_poc.sh
    # Restart the application after rebuild (e.g., systemctl --user start rag-poc.service)
    ```

*   **Option 2: Recreate Local Venvs, Update Dependencies & Rebuild**
    Use this to ensure a clean local development state. This option **completely removes** your existing local Python virtual environments (`backend/venv` and `frontend/venv_streamlit`), recreates them, reinstalls dependencies directly from the current `requirements.txt` files, and then freezes the resulting package versions back into `requirements.txt` *before* rebuilding the container images.
    ```bash
    # WARNING: This removes backend/venv and frontend/venv_streamlit first!
    # Make sure the application is stopped (e.g., systemctl --user stop rag-poc.service)
    ./scripts/rebuild_poc.sh --update-deps
    # OR use the short flag:
    # ./scripts/rebuild_poc.sh -u
    # Restart the application after rebuild (e.g., systemctl --user start rag-poc.service)
    ```
    *Note: This option effectively replaces the need for the separate `./scripts/dev-env_init.sh` if run with `-u`, as it recreates the venvs anyway.*

*   **Option 3: Force Rebuild Without Cache (`--no-cache`)**
    Use this flag (optionally combined with `-u`) if you suspect build cache corruption or want to ensure all build steps are re-run from scratch, ignoring any cached layers. This can fix errors like "Digest did not match" or issues where code/dependency changes aren't reflected in the image despite modifications.
    ```bash
    # Rebuild without cache
    ./scripts/rebuild_poc.sh --no-cache

    # Recreate venvs AND rebuild without cache
    ./scripts/rebuild_poc.sh -u --no-cache
    ```

*   **Option 4: Automated Rebuild & Restart**
    Use the `update_and_restart.sh` script to combine stopping the service, running the basic rebuild (without dependency update or cache clearing), and starting the service again.
    ```bash
    ./scripts/update_and_restart.sh
    ```
    *Note: To include the dependency update or no-cache options in this automated flow, you would need to modify `update_and_restart.sh` to pass the desired flags to `rebuild_poc.sh`.*

## Deployment Notes (OpenShift/Kubernetes)

*   The `deployment/rag-poc-kube.yaml` file is auto-generated from the local Podman pod setup.
*   **This file requires significant adaptation before it can be applied to a real OpenShift or Kubernetes cluster.** Key changes needed include:
    *   **Image Registry:** Push images to a registry accessible by the cluster (e.g., OpenShift internal registry, Quay.io) and update the `image:` fields.
    *   **Configuration:** Replace the `.env` file volume mount with Kubernetes `Secrets`.
    *   **Storage:** Replace the `hostPath` volume for `projects` with `PersistentVolumeClaim` (PVC).
    *   **Workload Type:** Replace `kind: Pod` with `kind: Deployment` or `kind: DeploymentConfig`.
    *   **Networking:** Define Kubernetes `Services` (e.g., `ClusterIP`) and OpenShift `Routes` for external access.
    *   **Security:** Adjust `securityContext` based on cluster Security Context Constraints (SCCs).

## Future Improvements / TODOs

*   Implement proper token counting before sending requests to the Gemini API to avoid errors.
*   Explore strategies (chunking, summarization) for handling document sets that exceed the context window, potentially moving towards a RAG approach if needed.
*   Improve error handling and reporting in both backend and frontend.
*   Make file parsing asynchronous (`aiofiles`) in the backend for potentially better performance.
*   Enhance the Streamlit UI (e.g., displaying source document snippets, better loading indicators).
*   Create robust, parameterized OpenShift deployment manifests (DeploymentConfig, PVC, Secret, Service, Route).
*   Add support for more document types (.xlsx, improved .md handling).
