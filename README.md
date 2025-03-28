
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

3.  **Create `.env` File:**
    The init script will warn if `backend/.env` is missing. Create this file manually inside the `backend/` directory and add your Google API key:
    ```dotenv
    # backend/.env
    GOOGLE_API_KEY=YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE
    ```
    **IMPORTANT:** This file is gitignored and must be created manually.

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
        ```

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

*   **Option 1: Manual Rebuild**
    Use the rebuild script. This stops the application, removes the old pod/containers, and builds new images. You will need to restart the application afterwards.
    ```bash
    # Stop the application if running (e.g., systemctl --user stop rag-poc.service)
    ./scripts/rebuild_poc.sh
    # Restart the application (e.g., systemctl --user start rag-poc.service)
    ```

*   **Option 2: Automated Rebuild & Restart**
    Use the combined script to stop the service, rebuild, and start the service again.
    ```bash
    ./scripts/update_and_restart.sh
    ```

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
