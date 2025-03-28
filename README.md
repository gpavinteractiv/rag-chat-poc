
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

2.  **Create `.env` File:**
    Create a file named `.env` inside the `backend/` directory and add your Google API key:
    ```dotenv
    # backend/.env
    GOOGLE_API_KEY=YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE
    ```
    **IMPORTANT:** This file is gitignored and must be created manually.

3.  **Add Project Data:**
    *   Create subdirectories inside the `projects/` directory for each project you want to define (e.g., `projects/my_legal_case`).
    *   Inside each project directory (e.g., `projects/my_legal_case/`):
        *   Create a `system_prompt.txt` file containing the base instructions for the LLM for this project.
        *   Create a `filelist.csv` file. It **must** contain at least one column named exactly `"file name"` listing the document files to load relative to this project directory.
        *   Place the actual document files (`.pdf`, `.csv`, `.docx`, `.md`) listed in `filelist.csv`.

4.  **Build Container Images:**
    Run the rebuild script once initially to build both backend and frontend images:
    ```bash
    ./scripts/rebuild_poc.sh
    ```

5.  **Setup Systemd Service (Optional but Recommended):**
    *   Copy the example service file to your user's systemd configuration directory:
        ```bash
        mkdir -p ~/.config/systemd/user/
        cp deployment/systemd-example/gemini-poc.service.example ~/.config/systemd/user/gemini-poc.service
        ```
    *   **Edit** `~/.config/systemd/user/gemini-poc.service` and **replace any placeholder paths** (like `/path/to/gemini-chat-poc`) with the actual absolute path to your cloned project directory. Also ensure the `User=` line is commented out or removed if running as a user service.
    *   Reload the systemd user daemon:
        ```bash
        systemctl --user daemon-reload
        ```
    *   Enable the service to start automatically on login:
        ```bash
        systemctl --user enable gemini-poc.service
        ```

## Running the Application

There are two primary ways to run the application:

**Method 1: Using the Systemd User Service (Recommended)**

*   **Start:**
    ```bash
    systemctl --user start gemini-poc.service
    ```
*   **Stop:**
    ```bash
    systemctl --user stop gemini-poc.service
    ```
*   **Check Status:**
    ```bash
    systemctl --user status gemini-poc.service
    ```
*   **View Logs:**
    ```bash
    journalctl --user -u gemini-poc.service -f
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
    # Stop the application if running (e.g., systemctl --user stop gemini-poc.service)
    ./scripts/rebuild_poc.sh
    # Restart the application (e.g., systemctl --user start gemini-poc.service)
    ```

*   **Option 2: Automated Rebuild & Restart**
    Use the combined script to stop the service, rebuild, and start the service again.
    ```bash
    ./scripts/update_and_restart.sh
    ```

## Deployment Notes (OpenShift/Kubernetes)

*   The `deployment/gemini-poc-kube.yaml` file is auto-generated from the local Podman pod setup.
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
