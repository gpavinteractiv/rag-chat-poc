## Prerequisites

*   **Podman:** Installed and running. (Tested with Podman >= 5.0).
*   **Git:** For cloning the repository.
*   **Python 3:** Primarily for creating virtual environments locally if modifying dependencies (Container uses Python 3.11).
*   **Google API Key:** A valid API key for Google AI Studio or Google Cloud AI Platform with access to the desired Gemini model.
*   **OpenRouter API Key:** A valid API key from [OpenRouter.ai](https://openrouter.ai/) if you intend to use models hosted there.
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
    *Note: This script is safe to run multiple times but only creates venvs if they don't exist. It installs the necessary Python packages into the `backend/venv`. Ensure this is run successfully before attempting to use the offline scripts mentioned in steps 6 and 7.*

3.  **Configure API Key (`.env` file):**
    The initialization script (`dev-env_init.sh`) checks for the `backend/.env` file.
    *   If `backend/.env` **does not exist**, the script will automatically copy the template file `backend/template.env` to `backend/.env`.
    *   You **must** then edit the newly created `backend/.env` file and replace the placeholders `YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE` and `YOUR_OPENROUTER_API_KEY_HERE` with your actual API keys. If you don't plan to use one of the providers, you can leave its placeholder, but the application might log warnings or errors if you try to select a model from that provider later.
    ```dotenv
    # backend/.env (Edit this file after it's copied)
    GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_API_KEY
    OPENROUTER_API_KEY=YOUR_ACTUAL_OPENROUTER_API_KEY
    ```
    *   If `backend/.env` **already exists**, the script will leave it untouched. Ensure it contains your valid API keys for the providers you intend to use.

    **IMPORTANT:** The `backend/.env` file is gitignored and contains sensitive credentials. Do not commit it to version control.

    *Note: This script is safe to run multiple times but only creates venvs if they don't exist.*

4.  **Add Project Data:**
    *   The `projects/` directory holds the data for different RAG contexts. Each subdirectory represents a separate project.
    *   A `projects/_template/` directory is provided as a starting point for new projects. It contains:
        *   `system_prompt.txt`: A default system prompt for the LLM. Customize this for your project's needs.
        *   `filelist.csv`: An empty CSV file. You **must** edit this file and add the relative paths (within the project directory) of the documents you want the LLM to use as context under a column named `file name`.
    *   **To create a new project:**
        1.  Copy the `projects/_template/` directory and rename the copy (e.g., `projects/MyNewProject/`).
        2.  Place your context documents (PDF, DOCX, MD, CSV) inside your new project directory (e.g., `projects/MyNewProject/`).
        3.  Edit `projects/MyNewProject/filelist.csv` and list the filenames of your documents under the `file name` column header.
        4.  (Optional) Edit `projects/MyNewProject/system_prompt.txt` to tailor the LLM's instructions.
    *   **Note:** Directories within `projects/` that start with an underscore (`_`), like `_template`, are ignored by the backend and will not appear in the project list in the UI.

5.  **Build Initial Container Images:**
    Run the rebuild script once initially to build both backend and frontend images:
    ```bash
    ./scripts/rebuild_poc.sh
    ```
    *(Optional: If you modified requirements.txt after running the init script, use `./scripts/rebuild_poc.sh -u` to sync your venvs before the first build).*

6.  **Calculate Initial Token Counts (Optional, Offline):**
    This step calculates estimated token counts for documents listed in each project's `filelist.csv` and adds/updates a `token_count` column. This count is used by the backend to manage the context window. Run this script after adding or modifying documents in a project.

    *   **For all projects:**
        Use the `update_all_token_counts.sh` script. It iterates through all valid project directories.
        ```bash
        # Run from the project root directory
        ./scripts/update_all_token_counts.sh
        ```
        *Note: This script activates the backend virtual environment internally using `source`. It may take some time depending on the number and size of your documents.*

    *   **For a single project:**
        Use the `run_calculate_tokens.sh` wrapper script, providing the path to the specific project directory.
        ```bash
        # Example for a project named 'MyProject'
        # Run from the project root directory
        # Make executable first: chmod +x scripts/run_calculate_tokens.sh
        ./scripts/run_calculate_tokens.sh projects/MyProject/
        ```
        *Note: This wrapper script directly executes the Python script using the interpreter inside `backend/venv`, so you don't need to activate the environment manually beforehand.*

7.  **Pre-generate Document Cache (Optional, Offline):**
    To improve the performance of the first chat request for each project, you can pre-generate a disk cache of the parsed document content. The backend will automatically use this cache if it's valid (checking file modification times). If the cache is missing or stale, the backend will parse documents on the fly during the chat request and update the disk cache automatically.

    First, make the wrapper script executable:
    ```bash
    chmod +x scripts/run_pregenerate_cache.sh
    ```

    *   **For all projects:**
        ```bash
        # Run from the project root directory
        ./scripts/run_pregenerate_cache.sh
        ```
    *   **For a single project:**
        ```bash
        # Example for a project named 'MyProject'
        # Run from the project root directory
        ./scripts/run_pregenerate_cache.sh --project-directory projects/MyProject/
        ```
    *Note: This script uses the backend virtual environment via the wrapper. The cache is stored in `backend/.cache/parsed_docs/` and is ignored by Git.*

8.  **Setup Systemd Service (Optional but Recommended):**
    *   Copy the example service file to your systemd user directory:
        ```bash
        mkdir -p ~/.config/systemd/user/
        cp deployment/systemd-example/rag-poc.service.example ~/.config/systemd/user/rag-poc.service
        ```
    *   **Edit the copied service file (`~/.config/systemd/user/rag-poc.service`)** and replace the placeholder `/path/to/your/cloned/repo/gemini-chat-poc` with the **absolute path** to where you cloned this repository. For example:
        ```ini
        [Unit]
        Description=RAG Chat PoC Podman Pod
        After=network.target

        [Service]
        WorkingDirectory=/home/youruser/dev/gemini-chat-poc # <-- CHANGE THIS PATH
        ExecStart=/usr/bin/podman pod start rag-poc-pod
        ExecStop=/usr/bin/podman pod stop rag-poc-pod
        Restart=on-failure

        [Install]
        WantedBy=default.target
        ```
    *   Reload the systemd user daemon:
        ```bash
        systemctl --user daemon-reload
        ```
    *   Enable the service to start on login (optional):
        ```bash
        systemctl --user enable rag-poc.service
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

## Context Management

The backend constructs the prompt context sent to the LLM based on the selected project's documents and the user's query.

*   **Token Limit:** The maximum number of tokens allowed in the context can be adjusted in the frontend sidebar ("Max Context Tokens"). The default is 900,000 tokens, suitable for models like Gemini 2.5 Pro.
*   **Pre-calculated Counts:** The backend uses pre-calculated token counts for each document (stored in `projects/<project_name>/filelist.csv` under the `token_count` column, generated by `scripts/calculate_token_counts.py`).
*   **Document Inclusion:** Documents are included in the context sequentially based on their order in `filelist.csv`.
*   **Skipping Documents:** If adding the next document (based on its pre-calculated token count) would exceed the configured "Max Context Tokens" limit, that document **and all subsequent documents** are skipped. They will not be included in the context sent to the LLM.
*   **Warning:** If any documents are skipped due to the token limit, a warning message will appear in the frontend UI listing the skipped files.

## Future Improvements / TODOs

*   Implement proper token counting before sending requests to the Gemini API to avoid errors.
*   Explore strategies (chunking, summarization) for handling document sets that exceed the context window, potentially moving towards a RAG approach if needed.
*   Improve error handling and reporting in both backend and frontend.
*   Make file parsing asynchronous (`aiofiles`) in the backend for potentially better performance.
*   Enhance the Streamlit UI (e.g., displaying source document snippets, better loading indicators).
*   Create robust, parameterized OpenShift deployment manifests (DeploymentConfig, PVC, Secret, Service, Route).
*   Add support for more document types (.xlsx, improved .md handling).
