# /mnt/LAB/rag-chat-poc/frontend/app.py

import streamlit as st
import httpx # For making API calls to the backend
import logging
import os
from typing import List, Dict, Optional, Tuple, TypedDict # Added TypedDict

# --- Configuration ---
# Using os.getenv temporarily until config refactor is complete/verified
# TODO: Replace with config import once verified: from config import settings
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
PROJECT_LIST_ENDPOINT = f"{BACKEND_URL}/projects"
MODELS_ENDPOINT = f"{BACKEND_URL}/models"
MODEL_DETAILS_ENDPOINT = f"{BACKEND_URL}/model-details"
# PROJECT_TOKENS_ENDPOINT removed
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="RAG Chat PoC",
    page_icon="🤖",
    layout="wide"
)

# Configure logging - Keep at INFO level to avoid file watcher loop with DEBUG
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', force=True)
logger = logging.getLogger(__name__)
logger.info("Frontend logging level set to: INFO")


# --- Type Definitions ---
class ProjectInfo(TypedDict):
    name: str
    description: Optional[str]
    file_count: int
    total_document_tokens: Optional[int] # Add field for pre-calculated token sum

# --- Helper Functions ---

@st.cache_data(ttl=300) # Cache project list for 5 minutes
def get_projects() -> List[ProjectInfo]:
    """Fetches the list of projects (including name and file count) from the backend API."""
    logger.info(f"Fetching project list from {PROJECT_LIST_ENDPOINT}")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(PROJECT_LIST_ENDPOINT)
            response.raise_for_status()
            projects_data: List[ProjectInfo] = response.json()
            logger.info(f"Successfully fetched {len(projects_data)} projects.")
            return sorted(projects_data, key=lambda p: p.get("name", ""))
    except httpx.RequestError as e:
        logger.error(f"HTTP error fetching projects: {e}", exc_info=True)
        st.error(f"Failed to connect to backend API ({BACKEND_URL}) to get projects. Is it running? Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching or parsing projects: {e}", exc_info=True)
        st.error(f"An error occurred while fetching projects: {e}")
        return []

@st.cache_data(ttl=300) # Cache model list for 5 minutes
def get_available_models() -> Dict[str, List[str]]:
    """Fetches the list of available models from the backend API."""
    logger.info(f"Fetching available models from {MODELS_ENDPOINT}")
    models_dict = {}
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(MODELS_ENDPOINT)
            response.raise_for_status()
            models_data = response.json()
            if "providers" in models_data:
                for provider_info in models_data["providers"]:
                    provider = provider_info.get("provider")
                    models = provider_info.get("models")
                    if provider and models:
                        models_dict[provider] = sorted(models)
                logger.info(f"Successfully fetched models for providers: {list(models_dict.keys())}")
            else:
                 logger.warning(f"Unexpected format received from {MODELS_ENDPOINT}: {models_data}")
            return models_dict
    except httpx.RequestError as e:
        logger.error(f"HTTP error fetching models: {e}", exc_info=True)
        st.error(f"Failed to connect to backend API ({BACKEND_URL}) to get models. Is it running? Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching or parsing models: {e}", exc_info=True)
        st.error(f"An error occurred while fetching models: {e}")
        return {}

@st.cache_data(ttl=3600) # Cache model details for 1 hour
def get_model_details(provider: str, model_name: str) -> Optional[Dict]:
    """Fetches details for a specific model from the backend."""
    if not provider or not model_name:
        return None
    api_url = f"{MODEL_DETAILS_ENDPOINT}/{provider}/{model_name}"
    logger.info(f"Fetching model details from {api_url}")
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(api_url)
            response.raise_for_status()
            details_data = response.json()
            logger.info(f"Successfully fetched details for {provider}/{model_name}")
            return details_data.get("details")
    except httpx.HTTPStatusError as e:
        response_json = e.response.json() if e.response else {}
        detail = response_json.get("detail", e.response.text)
        logger.error(f"HTTP error fetching model details ({e.response.status_code}): {detail}", exc_info=True)
        st.sidebar.warning(f"Could not fetch details for {model_name}: {detail}")
        return {"notes": f"Error fetching details: {detail}"}
    except httpx.RequestError as e:
        logger.error(f"Network error fetching model details: {e}", exc_info=True)
        st.sidebar.warning(f"Network error fetching model details for {model_name}.")
        return {"notes": f"Network error fetching details: {e}"}
    except Exception as e:
        logger.error(f"Error parsing model details: {e}", exc_info=True)
        st.sidebar.warning(f"Error parsing details for {model_name}.")
        return {"notes": f"Error parsing details: {e}"}

# Removed get_project_base_tokens function

def call_chat_api(project_name: str, query: str, provider: str, model_name: str) -> Optional[Dict]:
    """Calls the backend chat API for a given project, query, provider, and model."""
    if not all([project_name, query, provider, model_name]):
        st.error("Missing required information (project, query, provider, or model) for API call.")
        return None

    api_url = f"{CHAT_ENDPOINT}/{project_name}"
    payload = {
        "query": query,
        "provider": provider,
        "model_name": model_name
    }
    logger.info(f"Sending query to {api_url} using {provider}/{model_name}")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Received response from {api_url}")
            return response_data
    except httpx.HTTPStatusError as e:
        response_json = e.response.json() if e.response else {}
        detail = response_json.get("detail", e.response.text)
        logger.error(f"HTTP error calling chat API ({e.response.status_code}): {detail}", exc_info=True)
        st.error(f"API Error ({e.response.status_code}): {detail}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Network error calling chat API: {e}", exc_info=True)
        st.error(f"Network Error: Failed to connect to chat API ({api_url}). Details: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling chat API: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
        return None

      
# --- Main App UI ---
st.title("📄🤖 RAG Chat PoC")
st.caption(f"A proof-of-concept chat agent using Gemini via a FastAPI backend ({BACKEND_URL}).")


# --- Project Selection ---
available_projects_info: List[ProjectInfo] = get_projects()
selected_project = None
selected_project_doc_count = 0
selected_project_total_doc_tokens = None # Initialize total doc tokens

if not available_projects_info:
    st.sidebar.warning("No projects found or backend unavailable.")
    if "selected_project" in st.session_state: del st.session_state.selected_project
    if "selected_project_doc_count" in st.session_state: del st.session_state.selected_project_doc_count
    if "total_document_tokens" in st.session_state: del st.session_state.total_document_tokens # Clear tokens too
else:
    available_project_names = [p["name"] for p in available_projects_info]

    if "selected_project" not in st.session_state or st.session_state.selected_project not in available_project_names:
        st.session_state.selected_project = available_project_names[0]

    current_selection_name = st.session_state.selected_project

    selected_project_info = next((p for p in available_projects_info if p["name"] == current_selection_name), None)
    if selected_project_info:
        selected_project_doc_count = selected_project_info.get("file_count", 0)
        selected_project_total_doc_tokens = selected_project_info.get("total_document_tokens") # Get total tokens
        st.session_state.selected_project_doc_count = selected_project_doc_count
        st.session_state.total_document_tokens = selected_project_total_doc_tokens # Store in session state
    else:
        logger.warning(f"Selected project '{current_selection_name}' not found in fetched list.")
        selected_project_doc_count = 0
        selected_project_total_doc_tokens = None
        st.session_state.selected_project_doc_count = 0
        if "total_document_tokens" in st.session_state: del st.session_state.total_document_tokens


    selected_project = st.sidebar.selectbox(
        "Choose a project:",
        options=available_project_names,
        key="selected_project",
    )


# --- Model Selection ---
available_models_data = get_available_models()
selected_provider = None
selected_model = None

DEFAULT_MODELS = {
    "google": "gemini-2.5-pro-exp-03-25",
    "openrouter": "google/gemini-2.5-pro-exp-03-25:free"
}

if not available_models_data:
    st.sidebar.warning("No models found or backend unavailable.")
else:
    available_providers = list(available_models_data.keys())

    if "selected_provider" not in st.session_state or st.session_state.selected_provider not in available_providers:
        if "google" in available_providers:
            st.session_state.selected_provider = "google"
        elif "openrouter" in available_providers:
            st.session_state.selected_provider = "openrouter"
        elif available_providers:
            st.session_state.selected_provider = available_providers[0]
        else:
            st.session_state.selected_provider = None

    selected_provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        options=available_providers,
        key="selected_provider",
    )

    if selected_provider:
        models_for_provider = available_models_data.get(selected_provider, [])
        if not models_for_provider:
             st.sidebar.error(f"No models listed for provider '{selected_provider}'.")
             selected_model = None
        else:
            preferred_default = DEFAULT_MODELS.get(selected_provider)
            actual_default = None
            if preferred_default and preferred_default in models_for_provider:
                actual_default = preferred_default
            elif models_for_provider:
                actual_default = models_for_provider[0]

            if "selected_model" not in st.session_state or st.session_state.selected_model not in models_for_provider:
                 st.session_state.selected_model = actual_default

            current_model_index = 0
            if st.session_state.selected_model in models_for_provider:
                current_model_index = models_for_provider.index(st.session_state.selected_model)

            selected_model = st.sidebar.selectbox(
                f"Choose Model ({selected_provider}):",
                options=models_for_provider,
                key="selected_model"
            )
    else:
        selected_model = None

# --- Fetch and Store Model Details ---
if selected_provider and selected_model:
    details_fetch_key = f"details_{selected_provider}_{selected_model}"
    if details_fetch_key not in st.session_state:
        logger.info(f"Fetching details for newly selected model: {selected_provider}/{selected_model}")
        fetched_details = get_model_details(selected_provider, selected_model)
        st.session_state.current_model_details = {
            "provider": selected_provider,
            "model_name": selected_model,
            "details": fetched_details if fetched_details else {}
        }
        for key in list(st.session_state.keys()):
            if key.startswith("details_"):
                del st.session_state[key]
        st.session_state[details_fetch_key] = True
elif "current_model_details" in st.session_state:
     del st.session_state.current_model_details
     for key in list(st.session_state.keys()):
         if key.startswith("details_"):
             del st.session_state[key]


# Optional: Clear chat button & Reset Tokens
def clear_chat_and_tokens():
    """Clears chat history and resets token counts in session state."""
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.session_state.last_input_tokens = 0
    st.session_state.last_output_tokens = 0
    st.session_state.total_input_tokens = 0
    st.session_state.total_output_tokens = 0
    logger.info("Chat history and token counts cleared.")

if st.sidebar.button("Clear Chat History", key="clear_chat"):
    clear_chat_and_tokens()
    st.rerun()

# --- Dev Bar Toggle ---
if "show_dev_bar" not in st.session_state:
    st.session_state.show_dev_bar = False

show_dev_bar = st.sidebar.toggle("Show Dev Bar", key="show_dev_bar")

# Removed Debug Logging Toggle section

# --- Display Backend URL ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Backend API: {BACKEND_URL}/docs")


# --- Main App Layout (with potential Dev Bar) ---
main_col, dev_bar_col = st.columns([0.7, 0.3])

with main_col:
    st.title("📄🤖 RAG Chat PoC")
    st.caption(f"A proof-of-concept chat agent using LLMs via a FastAPI backend ({BACKEND_URL}/docs).")

    # --- Chat Interface Area ---
    st.header(f"Chat with Project: {selected_project if selected_project else 'N/A'}")

    if selected_project and selected_provider and selected_model:
        st.info(f"Using Model: **{selected_provider} / {selected_model}**")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_input_tokens" not in st.session_state:
            st.session_state.last_input_tokens = 0
        if "last_output_tokens" not in st.session_state:
            st.session_state.last_output_tokens = 0
        if "total_input_tokens" not in st.session_state:
            st.session_state.total_input_tokens = 0
        if "total_output_tokens" not in st.session_state:
            st.session_state.total_output_tokens = 0

        current_config_key = f"{selected_project}_{selected_provider}_{selected_model}"
        if "current_config" not in st.session_state:
             st.session_state.current_config = current_config_key

        if st.session_state.current_config != current_config_key:
            logger.info(f"Config changed from {st.session_state.current_config} to {current_config_key}. Clearing chat history and tokens.")
            clear_chat_and_tokens()
            # Removed clearing of project_base_tokens state
            st.session_state.current_config = current_config_key
            st.warning("Project or Model changed. Chat history and token counts cleared.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "model_used" in message:
                     st.caption(f"Model: {message['model_used']}")

        if prompt := st.chat_input(f"Ask '{selected_project}' using {selected_model}...", key="chat_input_box"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            logger.info(f"User query for project '{selected_project}' using '{selected_provider}/{selected_model}': {prompt}")
            with st.spinner(f"Asking {selected_model}..."):
                api_response = call_chat_api(
                    project_name=selected_project,
                    query=prompt,
                    provider=selected_provider,
                    model_name=selected_model
                )

            if api_response and "response" in api_response:
                assistant_response_content = api_response["response"]
                sources = api_response.get("sources_consulted", [])
                model_used = api_response.get("model_used", f"{selected_provider}/{selected_model}")
                input_tokens = api_response.get("input_tokens")
                output_tokens = api_response.get("output_tokens")

                st.session_state.last_input_tokens = input_tokens if input_tokens is not None else 0
                st.session_state.last_output_tokens = output_tokens if output_tokens is not None else 0
                st.session_state.total_input_tokens += st.session_state.last_input_tokens
                st.session_state.total_output_tokens += st.session_state.last_output_tokens
                logger.info(f"Token counts updated: Last In={st.session_state.last_input_tokens}, Last Out={st.session_state.last_output_tokens}, Total In={st.session_state.total_input_tokens}, Total Out={st.session_state.total_output_tokens}")

                if sources:
                    assistant_response_content += f"\n\n*Sources consulted: {', '.join(sources)}*"

                assistant_message = {
                    "role": "assistant",
                    "content": assistant_response_content,
                    "model_used": model_used
                }
                st.session_state.messages.append(assistant_message)

                with st.chat_message("assistant"):
                    st.markdown(assistant_message["content"])
                    st.caption(f"Model: {assistant_message['model_used']}")

            else:
                 logger.warning("No valid response received from backend API.")

    elif not selected_project:
        st.info("Select a project from the sidebar to start chatting.")
        clear_chat_and_tokens()
        if "current_config" in st.session_state: del st.session_state.current_config
        # Removed clearing of project_base_tokens state
    elif not available_models_data:
         st.warning("Model selection is unavailable. Cannot start chat.")
         clear_chat_and_tokens()
         if "current_config" in st.session_state: del st.session_state.current_config
         # Removed clearing of project_base_tokens state
    else: # Project selected but model/provider missing
         st.warning("Select a provider and model from the sidebar.")
         clear_chat_and_tokens()
         if "current_config" in st.session_state: del st.session_state.current_config
         # Removed clearing of project_base_tokens state

# --- Dev Bar Content (Conditional) ---
if st.session_state.show_dev_bar:
    with dev_bar_col:
        st.header("Dev Bar")
        st.subheader("Project Info")
        if selected_project:
             # Display doc count and total doc tokens side-by-side
             col1_proj, col2_proj = st.columns(2)
             with col1_proj:
                 st.metric(label="Documents", value=st.session_state.get("selected_project_doc_count", "N/A"))
             with col2_proj:
                 total_doc_tokens = st.session_state.get("total_document_tokens")
                 token_display_value = f"{total_doc_tokens:,}" if total_doc_tokens is not None else "N/A"
                 st.metric(label="Input Context (Est.)", value=token_display_value)
             st.caption(f"Project: '{selected_project}' (Tokens from filelist.csv)") # Add project name and clarification
        else:
             st.write("No project selected.")

        st.subheader("Token Usage (Estimated)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Last Input", value=f"{st.session_state.get('last_input_tokens', 0):,}")
            st.metric(label="Total Input", value=f"{st.session_state.get('total_input_tokens', 0):,}")
        with col2:
            st.metric(label="Last Output", value=f"{st.session_state.get('last_output_tokens', 0):,}")
            st.metric(label="Total Output", value=f"{st.session_state.get('total_output_tokens', 0):,}")

        st.divider()
        st.subheader("Selected Model Details")
        model_details = st.session_state.get("current_model_details")
        if model_details:
            st.json(model_details)
        elif selected_provider and selected_model:
             st.info("Model details might be loading or unavailable.")
        else:
             st.write("No model selected.")


        st.divider()
        st.subheader("Debug Info")
        # Always display session state when Dev Bar is shown
        st.caption("Full Session State:")
        st.json(st.session_state)
