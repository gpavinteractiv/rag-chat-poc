# /mnt/LAB/rag-chat-poc/frontend/app.py

import streamlit as st
import httpx # For making API calls to the backend
import logging
import os
from typing import List, Dict, Optional, Tuple # Added Tuple

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000") # Using IP address
PROJECT_LIST_ENDPOINT = f"{BACKEND_URL}/projects"
MODELS_ENDPOINT = f"{BACKEND_URL}/models" # New endpoint for models
CHAT_ENDPOINT = f"{BACKEND_URL}/chat" # Base endpoint, project name appended

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="RAG Chat PoC",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data(ttl=300) # Cache project list for 5 minutes
def get_projects() -> List[str]:
    """Fetches the list of projects from the backend API."""
    logger.info(f"Fetching project list from {PROJECT_LIST_ENDPOINT}")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(PROJECT_LIST_ENDPOINT)
            response.raise_for_status()
            projects_data = response.json()
            logger.info(f"Successfully fetched {len(projects_data)} projects.")
            project_names = [p.get("name") for p in projects_data if p.get("name")]
            return sorted(project_names) # Sort alphabetically
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
            models_data = response.json() # Expected: {"providers": [{"provider": "...", "models": [...]}]}
            if "providers" in models_data:
                for provider_info in models_data["providers"]:
                    provider = provider_info.get("provider")
                    models = provider_info.get("models")
                    if provider and models:
                        models_dict[provider] = sorted(models) # Sort models alphabetically
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

def call_chat_api(project_name: str, query: str, provider: str, model_name: str) -> Optional[Dict]:
    """Calls the backend chat API for a given project, query, provider, and model."""
    if not all([project_name, query, provider, model_name]):
        st.error("Missing required information (project, query, provider, or model) for API call.")
        return None

    api_url = f"{CHAT_ENDPOINT}/{project_name}"
    # Include provider and model_name in the payload
    payload = {
        "query": query,
        "provider": provider,
        "model_name": model_name
    }
    logger.info(f"Sending query to {api_url} using {provider}/{model_name}")

    try:
        with httpx.Client(timeout=120.0) as client: # 2 minutes timeout
            response = client.post(api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Received response from {api_url}")
            # Expected: {"response": "...", "sources_consulted": [...], "model_used": "..."}
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
st.caption(f"A proof-of-concept chat agent using LLMs via a FastAPI backend ({BACKEND_URL}).")

# --- Sidebar Setup ---
st.sidebar.header("Configuration")

# --- Project Selection ---
available_projects = get_projects()
if not available_projects:
    st.sidebar.warning("No projects found or backend unavailable.")
    selected_project = None
else:
    # Use session state to remember selection
    if "selected_project" not in st.session_state or st.session_state.selected_project not in available_projects:
        st.session_state.selected_project = available_projects[0] # Default to first project

    selected_project = st.sidebar.selectbox(
        "Choose a project:",
        options=available_projects,
        key="selected_project", # Use session state key directly
    )

# --- Model Selection ---
available_models_data = get_available_models()
selected_provider = None
selected_model = None

if not available_models_data:
    st.sidebar.warning("No models found or backend unavailable.")
else:
    available_providers = list(available_models_data.keys())

    # Initialize session state for provider if it doesn't exist or is invalid
    if "selected_provider" not in st.session_state or st.session_state.selected_provider not in available_providers:
        st.session_state.selected_provider = available_providers[0] if available_providers else None

    selected_provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        options=available_providers,
        key="selected_provider",
    )

    # Dynamically update model options based on provider
    if selected_provider:
        models_for_provider = available_models_data.get(selected_provider, [])
        if not models_for_provider:
             st.sidebar.error(f"No models listed for provider '{selected_provider}'.")
             selected_model = None
        else:
            # Initialize or update selected model based on provider change or if invalid
            if "selected_model" not in st.session_state or st.session_state.selected_model not in models_for_provider:
                 st.session_state.selected_model = models_for_provider[0] # Default to first model of provider

            selected_model = st.sidebar.selectbox(
                f"Choose Model ({selected_provider}):",
                options=models_for_provider,
                key="selected_model",
            )
    else:
        selected_model = None # No provider selected

# --- Chat Interface Area ---
st.header(f"Chat with Project: {selected_project if selected_project else 'N/A'}")

if selected_project and selected_provider and selected_model:
    st.info(f"Using Model: **{selected_provider} / {selected_model}**")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Store current project/model config to detect changes
    current_config_key = f"{selected_project}_{selected_provider}_{selected_model}"
    if "current_config" not in st.session_state:
         st.session_state.current_config = current_config_key

    # Clear chat history if project or model config changes
    if st.session_state.current_config != current_config_key:
        logger.info(f"Config changed from {st.session_state.current_config} to {current_config_key}. Clearing chat history.")
        st.session_state.messages = []
        st.session_state.current_config = current_config_key
        st.warning("Project or Model changed. Chat history cleared.") # Inform user

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display model used for assistant messages if available
            if message["role"] == "assistant" and "model_used" in message:
                 st.caption(f"Model: {message['model_used']}")

    # Chat input field
    if prompt := st.chat_input(f"Ask '{selected_project}' using {selected_model}...", key="chat_input_box"):
        # 1. Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call the backend API with selected provider and model
        logger.info(f"User query for project '{selected_project}' using '{selected_provider}/{selected_model}': {prompt}")
        with st.spinner(f"Asking {selected_model}..."):
            api_response = call_chat_api(
                project_name=selected_project,
                query=prompt,
                provider=selected_provider,
                model_name=selected_model
            )

        # 3. Add assistant response
        if api_response and "response" in api_response:
            assistant_response_content = api_response["response"]
            sources = api_response.get("sources_consulted", [])
            model_used = api_response.get("model_used", f"{selected_provider}/{selected_model}") # Fallback

            # Append sources to the content for display
            if sources:
                assistant_response_content += f"\n\n*Sources consulted: {', '.join(sources)}*"

            # Store message with model info
            assistant_message = {
                "role": "assistant",
                "content": assistant_response_content,
                "model_used": model_used
            }
            st.session_state.messages.append(assistant_message)

            # Display message
            with st.chat_message("assistant"):
                st.markdown(assistant_message["content"])
                st.caption(f"Model: {assistant_message['model_used']}") # Display model used

        else:
             logger.warning("No valid response received from backend API.")
             # Error is displayed by call_chat_api via st.error


    # Optional: Clear chat button
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
         if "messages" in st.session_state:
              st.session_state.messages = []
              logger.info("Chat history cleared by user.")
              st.rerun()


elif not selected_project:
    st.info("Select a project from the sidebar to start chatting.")
    if "messages" in st.session_state: del st.session_state.messages
    if "current_config" in st.session_state: del st.session_state.current_config
elif not available_models_data:
     st.warning("Model selection is unavailable. Cannot start chat.")
     if "messages" in st.session_state: del st.session_state.messages
     if "current_config" in st.session_state: del st.session_state.current_config
else: # Project selected but model/provider missing
     st.warning("Select a provider and model from the sidebar.")
     if "messages" in st.session_state: del st.session_state.messages
     if "current_config" in st.session_state: del st.session_state.current_config


# --- Display Backend URL ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Backend API: {BACKEND_URL}")
