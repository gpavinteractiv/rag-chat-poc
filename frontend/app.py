# /mnt/LAB/gemini-chat-poc/frontend/app.py

import streamlit as st
import httpx # For making API calls to the backend
import logging
import os
from typing import List, Dict, Optional # Added for type hinting

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000") # Using IP address
PROJECT_LIST_ENDPOINT = f"{BACKEND_URL}/projects"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat" # Base endpoint, project name appended

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Gemini Chat PoC",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data(ttl=300) # Cache project list for 5 minutes
def get_projects() -> List[str]:
    """Fetches the list of projects from the backend API."""
    logger.info(f"Fetching project list from {PROJECT_LIST_ENDPOINT}")
    try:
        # Use a timeout to prevent indefinite hangs (Source: httpx docs)
        with httpx.Client(timeout=10.0) as client:
            response = client.get(PROJECT_LIST_ENDPOINT)
            response.raise_for_status() # Check for HTTP errors
            projects_data = response.json()
            logger.info(f"Successfully fetched {len(projects_data)} projects.")
            project_names = [p.get("name") for p in projects_data if p.get("name")]
            return project_names
    except httpx.RequestError as e:
        logger.error(f"HTTP error fetching projects: {e}", exc_info=True)
        st.error(f"Failed to connect to backend API ({BACKEND_URL}). Is it running? Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching or parsing projects: {e}", exc_info=True)
        st.error(f"An error occurred while fetching projects: {e}")
        return []

def call_chat_api(project_name: str, query: str) -> Optional[Dict]:
    """Calls the backend chat API for a given project and query."""
    if not project_name:
        st.error("No project selected.")
        return None

    api_url = f"{CHAT_ENDPOINT}/{project_name}"
    payload = {"query": query}
    logger.info(f"Sending query to {api_url}")

    try:
        # Using sync client for simplicity in Streamlit's flow
        # Increase timeout for potentially long LLM responses
        with httpx.Client(timeout=120.0) as client: # 2 minutes timeout
            response = client.post(api_url, json=payload)
            response.raise_for_status() # Raise HTTP errors
            response_data = response.json()
            logger.info(f"Received response from {api_url}")
            return response_data # Expected: {"response": "...", "sources_consulted": [...]}
    except httpx.HTTPStatusError as e:
        # Handle specific HTTP errors from FastAPI backend
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

st.title("📄🤖 Gemini Large Context Chat PoC")
st.caption(f"A proof-of-concept chat agent using Gemini 2.5 Pro via a FastAPI backend ({BACKEND_URL}).")

# --- Project Selection ---
st.sidebar.header("Project Selection")
available_projects = get_projects()

if not available_projects:
    st.sidebar.warning("No projects found or backend unavailable.")
    selected_project = None
    # Clear chat if projects disappear? Optional.
    # if "messages" in st.session_state:
    #     del st.session_state.messages
else:
    # Use session state to remember the selection across reruns if needed,
    # or just rely on selectbox's default behavior.
    selected_project = st.sidebar.selectbox(
        "Choose a project:",
        options=available_projects,
        index=0, # Default to the first project
        key="project_selector" # Key helps maintain state
    )
    st.sidebar.info(f"Selected project: **{selected_project}**")

# --- Chat Interface Area ---
st.header(f"Chat with Project: {selected_project if selected_project else 'N/A'}")

if selected_project:
    # Initialize chat history in session state if it doesn't exist (Source: [2], [3], [6])
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Also store selected project in session state to detect changes
    if "current_project" not in st.session_state:
         st.session_state.current_project = selected_project

    # Clear chat history if project changes
    if st.session_state.current_project != selected_project:
        logger.info(f"Project changed from {st.session_state.current_project} to {selected_project}. Clearing chat history.")
        st.session_state.messages = []
        st.session_state.current_project = selected_project
        st.warning("Project changed. Chat history cleared.") # Inform user

    # Display existing chat messages (Source: [8], [9], [10])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # "user" or "assistant"
            st.markdown(message["content"])

    # Chat input field (Source: [8])
    # Using st.chat_input which is designed for this purpose
    if prompt := st.chat_input(f"Ask something about '{selected_project}'...", key="chat_input_box"):
        # 1. Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call the backend API
        logger.info(f"User query for project '{selected_project}': {prompt}")
        with st.spinner("Thinking..."): # Show loading indicator
            api_response = call_chat_api(selected_project, prompt)

        # 3. Add assistant response to history and display it
        if api_response and "response" in api_response:
            assistant_response = api_response["response"]
            # Optionally include consulted sources info
            sources = api_response.get("sources_consulted", [])
            if sources:
                assistant_response += f"\n\n*Sources consulted: {', '.join(sources)}*"

            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            # Error handling is done within call_chat_api and displayed via st.error
            # We could add a placeholder message in the chat if needed
            # st.session_state.messages.append({"role": "assistant", "content": "[Error receiving response. See error message above.]"})
            # with st.chat_message("assistant"):
            #      st.error("[Error receiving response. See error message above.]")
             logger.warning("No valid response received from backend API.")


    # Optional: Add a button to clear chat history for the current project
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
         if "messages" in st.session_state:
              st.session_state.messages = []
              logger.info("Chat history cleared by user.")
              st.rerun() # Force rerun to update the display


else: # No project selected
    st.info("Select a project from the sidebar to start chatting.")
    # Clear history if no project is selected
    if "messages" in st.session_state:
        del st.session_state.messages
    if "current_project" in st.session_state:
         del st.session_state.current_project


# --- Display Backend URL (for debugging) ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Backend API: {BACKEND_URL}")
