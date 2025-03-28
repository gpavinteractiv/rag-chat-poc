# /mnt/LAB/gemini-chat-poc/frontend/app.py

import streamlit as st
import httpx # For making API calls to the backend
import logging
import os

# --- Configuration ---
# Get the backend API URL. Default to localhost if not set via env var.
# This allows flexibility for containerized environments later.
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
PROJECT_LIST_ENDPOINT = f"{BACKEND_URL}/projects"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat" # Base endpoint, project name will be appended

# Set up basic logging for the frontend app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Gemini Chat PoC",
    page_icon="🤖",
    layout="wide" # Use wide layout for chat potentially
)

# --- Helper Functions ---

# Cache the project list to avoid fetching it on every interaction
# TTL can be adjusted based on how often projects might change
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_projects():
    """Fetches the list of projects from the backend API."""
    logger.info(f"Fetching project list from {PROJECT_LIST_ENDPOINT}")
    try:
        with httpx.Client() as client: # Using sync client for simplicity here
            response = client.get(PROJECT_LIST_ENDPOINT, timeout=10.0) # Add timeout
            response.raise_for_status() # Raise exception for 4xx/5xx errors
            projects_data = response.json()
            logger.info(f"Successfully fetched {len(projects_data)} projects.")
            # Extract just the names for the dropdown
            project_names = [p.get("name") for p in projects_data if p.get("name")]
            return project_names
    except httpx.RequestError as e:
        logger.error(f"HTTP error fetching projects: {e}", exc_info=True)
        st.error(f"Failed to connect to backend API at {BACKEND_URL}. Is it running? Error: {e}")
        return [] # Return empty list on error
    except Exception as e:
        logger.error(f"Error fetching or parsing projects: {e}", exc_info=True)
        st.error(f"An error occurred while fetching projects: {e}")
        return []

# --- Main App UI ---

st.title("📄🤖 Gemini Large Context Chat PoC")
st.caption("A proof-of-concept chat agent using Gemini 2.5 Pro via a FastAPI backend.")

# --- Project Selection ---
st.sidebar.header("Project Selection")
available_projects = get_projects()

if not available_projects:
    st.sidebar.warning("No projects found or backend unavailable.")
    selected_project = None
else:
    selected_project = st.sidebar.selectbox(
        "Choose a project:",
        options=available_projects,
        index=0, # Default to the first project
        key="project_selector" # Assign a key for potential state management
    )
    st.sidebar.info(f"Selected project: **{selected_project}**")

# --- Chat Interface Area ---
st.header(f"Chat with Project: {selected_project if selected_project else 'N/A'}")

if selected_project:
    # Placeholder for chat history display
    st.write("Chat history will appear here.")

    # Placeholder for user input
    user_query = st.text_input("Your query:", key="user_query_input", placeholder="Ask something about the project documents...")

    if st.button("Send Query", key="send_button"):
        if user_query:
            st.write(f"Sending query: '{user_query}' (Functionality not yet implemented)")
            # TODO: Implement API call to backend /chat/{selected_project}
            pass
        else:
            st.warning("Please enter a query.")
else:
    st.info("Select a project from the sidebar to start chatting.")

# --- Display Backend URL (for debugging) ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Backend API: {BACKEND_URL}")
