import streamlit as st
import requests
import time

# Set page config
st.set_page_config(
    page_title="ACE Advisory Legal Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean interface
st.markdown("""
    <style>
    /* Main container */
    .stApp {
        background-color: #ffffff;
        min-height: 100vh;
        position: relative;
        padding-bottom: 100px; /* Space for fixed input */
    }
    
    /* Fixed title container */
    .fixed-title {
        position: fixed;
        top: 20px; /* Lowered from top */
        left: 0;
        right: 0;
        background-color: #ffffff;
        padding: 1.5rem;
        z-index: 999;
        text-align: center;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Title and description */
    h1 {
        color: #111827 !important;
        margin: 0 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    .description {
        color: #6b7280 !important;
        margin: 0.5rem 0 !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    
    /* Chat container */
    .chat-container {
        margin-top: 140px; /* Increased space for fixed title */
        padding: 1rem;
        max-width: 800px;
        margin: 140px auto 100px; /* Space for fixed title and input */
        max-height: calc(100vh - 300px); /* Limit height to viewport minus fixed elements */
        overflow-y: auto; /* Make chat area scrollable */
    }
    
    /* User message */
    .user-message {
        background-color: #f3f4f6;
        color: #111827;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Bot message */
    .bot-message {
        background-color: #ffffff;
        color: #111827;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        max-width: 80%;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Fixed input container */
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #ffffff;
        padding: 1rem;
        z-index: 1000;
        border-top: 1px solid #e5e7eb;
        box-shadow: 0 -1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Input box */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 12px 15px !important;
        font-size: 16px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Send button */
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Clear chat button */
    .clear-chat {
        background-color: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Ensure content is visible */
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
    
    /* Remove default Streamlit padding */
    .stApp > div:first-child {
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Fixed title container
st.markdown("""
    <div class="fixed-title">
        <h1>⚖️ ACE Advisory Legal Chatbot</h1>
        <div class="description">
            Ask away—legal help is just a chat away!
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to send query to API
def send_query(query):
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"text": query}
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="bot-message">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fixed input container
st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("Enter your legal question:", key="query_input", placeholder="Type your question here...", label_visibility="collapsed")
with col2:
    if st.button("Send", key="send_button"):
        if query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Get bot response
            with st.spinner("Thinking..."):
                response = send_query(query)
                st.session_state.messages.append({"role": "bot", "content": response})
            
            # Rerun to clear input
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Clear chat button
if st.button("Clear Chat", key="clear_chat"):
    st.session_state.messages = []
    st.rerun()