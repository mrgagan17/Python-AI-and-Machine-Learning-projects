import streamlit as st
from bot_logic import get_bot_response

# 1. Page Configuration
st.set_page_config(page_title="AI Support Chatbot", page_icon="💬", layout="centered")

# 2. Custom CSS for UI
st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    /* Style the chat input area */
    .stChatInputContainer {
        background-color: #1A1A1E !important;
        border: 1px solid #3C4043 !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Support Assistant. How can I help you with your order, refunds, or account today?"}
    ]

# 4. UI Layout
st.title("💬 AI Support Chatbot")
st.markdown("A rule-based natural language processing agent powered by NLTK Pattern Matching.")
st.markdown("---")

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User Input Handling
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response from NLTK logic
    bot_reply = get_bot_response(prompt)
    
    # Add bot response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)