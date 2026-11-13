import streamlit as st
import tempfile
import os
from agent_orchestrator import PrivacyMultiAgentRAG

st.set_page_config(page_title="Multi-Agent RAG", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 { background: -webkit-linear-gradient(45deg, #10B981, #3B82F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .agent-box { background-color: #1A1A1E; padding: 15px; border-radius: 8px; border-left: 4px solid #3B82F6; margin-bottom: 10px; font-family: monospace;}
    .final-box { background-color: #1A1A1E; padding: 20px; border-radius: 8px; border: 1px solid #10B981; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_orchestrator():
    return PrivacyMultiAgentRAG()

rag_system = get_orchestrator()

st.title("🛡️ Privacy-First Multi-Agent RAG System")
st.markdown("An autonomous swarm of local AI agents (Researcher, Synthesizer, Critic) collaborating to analyze your private documents with zero cloud leakage.")
st.markdown("---")

# Sidebar for Enterprise Document Ingestion
with st.sidebar:
    st.header("📄 Secure Ingestion")
    uploaded_file = st.file_uploader("Upload Private PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Process & Embed Document"):
            with st.spinner("Chunking and Embedding locally via MiniLM..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                chunks = rag_system.ingest_document(tmp_path)
                os.unlink(tmp_path)
                st.success(f"Document embedded into {chunks} vector chunks in local ChromaDB!")

# Main Chat Interface
query = st.text_input("Ask a question about your uploaded document:")

if st.button("Launch Autonomous Agents ✦") and query:
    st.markdown("### 🧠 Live Agent Orchestration Log")
    log_placeholder = st.empty()
    logs = []
    
    final_answer = ""
    
    # Run the multi-agent generator
    for step in rag_system.run_orchestration(query):
        if step["agent"] == "Final":
            final_answer = step["content"]
        else:
            logs.append(f"**[{step['agent']}]** {step['status']}")
            
            # Update the UI dynamically
            log_ui = "".join([f"<div class='agent-box'>{l}</div>" for l in logs])
            log_placeholder.markdown(log_ui, unsafe_allow_html=True)
            
    if final_answer:
        st.markdown("### ✨ Verified Final Output")
        st.markdown(f"<div class='final-box'>{final_answer}</div>", unsafe_allow_html=True)