import streamlit as st
import whisper
import pyttsx3
import os
import tempfile
import ollama
from audio_recorder_streamlit import audio_recorder

# 1. Page Configuration & Dark Mode UI
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #FF4B4B, #9B72CB, #4285F4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .chat-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; margin-top: 10px; margin-bottom: 10px; }
    .user-text { color: #4285F4; font-weight: bold; }
    .ai-text { color: #10B981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 2. Initialize Local Offline Models
@st.cache_resource
def load_whisper_model():
    # Using 'base' for a good balance of speed and accuracy on CPU
    return whisper.load_model("base")

whisper_model = load_whisper_model()

def generate_tts_audio(text, output_path):
    """Generates offline Text-to-Speech audio using the native OS engine."""
    engine = pyttsx3.init()
    # Optional: Configure voice properties (speed, volume)
    engine.setProperty('rate', 170)
    engine.save_to_file(text, output_path)
    engine.runAndWait()

# 3. Dashboard Layout
st.title("🎙️ Offline AI Voice Assistant")
st.markdown("A 100% privacy-first, locally hosted voice agent. It uses **Whisper** for transcription, **Ollama** for intelligence, and native **TTS** for voice generation without any cloud APIs.")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<div style='text-align: center;'>Click the microphone to speak</div>", unsafe_allow_html=True)
    # Audio recorder widget
    audio_bytes = audio_recorder(text="", icon_size="2x", neutral_color="#E2E2E2", recording_color="#FF4B4B")

if audio_bytes:
    with st.spinner("Processing audio pipeline..."):
        try:
            # Step 1: Save audio bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                audio_path = temp_audio.name

            # Step 2: Speech-to-Text (Whisper)
            st.markdown("*Transcribing with Whisper...*")
            transcription = whisper_model.transcribe(audio_path)
            user_text = transcription["text"].strip()
            
            # Step 3: LLM Reasoning (Ollama - ensure you have llama3 or mistral pulled locally)
            st.markdown("*Generating response via Ollama...*")
            response = ollama.chat(model='qwen2:0.5b', messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful, concise, and highly intelligent AI voice assistant. Keep answers brief and conversational, as they will be spoken aloud.'
                },
                {
                    'role': 'user',
                    'content': user_text
                }
            ])
            ai_response_text = response['message']['content']

            # Step 4: Text-to-Speech (pyttsx3)
            st.markdown("*Synthesizing voice...*")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_tts:
                tts_path = temp_tts.name
            
            generate_tts_audio(ai_response_text, tts_path)

            # Display the conversation in the UI
            st.markdown(f"""
            <div class="chat-box">
                <p><span class="user-text">You:</span> {user_text}</p>
                <p><span class="ai-text">AI:</span> {ai_response_text}</p>
            </div>
            """, unsafe_allow_html=True)

            # Play the generated audio response
            st.audio(tts_path, format="audio/wav", autoplay=True)

            # Cleanup temp files
            os.unlink(audio_path)
            # os.unlink(tts_path) # Kept alive for streamlit to play

        except Exception as e:
            st.error(f"Error in processing pipeline: {e}")