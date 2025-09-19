import streamlit as st
import io
import PyPDF2
from google.generativeai import GenerativeModel
import time
from fpdf import FPDF
import os
import google.generativeai as gemini
import requests
import base64
import json
import wave
import tempfile
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
import whisper
import torch

# Set up the page configuration
st.set_page_config(page_title="AI Interviewer", layout="wide")

# Check for GPU and set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configure API keys from environment variables ---
# Using Google API Key from environment variable
gemini.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Initialize session state variables ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "interview_mode" not in st.session_state:
    st.session_state.interview_mode = "Voice-based"
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "questions_list" not in st.session_state:
    st.session_state.questions_list = None
if "question_index" not in st.session_state:
    st.session_state.question_index = 0

# --- LLM with retry logic ---
def get_gemini_response(prompt, history, model_name="gemini-2.5-flash-preview-05-20", max_retries=5):
    """
    Calls the Gemini API with a retry mechanism for rate limiting.
    """
    model = GenerativeModel(model_name)
    retries = 0
    while retries < max_retries:
        try:
            full_history = [
                {"role": "user" if item["role"] == "user" else "model", "parts": [item["content"]]}
                for item in history
            ]
            full_history.append({"role": "user", "parts": [prompt]})
            
            response = model.generate_content(full_history)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "too many requests" in str(e).lower():
                retries += 1
                delay = (2 ** retries) + (0.5 * retries)
                st.warning(f"Rate limit exceeded (429). Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                st.error(f"An unexpected error occurred: {e}")
                return None
    st.error(f"Failed to get response after {max_retries} retries.")
    return None

# --- Voice Mode Functions (TTS and STT) ---
def gemini_tts(text):
    """
    Generates and plays speech from text using Gemini TTS API.
    """
    if not text:
        return

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": "Kore" }
                }
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={os.getenv('GOOGLE_API_KEY')}",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        audio_data_base64 = result.get('candidates')[0].get('content').get('parts')[0].get('inlineData').get('data')
        pcm_data = base64.b64decode(audio_data_base64)
        
        # Convert PCM to WAV format in-memory using BytesIO
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_data)
        
        buffer.seek(0)
        st.audio(buffer.read(), format='audio/wav')

    except Exception as e:
        st.error(f"Error generating audio: {e}")

@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model and caches it."""
    # Using 'base.en' for English only, which is faster. Use 'base' for multiple languages.
    return whisper.load_model("base.en", device=DEVICE)

def whisper_transcribe(audio_file_path):
    """
    Transcribes audio from a file using the local Whisper model.
    """
    model = load_whisper_model()
    try:
        result = model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

# --- PDF handling functions ---
def read_pdf(file):
    """Reads a PDF file and extracts text."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_pdf(content, filename):
    """Generates a PDF file from a string and returns it as a downloadable file."""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'AI Interview Report', 0, 1, 'C')
            self.ln(10)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(content)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes, filename

# --- Streamlit UI ---
st.title("AI Interviewer")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    uploaded_questions_file = st.file_uploader("Upload Your Own Questions (TXT, PDF)", type=["txt", "pdf"])
    
    interview_mode = st.radio(
        "Select Interview Mode:",
        ("Voice-based", "Text-based"),
        key="interview_mode"
    )

    start_button = st.button("Start Interview", use_container_width=True)
    if start_button:
        if uploaded_file is None:
            st.warning("Please upload a resume to begin.")
        else:
            with st.spinner("Reading resume..."):
                st.session_state.resume_text = read_pdf(uploaded_file)
            
            st.session_state.interview_started = True
            st.session_state.chat_history = []
            st.session_state.question_index = 0 # Reset question index

            if uploaded_questions_file:
                # Read questions based on file type
                file_extension = os.path.splitext(uploaded_questions_file.name)[1].lower()
                if file_extension == ".pdf":
                    questions_text = read_pdf(uploaded_questions_file)
                else: # Assumes .txt
                    questions_text = uploaded_questions_file.getvalue().decode("utf-8")
                
                # Split questions by newline and filter out empty strings to handle different formats
                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                
                if questions:
                    st.session_state.questions_list = questions
                    first_question = st.session_state.questions_list[0]
                    st.session_state.question_index += 1
                    initial_prompt = f"You are an AI recruiter. You are a professional, but friendly interviewer. Your first question is: '{first_question}'"
                else:
                    st.warning("The uploaded file does not contain any questions.")
                    st.session_state.questions_list = None
                    initial_prompt = f"""
                    You are an AI recruiter. You are a professional, but friendly interviewer.
                    The candidate has provided the following resume.
                    
                    RESUME:
                    {st.session_state.resume_text}
                    
                    Based on the resume, start the interview by asking a single, relevant question.
                    """
            else:
                st.session_state.questions_list = None
                initial_prompt = f"""
                You are an AI recruiter. You are a professional, but friendly interviewer.
                The candidate has provided the following resume.
                
                RESUME:
                {st.session_state.resume_text}
                
                Based on the resume, start the interview by asking a single, relevant question.
                """
            
            with st.spinner("Preparing the first question..."):
                first_question_to_ask = get_gemini_response(initial_prompt, [])
            
            if first_question_to_ask:
                st.session_state.chat_history.append({"role": "ai", "content": first_question_to_ask})
                if st.session_state.interview_mode == "Voice-based":
                    with st.spinner("The AI is preparing to speak..."):
                        gemini_tts(first_question_to_ask)
                st.rerun()

    st.markdown("---")

    if st.session_state.interview_started:
        if st.button("End Interview", use_container_width=True):
            report_prompt = f"""
            The following is a transcript of an interview. Please act as a recruiter and provide a professional, one-paragraph summary of the candidate's performance, followed by a bulleted list of their key strengths and a separate bulleted list of areas for improvement.
            
            Interview Transcript:
            {st.session_state.chat_history}
            """
            with st.spinner("Generating report..."):
                report = get_gemini_response(report_prompt, st.session_state.chat_history)
            
            if report:
                st.session_state.chat_history.append({"role": "report", "content": report})
                st.rerun()

with col2:
    st.header("Transcript")
    
    chat_container = st.container()
    
    for message in st.session_state.chat_history:
        if message["role"] == "ai":
            with chat_container.chat_message("AI Recruiter", avatar="💼"):
                st.markdown(message["content"])
                
        elif message["role"] == "user":
            with chat_container.chat_message("You", avatar="👤"):
                st.markdown(message["content"])
        elif message["role"] == "report":
            st.markdown("---")
            st.header("Interview Report")
            st.markdown(message["content"])
            
            # Button to download the report
            report_bytes, filename = generate_pdf(message["content"], "AI_Interview_Report.pdf")
            st.download_button(
                label="Download Report as PDF",
                data=report_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )

    if st.session_state.interview_started and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] != "report":
        st.markdown("---")
        
        if st.session_state.interview_mode == "Voice-based":
            st.subheader("Voice Mode")
            st.info("The microphone is listening. Click to record, then click again to stop.")
            
            # Use a key to manage the audio recorder's state
            st.session_state.audio_bytes = audio_recorder(
                text="Click to record/stop",
                recording_color="#e33719",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                key="audio_recorder"
            )

            # Check if new audio has been recorded and is not empty
            if st.session_state.audio_bytes and len(st.session_state.audio_bytes) > 0:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(st.session_state.audio_bytes)
                    audio_path = tmp_file.name

                with st.spinner("Transcribing your answer with local Whisper..."):
                    transcription = whisper_transcribe(audio_path)
                
                os.remove(audio_path)
                
                if transcription:
                    st.session_state.chat_history.append({"role": "user", "content": transcription})

                    # Logic to check if there are more pre-defined questions
                    if st.session_state.questions_list and st.session_state.question_index < len(st.session_state.questions_list):
                        next_question_from_list = st.session_state.questions_list[st.session_state.question_index]
                        next_prompt = f"""
                        The candidate just said: "{transcription}". Based on their previous answers and the provided resume, the next question is: '{next_question_from_list}'
                        RESUME:
                        {st.session_state.resume_text}
                        """
                        st.session_state.question_index += 1
                    else:
                        next_prompt = f"""
                        The candidate just said: "{transcription}". Based on their previous answers and the provided resume, please ask the next logical interview question. Keep your questions concise.
                        RESUME:
                        {st.session_state.resume_text}
                        """
                    
                    with st.spinner("Generating next question..."):
                        next_question = get_gemini_response(next_prompt, st.session_state.chat_history)

                    if next_question:
                        st.session_state.chat_history.append({"role": "ai", "content": next_question})
                        gemini_tts(next_question)
                        st.rerun()
                
                # After processing, reset the audio bytes in session state
                st.session_state.audio_bytes = None
            elif st.session_state.audio_bytes is not None and len(st.session_state.audio_bytes) == 0:
                # Handle empty recording
                st.warning("No audio detected. Please ensure your microphone is enabled and try recording again.")

        else: # Text-based mode
            st.subheader("Text Mode")
            user_input = st.text_input("Your response:", key="user_input_text")
            
            if st.button("Send", use_container_width=True) and user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Logic to check if there are more pre-defined questions
                if st.session_state.questions_list and st.session_state.question_index < len(st.session_state.questions_list):
                    next_question_from_list = st.session_state.questions_list[st.session_state.question_index]
                    next_prompt = f"""
                    The candidate just said: "{user_input}". Based on their previous answers and the provided resume, the next question is: '{next_question_from_list}'
                    RESUME:
                    {st.session_state.resume_text}
                    """
                    st.session_state.question_index += 1
                else:
                    next_prompt = f"""
                    The candidate just said: "{user_input}". Based on their previous answers and the provided resume, please ask the next logical interview question. Keep your questions concise.
                    RESUME:
                    {st.session_state.resume_text}
                    """
                
                with st.spinner("Generating next question..."):
                    next_question = get_gemini_response(next_prompt, st.session_state.chat_history)
                
                if next_question:
                    st.session_state.chat_history.append({"role": "ai", "content": next_question})
                    st.rerun()