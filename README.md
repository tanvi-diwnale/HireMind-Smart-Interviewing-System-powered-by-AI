# HireMind-Smart-Interviewing-System-powered-by-AI
HireMind is an AI-driven interview assistant that helps automate and streamline the candidate interview process. It supports voice-based and text-based interviews, evaluates responses in real time, and generates a professional interview summary report.
✅ Voice & Text Interview Modes – Choose between speaking or typing answers
✅ Resume Parsing – Upload a candidate’s PDF resume for context-aware questioning
✅ AI-Generated or Custom Questions – Dynamically generated questions or use your own question set
✅ Text-to-Speech (TTS) – AI recruiter speaks questions using Gemini
✅ Speech-to-Text (STT) – Converts candidate responses to text with Whisper
✅ Interview Report – AI summarizes candidate performance with strengths & improvement areas
✅ Downloadable PDF – Generate a professional report at the end of the interview

🏗️ Tech Stack

Frontend: Streamlit

AI & NLP: Google Gemini (gemini-2.5-flash-preview-05-20), Whisper (Speech-to-Text)

Libraries: PyPDF2, FPDF, requests, audio_recorder_streamlit, soundfile, torch

Backend: Python (session state management, API calls)

Deployment: Streamlit Cloud or local execution

⚙️ Installation & Setup

Clone the Repository

git clone https://github.com/tanvi-diwnale/HireMind-AI-Interviewer.git
cd HireMind-AI-Interviewer

Create and Activate Virtual Environment (recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install Dependencies

pip install -r requirements.txt


Set Environment Variables
Create a .env file in the root folder and add your Google Gemini API key:

GOOGLE_API_KEY=your_google_api_key_here


Run the App

streamlit run app.py
