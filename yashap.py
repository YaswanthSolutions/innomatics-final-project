import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import logging
import os

# Static Google API Key (replace with your actual key)
GOOGLE_API_KEY = "your_api_key"

# Initialize models through LangChain with correct model names
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)


# Error handling function
def handle_error(error):
    logging.error(error)
    st.error(f"Error: {str(error)}")


# Scene understanding function
def scene_understanding(image):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        # Create the prompt and message structure for the image description
        prompt = """Describe this image for visually impaired individuals, including:
            1. Scene layout
            2. Main objects and their positions
            3. People and their activities (if any)
            4. Colors and lighting
            5. Notable features or points of interest"""

        # The correct format for the message with role and content
        message = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"}
        ]

        response = vision_llm.invoke(message)
        return response.content
    except Exception as e:
        handle_error(e)
        return "Error generating description."


# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"


# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text, lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        return f"Error generating speech: {str(e)}"


# Streamlit configuration
st.set_page_config(page_title="AI Visual Assistant", page_icon="🎨", layout="wide")

# Custom CSS for card-based UI with gradient shades
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #f8f9fa, #e9ecef, #dee2e6);
            font-family: 'Arial', sans-serif;
        }
        .card {
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .card-blue { background: linear-gradient(to right, #0077b6, #00b4d8); }
        .card-green { background: linear-gradient(to right, #2a9d8f, #43aa8b); }
        .header {
            text-align: center;
            color: #1d3557;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            color: #6c757d;
            font-size: 14px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='header'>AI Visual Assistant</h1>", unsafe_allow_html=True)

# -------------------------- Image Description Section --------------------------
st.markdown("<div class='card card-blue'>", unsafe_allow_html=True)
st.markdown("<h3>Image Description</h3>", unsafe_allow_html=True)
st.write(
    "Upload an image, and the app will generate a description of the scene, including actions, emotions, and visual elements.")

uploaded_file = st.file_uploader("Upload an image for description...", type=["jpg", "jpeg", "png"],
                                 label_visibility="visible")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating description..."):
        description = scene_understanding(image)
        st.subheader("Generated Description:")
        st.write(description)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- OCR and Text-to-Speech Section --------------------------
st.markdown("<div class='card card-green'>", unsafe_allow_html=True)
st.markdown("<h3>OCR and Text-to-Speech</h3>", unsafe_allow_html=True)
st.write("Upload an image with text, and the app will extract the text and convert it to speech.")

ocr_uploaded_file = st.file_uploader("Upload an image with text...", type=["jpg", "jpeg", "png"],
                                     label_visibility="visible")

if ocr_uploaded_file:
    image = Image.open(ocr_uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting text..."):
        text = extract_text_from_image(image)

    if text:
        st.subheader("Extracted Text:")
        st.write(text)

        with st.spinner("Converting text to speech..."):
            audio_file = text_to_speech(text)

        if os.path.exists(audio_file):
            st.subheader("Audio Playback:")
            audio = open(audio_file, "rb")
            st.audio(audio, format="audio/mp3")
            audio.close()
            os.remove(audio_file)
    else:
        st.warning("No text found in the image. Please try another image with visible text.")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>🔍 Powered by Tesseract OCR, gTTS, and Google's Generative AI | Built with ❤ using Streamlit</div>",
    unsafe_allow_html=True)
