import streamlit as st
import pyttsx3
from PIL import Image
import pytesseract
from gtts import gTTS
from transformers import AutoProcessor, BlipForConditionalGeneration
import google.generativeai as genai
import os

# Set up Google Gemini API key (replace 'YOUR_API_KEY' with actual key)
GENAI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"
genai.configure(api_key=GENAI_API_KEY)

# Initialize BLIP model for image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Function to generate image caption using BLIP
def generate_image_caption(image):
    image = image.convert('RGB')
    inputs = processor(images=image, text="Describe the image", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

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
st.write("Upload an image, and the app will generate a description of the scene, including actions, emotions, and visual elements.")

uploaded_file = st.file_uploader("Upload an image for description...", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating description..."):
        caption = generate_image_caption(image)
        st.subheader("Generated Caption:")
        st.write(caption)

    # Function to generate scene description using Google Gemini API
    def generate_scene_description_with_gemini(caption):
        try:
            prompt = f"Generate an emotionally rich and action-based description of the following scene: {caption}"
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            ai_assistant = model.start_chat(history=[])
            response = ai_assistant.send_message(prompt)
            return response.text.strip() if response and response.text else "No description generated."
        except Exception as e:
            return f"Error generating description: {str(e)}"

    description = generate_scene_description_with_gemini(caption)
    if "Error" in description:
        st.error(description)
    else:
        st.subheader("Generated Description:")
        st.write(description)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------- OCR and Text-to-Speech Section --------------------------
st.markdown("<div class='card card-green'>", unsafe_allow_html=True)
st.markdown("<h3>OCR and Text-to-Speech</h3>", unsafe_allow_html=True)
st.write("Upload an image with text, and the app will extract the text and convert it to speech.")

ocr_uploaded_file = st.file_uploader("Upload an image with text...", type=["jpg", "jpeg", "png"], label_visibility="visible")

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
st.markdown("<div class='footer'>🔍 Powered by Tesseract OCR, gTTS, and Google's Generative AI | Built with ❤ using Streamlit</div>", unsafe_allow_html=True)
