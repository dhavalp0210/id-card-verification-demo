import streamlit as st
import cv2
import face_recognition
import numpy as np
import base64
from groq import Groq
from PIL import Image
import tempfile
import os
import json
import re
from dotenv import load_dotenv

# Load API Key
load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# Encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

# Extract Aadhaar details using OCR API
def extract_aadhaar_details(image_path):
    try:
        client = Groq()
        base64_image = encode_image(image_path)

        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract the following details from this ID card image and return ONLY valid JSON.
                        Ensure your response is strictly formatted as:
                        ```json
                        {"name": "Full Name", "dob": "DD-MM-YYYY", "gender": "Male/Female", "aadhaar_number": "XXXX-XXXX-XXXX"}
                        ```
                        DO NOT include any explanations or extra text, only return valid JSON."""
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}  
                ]
            }],
            model="llama-3.2-11b-vision-preview",
            temperature=0,
        )

        response = chat_completion.choices[0].message.content.strip()
        json_text = re.search(r"\{.*\}", response, re.DOTALL)
        
        if json_text:
            try:
                return json.loads(json_text.group(0))
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON: {str(e)}")
                return None
        else:
            st.error("Failed to find JSON structure in the response.")
            return None
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Extract face from Aadhaar image
def extract_face(image, margin_top=100, margin_other=50):
    # Convert image to RGB and ensure it's a valid NumPy array
    rgb_image = np.array(image.convert("RGB"), dtype=np.uint8)  

    # Validate Image Format
    if rgb_image.dtype != np.uint8:
        raise ValueError("Image data must be 8-bit per channel.")

    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        st.warning("⚠️ No face detected. Try another image.")
        return None

    # Extract first detected face
    top, right, bottom, left = face_locations[0]
    
    # Add margins
    height, width, _ = rgb_image.shape
    top = max(0, top - margin_top)  
    left = max(0, left - margin_other)
    bottom = min(height, bottom + margin_other)
    right = min(width, right + margin_other)

    # Crop and return the face
    return image.crop((left, top, right, bottom))



# Face verification
def verify_identity(aadhaar_img, webcam_img):
    a_enc = face_recognition.face_encodings(np.array(aadhaar_img.convert("RGB")))
    w_enc = face_recognition.face_encodings(np.array(webcam_img.convert("RGB")))
    
    if not a_enc or not w_enc:
        return "❌ Not Recognizable"
    
    return "✅ Verified" if face_recognition.compare_faces([a_enc[0]], w_enc[0], tolerance=0.45)[0] else "❌ Not Recognizable"

# Streamlit UI
st.title("📜 Aadhaar Verification System")
col1, col2 = st.columns(2)

# Aadhaar Upload (Left Column)
with col1:
    st.subheader("📂 Upload Aadhaar Card")
    aadhaar_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if aadhaar_file:
        aadhaar_img = Image.open(aadhaar_file)
        st.image(aadhaar_img, caption="📌 Aadhaar Card", use_container_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            aadhaar_img.save(tmp.name)
            details = extract_aadhaar_details(tmp.name)
            aadhaar_face = extract_face(aadhaar_img)
        os.unlink(tmp.name)

        if details:
            st.subheader("Extracted Aadhaar Details")
            col1_1, col1_2 = st.columns([2, 1])
            
            with col1_1:
                st.markdown(f"**👤 Name:** {details.get('name', 'N/A')}")
                st.markdown(f"**📅 Date of Birth:** {details.get('dob', 'N/A')}")
                st.markdown(f"**⚧ Gender:** {details.get('gender', 'N/A')}")
                st.markdown(f"**🔢 Aadhaar Number:** {details.get('aadhaar_number', 'N/A')}")
            
            with col1_2:
                if aadhaar_face:
                    st.image(aadhaar_face, caption="🖼️ Face", width=150)
                else:
                    st.warning("⚠️ No face detected.")

# Webcam Capture & Verification (Right Column)
with col2:
    st.subheader("📸 Capture Image")
    webcam_file = st.camera_input("Take a Photo")

    if webcam_file and aadhaar_file:
        user_img = Image.open(webcam_file)
        st.image(user_img, caption="🧑 Captured Image", width=150)
        result = verify_identity(aadhaar_face, user_img)
        st.subheader(result)
