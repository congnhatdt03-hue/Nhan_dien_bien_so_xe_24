import streamlit as st
import cv2
import numpy as np
import re
from PIL import Image
import google.generativeai as genai
import tempfile

# Streamlit page config
st.set_page_config(page_title="License Plate Recognition", layout="wide")

# Gemini API configuration
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key:", type="password")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    st.error("Please enter a valid Gemini API Key.")
    st.stop()

# Regex for normalizing license plate
FALLBACK_REGEX = re.compile(r"(\d{2})\s*[- ]?\s*([A-Z]{1,2})\s*[- ]?\s*([0-9]{2,6})")
def normalize_plate(text: str) -> str:
    text = text.upper().replace(" ", "").replace("-", "")
    match = re.match(r"^(\d{2})([A-Z]{1,2})(\d{2,6})$", text)
    if match:
        p1, p2, p3 = match.groups()
        return f"{p1}-{p2} {p3}"
    match2 = FALLBACK_REGEX.search(text)
    if match2:
        p1, p2, p3 = match2.groups()
        return f"{p1}-{p2} {p3}"
    return text

# OCR with Gemini
def ocr_with_gemini(image: Image.Image) -> str:
    try:
        prompt = "Extract the license plate number from the image (return only the license plate number, no extra characters):"
        res = gemini_model.generate_content([prompt, image])
        return normalize_plate(res.text.strip())
    except Exception as e:
        return ""

# Minimalist UI
st.title("License Plate Recognition")
input_type = st.radio("Input Type:", ["Image", "Video"], horizontal=True)
uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file and st.button("Process"):
    plates = []

    if input_type == "Image":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        text = ocr_with_gemini(pil_image)
        if text:
            plates.append(text)
            st.image(image, channels="BGR", caption="Processed Image")
        else:
            st.warning("No license plate detected.")

    else:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_video.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 0.5)  # Process every 0.5 seconds
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = st.progress(0.0)
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                text = ocr_with_gemini(pil_frame)
                if text:
                    plates.append(text)
            idx += 1
            pbar.progress(min(idx / frames, 1.0))
        cap.release()
        os.unlink(temp_video.name)
        st.video(uploaded_file)

    # Display results
    if plates:
        st.subheader("Detected License Plates:")
        for plate in set(plates):
            st.write(f"- {plate} ({plates.count(plate)} times)")
    else:
        st.warning("No license plates detected.")
