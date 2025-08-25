import streamlit as st
import cv2
import numpy as np
import re
from PIL import Image
import google.generativeai as genai
import tempfile
import easyocr
import os

# Cấu hình trang Streamlit
st.set_page_config(page_title="Nhận Diện Biển Số Xe", layout="wide")

# Regex chuẩn hóa biển số
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

# Tải EasyOCR
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'], gpu=False)  # Tắt GPU để tương thích với Render

# OCR với EasyOCR
def ocr_with_easyocr(image: np.ndarray) -> str:
    try:
        reader = load_easyocr()
        results = reader.readtext(image, detail=0)
        txt = " ".join(results)
        return normalize_plate(txt)
    except Exception as e:
        st.warning(f"Lỗi EasyOCR: {e}")
        return ""

# OCR với Gemini
def ocr_with_gemini(image: Image.Image, model) -> str:
    try:
        prompt = "Đọc biển số xe trong ảnh (chỉ trả về biển số, không thêm ký tự nào khác):"
        res = model.generate_content([prompt, image])
        return normalize_plate(res.text.strip())
    except Exception as e:
        st.warning(f"Lỗi Gemini: {e}")
        return ""

# Cấu hình Gemini API
GEMINI_API_KEY = st.sidebar.text_input("🔑 Nhập Gemini API Key (tùy chọn):", type="password")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        st.sidebar.success("Đã kết nối với Gemini API.")
    except Exception as e:
        st.sidebar.error(f"Lỗi kết nối Gemini: {e}")
        gemini_model = None
else:
    st.sidebar.info("Không có Gemini API Key, sẽ sử dụng EasyOCR.")

# Giao diện tối giản
st.title("🚗 Nhận Diện Biển Số Xe")
input_type = st.radio("Loại dữ liệu:", ["Ảnh", "Video"], horizontal=True)
uploaded_file = st.file_uploader("📤 Tải lên ảnh hoặc video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# Xem trước file tải lên
if uploaded_file:
    st.subheader("📎 Xem trước")
    if input_type == "Ảnh":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Ảnh đã tải lên")
        uploaded_file.seek(0)  # Reset con trỏ file để xử lý tiếp
    else:
        st.video(uploaded_file)
        uploaded_file.seek(0)  # Reset con trỏ file để xử lý tiếp

# Xử lý nhận diện
if uploaded_file and st.button("🚀 Xử lý"):
    plates = []

    if input_type == "Ảnh":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if gemini_model:
            text = ocr_with_gemini(pil_image, gemini_model)
        else:
            text = ocr_with_easyocr(image)
        if text:
            plates.append(text)
        else:
            st.warning("❌ Không phát hiện biển số.")

    else:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_video.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 0.5)  # Xử lý mỗi 0.5 giây
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = st.progress(0.0)
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if gemini_model:
                    text = ocr_with_gemini(pil_frame, gemini_model)
                else:
                    text = ocr_with_easyocr(frame)
                if text:
                    plates.append(text)
            idx += 1
            pbar.progress(min(idx / frames, 1.0))
        cap.release()
        os.unlink(temp_video.name)

    # Hiển thị kết quả
    st.subheader("📋 Kết quả nhận diện")
    if plates:
        for plate in set(plates):
            st.write(f"- **{plate}** ({plates.count(plate)} lần)")
    else:
        st.warning("❌ Không phát hiện biển số.")
