
import streamlit as st
import cv2
import numpy as np
import re
import os
from PIL import Image
from ultralytics import YOLO
import easyocr
import tempfile
from collections import Counter


st.set_page_config(page_title="Nhận diện biển số xe", layout="wide")

# ====== CONFIG ======
YOLO_PATH = "best.pt"
FRAME_STEP = 200  # ms
CONFIDENCE_THRESHOLD = 0.25
USE_EASYOCR_FALLBACK = True

# ====== GEMINI API KEY ======
import google.generativeai as genai
GEMINI_API_KEY = st.sidebar.text_input("🔑 Nhập Gemini API Key:", type="password")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

# ====== LOAD YOLO MODEL ======
@st.cache_resource
def load_yolo(path):
    model = YOLO(path)
    return model
model = load_yolo(YOLO_PATH)

# ====== REGEX CHUẨN HÓA ======
FALLBACK_REGEX = re.compile(r"(\d{2})\s*[- ]?\s*([A-Z]{1,2})\s*[- ]?\s*([0-9]{2,6})")
def normalize_plate(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    match = re.match(r"^(\d{2})([A-Z]{1,2})(\d{2,6})$", text)
    if match:
        p1, p2, p3 = match.groups()
        return f"{p1}-{p2} {p3}"
    match2 = FALLBACK_REGEX.search(text)
    if match2:
        p1, p2, p3 = match2.groups()
        return f"{p1}-{p2} {p3}"
    return text

# ====== OCR ======
def ocr_with_gemini(image: Image.Image) -> str:
    try:
        prompt = "Đọc biển số xe trong ảnh (chỉ trả lời biển số, không thêm ký tự nào khác):"
        res = gemini_model.generate_content([prompt, image])
        txt = res.text.strip()
        return normalize_plate(txt)
    except Exception as e:
        st.warning(f"Lỗi Gemini: {e}")
        return ""

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'])

def ocr_with_easyocr(image: np.ndarray) -> str:
    reader = load_easyocr()
    results = reader.readtext(image)
    txt = " ".join([r[1] for r in results])
    return normalize_plate(txt)

# ====== TIỀN XỬ LÝ ROI ======
def enhance_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.5, cv2.GaussianBlur(clahe, (0,0), 1), -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

# ====== HIỂN THỊ GIAO DIỆN ======
st.title("🚗 Nhận diện biển số xe từ ảnh hoặc video")
st.markdown("<div style='text-align:center; font-size:18px; color:gray;'>Đồ án II – GVHD: ThS. Nguyễn Thị Huế</div>", unsafe_allow_html=True)

input_type = st.radio("Chọn loại dữ liệu:", ["Ảnh", "Video"])
uploaded_file = st.file_uploader("📤 Tải lên ảnh hoặc video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file and st.button("🚀 Bắt đầu nhận diện"):
    plate_counter = Counter()

    if input_type == "Ảnh":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = model.predict(image, conf=CONFIDENCE_THRESHOLD)[0]
        annotated = image.copy()

        if len(result.boxes) == 0:
            st.warning("❌ Không có biển số được phát hiện.")
        else:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = annotated[y1:y2, x1:x2]
                roi = enhance_roi(roi)

                if gemini_model:
                    pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    text = ocr_with_gemini(pil_roi)
                else:
                    text = ocr_with_easyocr(roi) if USE_EASYOCR_FALLBACK else ""

                plate_counter[text] += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            st.image(annotated, caption="Ảnh đã nhận diện", channels="BGR", use_column_width=True)

    else:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * FRAME_STEP / 1000)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = "output_annotated.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        idx = 0
        pbar = st.progress(0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        box_found = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                result = model.predict(frame, conf=CONFIDENCE_THRESHOLD)[0]
                if len(result.boxes) > 0:
                    box_found = True
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]
                    roi = enhance_roi(roi)

                    text = ocr_with_easyocr(roi) if USE_EASYOCR_FALLBACK else ""

                    plate_counter[text] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            out.write(frame)
            idx += 1
            pbar.progress(min(idx / frames, 1.0))
        cap.release()
        out.release()

        if not box_found:
            st.warning("❌ Không có biển số được phát hiện trong video.")
        else:
            st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("📥 Tải video đã nhận diện", f, file_name="output_annotated.mp4")

    st.subheader("📋 Biển số phát hiện được:")
    found_plate = False
    for plate, count in plate_counter.items():
        if plate.strip():
            st.write(f"- **{plate}** ({count} lần)")
            found_plate = True
    if not found_plate:
        st.warning("❌ Không có biển số được phát hiện.")

# ===== CLEANUP =====
if 'temp_video' in locals() and os.path.exists(temp_video.name):
    os.unlink(temp_video.name)
if 'out_path' in locals() and os.path.exists(out_path):
    os.unlink(out_path)
