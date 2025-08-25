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

# ================== Regex chuẩn hóa biển số ==================
FALLBACK_REGEX = re.compile(r"(\d{2})\s*[- ]?\s*([A-Z]{1,2})\s*[- ]?\s*([0-9]{2,6})")

def normalize_plate(text: str) -> str:
    if not text:
        return ""
    text = text.upper().strip()
    # Loại bỏ các ký tự lạ trừ A-Z 0-9 và khoảng/-
    text = re.sub(r"[^A-Z0-9\- ]+", "", text)
    text = text.replace(" ", "").replace("-", "")
    m = re.match(r"^(\d{2})([A-Z]{1,2})(\d{2,6})$", text)
    if m:
        p1, p2, p3 = m.groups()
        return f"{p1}-{p2} {p3}"
    m2 = FALLBACK_REGEX.search(text)
    if m2:
        p1, p2, p3 = m2.groups()
        return f"{p1}-{p2} {p3}"
    return text

# ================== EasyOCR (cache) ==================
@st.cache_resource
def load_easyocr():
    # Tắt GPU để tương thích Render/Colab nhẹ
    return easyocr.Reader(['en'], gpu=False)

def ocr_with_easyocr(image_bgr: np.ndarray) -> str:
    try:
        reader = load_easyocr()
        # EasyOCR kỳ vọng RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = reader.readtext(image_rgb, detail=0)
        txt = " ".join(results).strip()
        return normalize_plate(txt)
    except Exception as e:
        st.warning(f"Lỗi EasyOCR: {e}")
        return ""

# ================== Gemini ==================
def ocr_with_gemini(image_pil: Image.Image, model) -> str:
    try:
        prompt = "Đọc biển số xe trong ảnh (chỉ trả về biển số, không thêm ký tự nào khác):"
        res = model.generate_content([prompt, image_pil])
        return normalize_plate((res.text or "").strip())
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

# ================== Giao diện ==================
st.title("🚗 Nhận Diện Biển Số Xe")
input_type = st.radio("Loại dữ liệu:", ["Ảnh", "Video"], horizontal=True)
uploaded_file = st.file_uploader("📤 Tải lên ảnh hoặc video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# Xem trước file tải lên
if uploaded_file:
    st.subheader("📎 Xem trước")
    if input_type == "Ảnh":
        # Đọc 1 lần để preview
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Không đọc được ảnh. Vui lòng thử lại.")
        else:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh đã tải lên", use_container_width=True)
        uploaded_file.seek(0)  # Reset con trỏ để xử lý tiếp
    else:
        st.video(uploaded_file)
        uploaded_file.seek(0)

# ================== Xử lý nhận diện ==================
if uploaded_file and st.button("🚀 Xử lý"):
    status_placeholder = st.empty()
    status_placeholder.info("Đang xử lý bằng Gemini API..." if gemini_model else "Đang xử lý bằng EasyOCR...")

    plates = []

    if input_type == "Ảnh":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Không đọc được ảnh để xử lý.")
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = ocr_with_gemini(pil_image, gemini_model) if gemini_model else ocr_with_easyocr(image)
            if text:
                plates.append(text)
            else:
                st.warning("❌ Không phát hiện biển số.")

    else:  # Video
        # Lưu tạm video
        suffix = os.path.splitext(uploaded_file.name or "")[-1].lower() or ".mp4"
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_video.write(uploaded_file.read())
        temp_video.flush()
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        if not cap.isOpened():
            st.error("Không mở được video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25  # fallback an toàn
            frame_interval = max(1, int(fps * 0.5))  # xử lý mỗi 0.5 giây

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            pbar = st.progress(0.0)
            idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    text = ocr_with_gemini(pil_frame, gemini_model) if gemini_model else ocr_with_easyocr(frame)
                    if text:
                        plates.append(text)
                idx += 1
                if total_frames > 0:
                    pbar.progress(min(idx / total_frames, 1.0))
            cap.release()
        # Xoá file tạm
        try:
            os.unlink(temp_video.name)
        except Exception:
            pass

    status_placeholder.empty()

    # ================== GHI CHÚ (footer cố định) ==================
    st.markdown(
        """
        <div style='position: fixed; bottom: 10px; right: 10px; font-size: 14px; color: gray;'>
            By Võ Công Nhật-20222627.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================== Hiển thị kết quả ==================
    st.subheader("📋 Kết quả nhận diện")
    if plates:
        unique = sorted(set(plates))
        for plate in unique:
            st.write(f"- **{plate}** ({plates.count(plate)} lần)")
    else:
        st.warning("❌ Không phát hiện biển số.")
