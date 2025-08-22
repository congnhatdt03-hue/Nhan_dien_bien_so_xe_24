# Before running this code, install required packages in Colab:
# !pip install streamlit google-generative-ai pillow opencv-python

import streamlit as st
import google.generativeai as genai
from PIL import Image
import io, os, re, json, time
import cv2
import numpy as np
from typing import List

# ====== Regex & chuẩn hoá ======
FALLBACK_REGEX = re.compile(
    r"(\d{2})\s*[- ]?\s*([A-Z]{1,2})\s*[- ]?\s*(\d{2,6})"
)

def normalize_plate(raw: str) -> str:
    if not raw:
        return ""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    m = re.fullmatch(r"(\d{2})([A-Z]{1,2})(\d{2,6})", s)
    if not m:
        return ""
    p, series, num = m.groups()
    if len(num) == 5:
        return f"{p}-{series} {num[:2]}.{num[2:]}"
    elif len(num) == 6:
        return f"{p}-{series} {num[:3]}.{num[3:]}"
    return ""

def dedupe_preserve_order(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

class GeminiModel:
    def __init__(self, api_key: str, model_name="gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_text_from_image(self, image_bytes: bytes) -> List[str]:
        prompt = (
            "Extract ALL Vietnamese vehicle license plates in the image. "
            'Return ONLY a JSON array of uppercase strings. Example: ["37-M156.341","29-AY005.540"]. Do not add explanations.'
        )
        try:
            img = Image.open(io.BytesIO(image_bytes))
            resp = self.model.generate_content([prompt, img])
            txt = (resp.text or "").strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
                txt = re.sub(r"\n?```$", "", txt).strip()
            plates = []
            try:
                data = json.loads(txt)
                if isinstance(data, list):
                    plates = [normalize_plate(str(x)) for x in data]
                    plates = [p for p in plates if re.fullmatch(r"\d{2}-[A-Z]{1,2}\s+\d{2,3}\.\d{2,3}", p)]
            except Exception:
                pass
            if not plates:
                U = txt.upper()
                cand = [normalize_plate("".join(m)) for m in FALLBACK_REGEX.findall(U)]
                plates = [p for p in cand if re.fullmatch(r"\d{2}-[A-Z]{1,2}\s+\d{2,3}\.\d{2,3}", p)]
            return dedupe_preserve_order(plates)
        except Exception:
            return []

def process_video(video_bytes: bytes) -> List[str]:
    # Convert video bytes to OpenCV format
    video_file = io.BytesIO(video_bytes)
    cap = cv2.VideoCapture(cv2.CAP_FFMPEG)
    cap.open(video_file)
    
    plates = set()
    frame_count = 0
    max_frames = 30  # Process only 30 frames to limit API calls
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to reduce processing time
        frame = cv2.resize(frame, (640, 480))
        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_byte_arr = io.BytesIO()
        Image.fromarray(frame_rgb).save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        # Extract plates from this frame
        frame_plates = gemini.extract_text_from_image(img_bytes)
        plates.update(frame_plates)
        frame_count += 1
    
    cap.release()
    return list(plates)

def main():
    st.set_page_config(page_title="Nhận Diện Biển Số Xe", page_icon="🔎")
    st.title("Nhận Diện Biển Số Xe")
    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("GEMINI_API_KEY", type="password")
    if not api_key:
        st.info("Nhập GEMINI_API_KEY.")
        return
    try:
        global gemini
        gemini = GeminiModel(api_key)
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {e}")
        return
    
    input_type = st.radio("Chọn loại đầu vào:", ["Ảnh", "Video"])
    
    if input_type == "Ảnh":
        up = st.file_uploader("Chọn ảnh (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"])
        if up:
            img_bytes = up.read()
            st.image(img_bytes, caption="Ảnh đầu vào", use_column_width=True)
            if st.button("Nhận Diện Biển Số"):
                with st.spinner("Đang xử lý..."):
                    plates = gemini.extract_text_from_image(img_bytes)
                if plates:
                    st.success("Kết quả nhận diện:")
                    for p in plates:
                        st.write(p)
                    st.download_button("Tải xuống kết quả", "\n".join(plates),
                                       file_name="plates.txt", mime="text/plain")
                else:
                    st.error("Không nhận diện được biển số phù hợp.")
    
    elif input_type == "Video":
        up = st.file_uploader("Chọn video (MP4)", type=["mp4"])
        if up:
            video_bytes = up.read()
            st.video(video_bytes, caption="Video đầu vào")
            if st.button("Nhận Diện Biển Số"):
                with st.spinner("Đang xử lý..."):
                    plates = process_video(video_bytes)
                if plates:
                    st.success("Kết quả nhận diện:")
                    for p in plates:
                        st.write(p)
                    st.download_button("Tải xuống kết quả", "\n".join(plates),
                                       file_name="plates.txt", mime="text/plain")
                else:
                    st.error("Không nhận diện được biển số phù hợp.")
    
    st.markdown(
        """
        <div style='position: fixed; bottom: 10px; right: 10px; font-size: 15px; color: gray;'>
            Đồ án II- GVHD: ThS. Nguyễn Thị Huế
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
