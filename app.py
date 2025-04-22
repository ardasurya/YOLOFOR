import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import tempfile

st.set_page_config(layout="wide")
st.title("🛩️ YOLOv5 + Optical Flow | Drone Forensic")

# Load YOLOv5s model from torch.hub (online)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# Inference function using YOLOv5
def infer_image(img_path):
    results = model(img_path)
    results.render()
    return Image.fromarray(results.ims[0])

# Optical Flow Drawing Function
def draw_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    if p0 is None:
        return next_img

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

    for i, (pt1, pt2) in enumerate(zip(p0, p1)):
        if status[i]:
            x0, y0 = pt1.ravel()
            x1, y1 = pt2.ravel()
            cv2.arrowedLine(next_img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2, tipLength=0.3)

    return next_img

# Upload Images
st.subheader("Upload two consecutive images (e.g., drone frames):")
uploaded_imgs = st.file_uploader("Upload exactly 2 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_imgs and len(uploaded_imgs) == 2:
    temp_paths = []
    for img in uploaded_imgs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img.read())
            temp_paths.append(tmp.name)

    col1, col2 = st.columns(2)
    with col1:
        st.image(temp_paths[0], caption="Frame 1", use_column_width=True)
        result1 = infer_image(temp_paths[0])
        st.image(result1, caption="YOLOv5 Detection 1", use_column_width=True)

    with col2:
        st.image(temp_paths[1], caption="Frame 2", use_column_width=True)
        result2 = infer_image(temp_paths[1])
        st.image(result2, caption="YOLOv5 Detection 2", use_column_width=True)

    # Optical Flow
    st.subheader("🔄 Optical Flow Visualization")
    frame1 = cv2.imread(temp_paths[0])
    frame2 = cv2.imread(temp_paths[1])
    flow_output = draw_optical_flow(frame1, frame2)
    st.image(cv2.cvtColor(flow_output, cv2.COLOR_BGR2RGB), caption="Optical Flow Overlay", use_column_width=True)

elif uploaded_imgs:
    st.warning("Please upload **exactly two images**.", icon="⚠️")
