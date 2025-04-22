import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🛩️ YOLOv5 + Optical Flow | Drone Forensic App")

# Load YOLOv5s using ultralytics
@st.cache_resource
def load_model():
    return YOLO('yolov5s.pt')  # pretrained model included in package

model = load_model()

# Inference
def infer_image(img_path):
    results = model(img_path)
    img_array = results[0].plot()  # returns BGR image
    return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

# Optical Flow
def draw_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    if p0 is None:
        return next_img
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None)
    for i, (a, b) in enumerate(zip(p0, p1)):
        if status[i]:
            x0, y0 = a.ravel()
            x1, y1 = b.ravel()
            cv2.arrowedLine(next_img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2, tipLength=0.3)
    return next_img

# Upload
st.subheader("Upload 2 images (consecutive frames)")
uploaded = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded and len(uploaded) == 2:
    paths = []
    for file in uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file.read())
            paths.append(tmp.name)

    col1, col2 = st.columns(2)
    with col1:
        st.image(paths[0], caption="Frame 1", use_column_width=True)
        st.image(infer_image(paths[0]), caption="YOLOv5 Prediction", use_column_width=True)
    with col2:
        st.image(paths[1], caption="Frame 2", use_column_width=True)
        st.image(infer_image(paths[1]), caption="YOLOv5 Prediction", use_column_width=True)

    # Optical flow overlay
    st.subheader("🔄 Optical Flow")
    frame1 = cv2.imread(paths[0])
    frame2 = cv2.imread(paths[1])
    flow = draw_optical_flow(frame1, frame2)
    st.image(cv2.cvtColor(flow, cv2.COLOR_BGR2RGB), caption="Optical Flow Result", use_column_width=True)

elif uploaded:
    st.warning("Please upload exactly **2 images**.")
