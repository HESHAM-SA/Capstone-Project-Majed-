import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
from keras import models
import os
import random
import keras
from keras import ops
from keras import layers
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
from ultralytics.utils.plotting import Annotator
from datetime import datetime
import pandas as pd

# RandomColorAffine
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = ops.shape(images)[0]
            brightness_scales = 1 + keras.random.uniform(
                (batch_size, 1, 1, 1),
                minval=-self.brightness,
                maxval=self.brightness,
                seed=self.seed_generator,
            )
            jitter_matrices = keras.random.uniform(
                (batch_size, 1, 3, 3), 
                minval=-self.jitter, 
                maxval=self.jitter,
                seed=self.seed_generator,
            )
            color_transforms = (
                ops.tile(ops.expand_dims(ops.eye(3), axis=0), (batch_size, 1, 1, 1))
                * brightness_scales
                + jitter_matrices
            )
            images = ops.clip(ops.matmul(images, color_transforms), 0, 1)
        return images

# Preprocess Video to Extract Frames
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps/fps)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames, fps

# Detect Faces in Frames using YOLO Model
def get_detection_crops(yolo_result, threshold=0.5, crop_size=(128, 128), numpy=False):
    img = Image.fromarray(yolo_result.orig_img[..., ::-1])
    confidences = yolo_result.boxes.conf.tolist()
    coordinates = yolo_result.boxes.xywh.tolist()
    crops = []
    for conf, (x, y, w, h) in zip(confidences, coordinates):
        if conf < threshold:
            continue
        half = max(w, h) / 2
        x_min = max(x - half, 0)
        y_min = max(y - half, 0)
        x_max = min(x + half, img.size[0])
        y_max = min(y + half, img.size[1])
        crop = img.crop((x_min, y_min, x_max, y_max)).resize(crop_size)
        padding = (crop_size[0] - crop.size[0], crop_size[1] - crop.size[1])
        if padding != (0, 0):
            crop = ImageOps.expand(crop, border=padding, fill='black')
        crops.append(crop)
    if numpy:
        if crops:
            crops = np.stack([np.asarray(crop) for crop in crops])
        else:
            crops = None
    return crops

# Load Fine-tuning Model
@st.cache_resource
def load_model():
    model = keras.models.load_model('Attendees_management_system/Attendees_management_system/finetuning_model_classifction.keras', custom_objects={"RandomColorAffine": RandomColorAffine})
    model.pop()  # Remove the last layer
    return model

finetuning_model = load_model()

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO('Attendees_management_system/Attendees_management_system/best.pt')

yolo_detect_model = load_yolo_model()

# Feature Extraction and Similarity Function
def extract_features(crops):
    crop_arrays = np.stack([np.asarray(crop) for crop in crops])
    return finetuning_model.predict(crop_arrays, verbose=False)

def similarity(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.dot(x, y)

# Load existing known faces
def load_known_faces(known_face_names, known_face_FVs):
    Students_folder = 'Students'
    if not os.path.exists(Students_folder):
        os.makedirs(Students_folder)
    for student_name in os.listdir(Students_folder):
        student_folder = os.path.join(Students_folder, student_name)
        if os.path.isdir(student_folder):
            known_face_names.append(student_name)
            face_vectors = []
            for image_file in os.listdir(student_folder):
                image_path = os.path.join(student_folder, image_file)
                image = Image.open(image_path)
                image_array = np.asarray(image)
                crop = np.expand_dims(image_array, axis=0)
                FV = finetuning_model.predict(crop, verbose=False)
                face_vectors.append(FV[0])
            known_face_FVs.append(face_vectors)

# Create Directories for Storing Images
os.makedirs('Students', exist_ok=True)
os.makedirs('uploaded_videos', exist_ok=True)

# Initialize lists for feature vectors and names
known_face_names = []
known_face_FVs = []

# Load existing known faces
load_known_faces(known_face_names, known_face_FVs)

# Function to generate a random 4-digit ID
def generate_4_digit_id():
    return f'{random.randint(1000, 9999)}'

# Attendance Tracking
attendance_file = 'attendance.xlsx'
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name/Date'])
    df.to_excel(attendance_file, index=False)

def mark_attendance(name):
    try:
        df = pd.read_excel(attendance_file, index_col=0)
        current_date = datetime.now().date().strftime("%Y-%m-%d")
        if current_date not in df.columns:
            df[current_date] = ""
        if name not in df.index:
            df.loc[name] = [""] * len(df.columns)
        df.at[name, current_date] = "P"
        df.to_excel(attendance_file)
    except Exception as e:
        st.error(f"Error in marking attendance: {e}")

# Streamlit App

st.markdown(
    """
    <div style="text-align: center;">
        <h1>نظام التحضير التلقائي</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar configuration
confidence_threshold = st.sidebar.slider('confidence Threshold', 0., 1., 0.5, step=0.05)
similarity_threshold = st.sidebar.slider('Similarity Threshold', 0., 1., 0.82, step=0.01)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
progress_bar = st.progress(0)

if uploaded_file:
    video_path = f"uploaded_videos/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    frames, fps = preprocess_video(video_path)
    total_frames = len(frames)
    output_frames = []
    for i, frame in enumerate(frames):
        yolo_result = yolo_detect_model(frame)[0]
        crops = get_detection_crops(yolo_result,confidence_threshold)
        if not crops:
            progress_bar.progress((i + 1) / total_frames)
            output_frames.append(frame)
            continue
        new_FVs = extract_features(crops)
        
        for crop, new_FV in zip(crops, new_FVs):
            max_similarity = 0
            best_match_index = -1
            for j, known_FV_list in enumerate(known_face_FVs):
                for known_FV in known_FV_list:
                    sim = similarity(new_FV, known_FV)
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match_index = j
            
            if max_similarity > 0.90:
                continue

            if max_similarity > similarity_threshold:
                student_name = known_face_names[best_match_index]
                student_folder = f'Students/{student_name}'
                os.makedirs(student_folder, exist_ok=True)
                if len(known_face_FVs[best_match_index]) < 20:
                    known_face_FVs[best_match_index].append(new_FV)
                    crop.save(os.path.join(student_folder, f'{generate_4_digit_id()}.png'))
            else:
                new_id = generate_4_digit_id()
                known_face_names.append(new_id)
                known_face_FVs.append([new_FV])
                student_name = new_id
                student_folder = f'Students/{new_id}'
                os.makedirs(student_folder, exist_ok=True)
                crop.save(os.path.join(student_folder, f'{generate_4_digit_id()}.png'))
        
        # Draw bounding boxes and names on the frame
        for box in yolo_result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            # Draw the name
            if max_similarity > similarity_threshold:
                student_name = known_face_names[best_match_index]
            else:
                student_name = new_id
            cv2.putText(frame, student_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        output_frames.append(frame)
        progress_bar.progress((i + 1) / total_frames)

    # Save the processed video
    output_video_path = f"processed_videos/processed_{uploaded_file.name}"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()

    progress_bar.progress(1.0)
    st.success('Video processing complete!')
    st.video(output_video_path)

# Real-Time Face Recognition
st.markdown('Start Attendees')

def plot_bounding_boxes(yolo_result, frame, known_face_names, known_face_FVs):
    img = yolo_result.orig_img[..., ::-1]
    annotator = Annotator(np.ascontiguousarray(img))
    boxes = yolo_result.boxes.xyxy
    crops = []
    
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        crop = Image.fromarray(img[y_min:y_max, x_min:x_max]).resize((128, 128))
        crops.append((crop, (x_min, y_min, x_max, y_max)))
    
    if crops:
        new_FVs = extract_features([crop[0] for crop in crops])
        for (crop, (x_min, y_min, x_max, y_max)), new_FV in zip(crops, new_FVs):
            max_similarity = 0
            best_match_index = -1
            for j, known_FV_list in enumerate(known_face_FVs):
                for known_FV in known_FV_list:
                    sim = similarity(new_FV, known_FV)
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match_index = j
            
            if max_similarity > similarity_threshold:
                student_name = known_face_names[best_match_index]
                annotator.box_label((x_min, y_min, x_max, y_max), student_name, color=(255, 0, 0))
                mark_attendance(student_name)
            else:
                annotator.box_label((x_min, y_min, x_max, y_max), '', color=(255, 0, 0))

    return Image.fromarray(annotator.result())

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_image()
    yolo_result = yolo_detect_model(img, conf=confidence_threshold)[0]
    result_img = plot_bounding_boxes(yolo_result, img, known_face_names, known_face_FVs)
    return av.VideoFrame.from_image(result_img)

def app():
    webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_frame_callback=video_frame_callback,
    )

if __name__ == "__main__":
    app()
