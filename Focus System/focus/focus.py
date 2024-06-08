import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import math
import streamlit as st

# Constants
FONTS = cv.FONT_HERSHEY_SIMPLEX
CENTER_THRESHOLD = 2  # Time threshold for looking at the center
SIDE_THRESHOLD = 1.5    # Time threshold for looking to the side
BLINK_THRESHOLD = 5   # Time threshold for closing eyes
DISCOUNT_CENTER = 1  # Percentage discount for not looking at center
DISCOUNT_SIDE = 1    # Percentage discount for looking to the side
DISCOUNT_EYES = 20    # Percentage discount for closing eyes

# Eye and iris landmark indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


mp_face_mesh = mp.solutions.face_mesh


def euclidean_distance(point, point1):
    return math.sqrt((point1[0] - point[0])**2 + (point1[1] - point[1])**2)


def blink_ratio(landmarks, right_indices, left_indices):
    rh_distance = euclidean_distance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    rv_distance = euclidean_distance(landmarks[right_indices[12]], landmarks[right_indices[4]])
    lh_distance = euclidean_distance(landmarks[left_indices[0]], landmarks[left_indices[8]])
    lv_distance = euclidean_distance(landmarks[left_indices[12]], landmarks[left_indices[4]])

    if rv_distance == 0 or lv_distance == 0:
        return float('inf')

    re_ratio = rh_distance / rv_distance
    le_ratio = lh_distance / lv_distance
    return (re_ratio + le_ratio) / 2


def landmarks_detection(img, results):
    img_height, img_width = img.shape[:2]
    return [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]


def eye_direction(eye_points, iris_center, ratio):
    eye_left = np.min(eye_points[:, 0])
    eye_right = np.max(eye_points[:, 0])

    hor_range = eye_right - eye_left

    iris_x, iris_y = iris_center

    if ratio > 5.5:
        return "Blink"
    elif iris_x < eye_left + hor_range * 0.3:
        return "Left"
    elif iris_x > eye_right - hor_range * 0.3:
        return "Right"
    else:
        return "Center"

# processing function
def process_frame(frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time):
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_points = landmarks_detection(frame, results)
        ratio = blink_ratio(mesh_points, RIGHT_EYE, LEFT_EYE)

        # Iris detection and direction
        left_iris_points = np.array([mesh_points[i] for i in LEFT_IRIS], dtype=np.int32)
        right_iris_points = np.array([mesh_points[i] for i in RIGHT_IRIS], dtype=np.int32)

        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        left_eye_direction = eye_direction(np.array([mesh_points[p] for p in LEFT_EYE]), center_left, ratio)
        right_eye_direction = eye_direction(np.array([mesh_points[p] for p in RIGHT_EYE]), center_right, ratio)

        if left_eye_direction == right_eye_direction:
            eye_direction_text = left_eye_direction
        else:
            if left_eye_direction in ["Left", "Right"]:
                eye_direction_text = left_eye_direction
            elif right_eye_direction in ["Left", "Right"]:
                eye_direction_text = right_eye_direction
            else:
                eye_direction_text = left_eye_direction

        if ratio > 5.5:
            if not blink_detected:
                blink_start_time = time.time()  
                blink_detected = True
        else:
            if blink_detected:
                blink_duration = time.time() - blink_start_time
                total_blink_duration += blink_duration
                blink_detected = False

        current_time = time.time()
        if eye_direction_text == "Center":
            if last_look_centered_time is None:
                last_look_centered_time = current_time
            not_looking_start_time = None
            if current_time - last_look_centered_time >= CENTER_THRESHOLD:
                focus_score = 100
                total_blink_duration = 0
        else:
            last_look_centered_time = None
            if not not_looking_start_time:
                not_looking_start_time = current_time
            elif current_time - not_looking_start_time >= SIDE_THRESHOLD:
                focus_score -= DISCOUNT_SIDE
                not_looking_start_time = None 

        if eyes_closed_start_time is not None and current_time - eyes_closed_start_time >= BLINK_THRESHOLD:
            focus_score -= DISCOUNT_EYES  

        focus_score = max(0, min(focus_score, 100))

        # Draw landmarks and circles around iris
        # cv.polylines(frame, [np.array([mesh_points[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
        # cv.polylines(frame, [np.array([mesh_points[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
        # cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
        # cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

        cv.putText(frame, f"Eyes: {eye_direction_text}", (50, 100), FONTS, 1, (110, 240, 23), 2, cv.LINE_AA)
        cv.putText(frame, f"Focus Score: {focus_score}%", (50, 150), FONTS, 1, (110, 240, 23), 2, cv.LINE_AA)

    else:
        focus_score = 0  

    return frame, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time

# Streamlit
def main():
    st.title("Focus Detection")
    st.markdown(
        """
        <style>
        .reportview-container {
            background: skyblue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Webcam Feed")

    col1, col2 = st.columns([1, 2])
    with col1:
        start_button = st.button("Start")
        stop_button = st.button("Stop")

    placeholder = col2.empty()
    progress_placeholder = st.empty()
    focus_score = 100

    if start_button:
        cap = cv.VideoCapture(0)
        last_look_centered_time = None
        not_looking_start_time = None
        blink_start_time = None
        total_blink_duration = 0
        blink_detected = False
        eyes_closed_start_time = None

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as face_mesh:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time = process_frame(
                    frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time
                )

                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                placeholder.image(frame, channels="RGB")
                progress_placeholder.progress(focus_score / 100.0)
                st.text(f"Focus Score: {focus_score}%")

                if stop_button:
                    break

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
