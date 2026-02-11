import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def compute_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark

    def point(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    left_eye = point(33)
    right_eye = point(263)
    nose_tip = point(1)
    chin = point(152)
    mouth_left = point(61)
    mouth_right = point(291)
    left_face = point(234)
    right_face = point(454)
    forehead = point(10)

    face_width = np.linalg.norm(left_face - right_face)
    face_height = np.linalg.norm(forehead - chin)

    eye_slant = (right_eye[1] - left_eye[1]) / face_width
    intercanthal_ratio = np.linalg.norm(left_eye - right_eye) / face_width
    nose_width_ratio = np.linalg.norm(point(98) - point(326)) / face_width
    mouth_width_ratio = np.linalg.norm(mouth_left - mouth_right) / face_width
    chin_ratio = np.linalg.norm(nose_tip - chin) / face_height
    roundness = face_width / face_height
    symmetry = abs(left_eye[1] - right_eye[1]) / face_height
    midface_ratio = np.linalg.norm(forehead - nose_tip) / face_height
    jaw_width_ratio = face_width / face_height
    eye_width_ratio = intercanthal_ratio
    nose_height_ratio = np.linalg.norm(nose_tip - forehead) / face_height
    lower_face_ratio = np.linalg.norm(nose_tip - chin) / face_height

    return [
        eye_slant,
        intercanthal_ratio,
        nose_width_ratio,
        mouth_width_ratio,
        chin_ratio,
        roundness,
        symmetry,
        midface_ratio,
        jaw_width_ratio,
        eye_width_ratio,
        nose_height_ratio,
        lower_face_ratio
    ]

dataset = []
labels = []

for label in ["DownSyndrome", "Normal"]:
    folder = f"dataset/{label}"
    for img_file in os.listdir(folder):
        path = os.path.join(folder, img_file)
        features = compute_features(path)
        if features is not None:
            dataset.append(features)
            labels.append(1 if label == "DownSyndrome" else 0)

columns = [
    "EyeSlant",
    "IntercanthalRatio",
    "NoseWidthRatio",
    "MouthWidthRatio",
    "ChinRatio",
    "FaceRoundness",
    "Symmetry",
    "MidfaceRatio",
    "JawWidthRatio",
    "EyeWidthRatio",
    "NoseHeightRatio",
    "LowerFaceRatio"
]

df = pd.DataFrame(dataset, columns=columns)
df["Label"] = labels

df.to_csv("down_syndrome_12_features.csv", index=False)

print("12-feature dataset created successfully.")
