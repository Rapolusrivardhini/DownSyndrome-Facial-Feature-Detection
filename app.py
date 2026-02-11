import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train model
data = pd.read_csv("down_syndrome_12_features.csv")
X = data.drop("Label", axis=1)
y = data["Label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def compute_features(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

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

    features = [
        (right_eye[1] - left_eye[1]) / face_width,
        np.linalg.norm(left_eye - right_eye) / face_width,
        np.linalg.norm(point(98) - point(326)) / face_width,
        np.linalg.norm(mouth_left - mouth_right) / face_width,
        np.linalg.norm(nose_tip - chin) / face_height,
        face_width / face_height,
        abs(left_eye[1] - right_eye[1]) / face_height,
        np.linalg.norm(forehead - nose_tip) / face_height,
        face_width / face_height,
        np.linalg.norm(left_eye - right_eye) / face_width,
        np.linalg.norm(nose_tip - forehead) / face_height,
        np.linalg.norm(nose_tip - chin) / face_height
    ]

    return np.array(features).reshape(1, -1)

st.title("Down Syndrome Facial Feature Detection")
st.write("Upload an image to analyze 12 geometric facial features.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = compute_features(image)

    if features is None:
        st.error("No face detected in the image.")
    else:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]


        if prediction == 1:
            st.error("Prediction: Down Syndrome Features Detected")
        else:
            st.success("Prediction: Normal")
        st.subheader("Prediction Confidence:")
        st.write(f"Normal: {probability[0]*100:.2f}%")
        st.write(f"Down Syndrome: {probability[1]*100:.2f}%")

        st.subheader("Extracted Feature Values:")
        feature_names = X.columns.tolist()

        for name, value in zip(feature_names, features[0]):
            st.write(f"{name}: {value:.4f}")