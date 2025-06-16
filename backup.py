import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io

# -------------------------------
# Configuration
# -------------------------------
YOLO_MODEL_PATH = "/models/best.pt"
CNN_MODEL_PATH = "/models/cnn.keras"

# -------------------------------
# Load Models
# -------------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)

CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_int = {char: i for i, char in enumerate(CHARACTERS)}
int_to_char = {i: char for char, i in char_to_int.items()}

# -------------------------------
# Utility Functions
# -------------------------------
def show_image(title, image, cmap='gray'):
    buf = io.BytesIO()
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    st.image(buf.getvalue(), caption=title)

def detect_plates(image_np, image_name):
    results = yolo_model(image_np)
    boxes = results[0].boxes.xyxy
    plates = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image_np[y1:y2, x1:x2]
        plates.append((crop, (x1, y1, x2, y2)))

    return plates, results[0].plot()

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upsampled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upsampled)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    show_image("Enhanced Image", sharpened)
    return sharpened

def adaptive_threshold(img):
    mean_intensity = np.mean(img)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mean_intensity < 127:
        binary = cv2.bitwise_not(binary)
    show_image("Binarized Image", binary)
    return binary

def segment_characters(image):
    enhanced = enhance_image(image)
    resized = cv2.resize(enhanced, (333, 75))
    binary = adaptive_threshold(resized)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    show_image("Post Morphology", binary)

    LP_WIDTH, LP_HEIGHT = binary.shape
    binary[:3, :] = 255
    binary[:, :3] = 255
    binary[LP_WIDTH-3:, :] = 255
    binary[:, LP_HEIGHT-3:] = 255

    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    debug = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug, contours, -1, (0, 255, 255), 1)
    show_image("Contours", debug, cmap=None)

    char_imgs, bboxes, centroids = [], [], []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 15 < w < 100 and 20 < h < 100:
            char = binary[y:y+h, x:x+w]
            char = cv2.copyMakeBorder(char, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            resized_char = cv2.resize(char, (32, 32))
            char_imgs.append(resized_char)
            bboxes.append((x, y, w, h))
            centroids.append((x + w//2, y + h//2))

    if not char_imgs:
        return [], [], debug

    cy_range = max(c[1] for c in centroids) - min(c[1] for c in centroids)
    if cy_range < 20:
        sorted_chars = sorted(zip(bboxes, char_imgs), key=lambda b: b[0][0])
    else:
        median_cy = np.median([cy for _, cy in centroids])
        top = [i for i, (_, cy) in enumerate(centroids) if cy < median_cy]
        bottom = [i for i in range(len(char_imgs)) if i not in top]
        sorted_chars = sorted([(bboxes[i], char_imgs[i]) for i in top], key=lambda b: b[0][0]) + \
                       sorted([(bboxes[i], char_imgs[i]) for i in bottom], key=lambda b: b[0][0])

    bboxes = [item[0] for item in sorted_chars]
    char_imgs = [item[1] for item in sorted_chars]

    return char_imgs, bboxes, debug

def pad_and_prepare(img):
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def correct_plate_text(pred):
    # your detailed correction logic (unchanged for brevity)
    # can be inserted here from your previous version
    return pred, "Unknown State"  # simplified; replace with actual logic

def predict_plate(chars, bboxes, plate_img):
    prediction = ""
    debug_img = plate_img.copy()
    for char_img, bbox in zip(chars, bboxes):
        proc = pad_and_prepare(char_img)
        pred = cnn_model.predict(proc, verbose=0)
        label = int_to_char[np.argmax(pred)]
        prediction += label
        x, y, w, h = bbox
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    show_image("Final Prediction", debug_img, cmap=None)
    corrected_prediction = correct_plate_text(prediction)
    return prediction, corrected_prediction

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🚗 Vehicle License Plate Detection and Recognition")
option = st.sidebar.radio("Select Mode", ["Upload Images"])

if option == "Upload Images":
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_name = uploaded_file.name.split('.')[0]
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            st.subheader(f"🗃️ {uploaded_file.name}")
            plates, annotated = detect_plates(image_bgr, image_name)
            st.image(annotated, caption="Detected Plates")

            for idx, (plate_img, box) in enumerate(plates):
                st.markdown(f"#### Plate Region {idx + 1}")
                chars, bboxes, segmented_debug = segment_characters(plate_img)
                if not chars:
                    st.warning("No characters found.")
                    continue
                raw_text, (corrected, state) = predict_plate(chars, bboxes, segmented_debug)

                st.markdown(f"**📄 Raw Predicted Text:** `{raw_text}`")
                st.markdown(f"**✅ Corrected Text:** `{corrected}`")
                st.markdown(f"🌐 Vehicle registered in: `{state}`")
