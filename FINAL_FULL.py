import streamlit as st 
import cv2
import numpy as np
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
from ultralytics import YOLO
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import string

# --------------------------------
# Config
# --------------------------------
YOLO_MODEL_PATH = "best.pt"
OCR_MODEL_PATH_PT = "ocr_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character set
CHARACTERS = list(string.digits + string.ascii_uppercase)
char_to_int = {char: i for i, char in enumerate(CHARACTERS)}
int_to_char = {i: char for char, i in char_to_int.items()}
num_classes = len(CHARACTERS)

# Define the OCR model architecture (must match the saved model)# Define your model class (must match the training model)import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Assuming input image size is 28x28
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the weights into a new model instance
ocr_model = SimpleCNN(num_classes=36).to(device)
ocr_model.load_state_dict(torch.load("ocr_model.pth", map_location=device))
ocr_model.eval()



# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)


# CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# char_to_int = {char: i for i, char in enumerate(CHARACTERS)}
# int_to_char = {i: char for char, i in char_to_int.items()}

# --------------------------------
# Helper functions
# --------------------------------
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

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = image_np[y1:y2, x1:x2]
        plates.append((crop, (x1, y1, x2, y2)))  # No need to save to disk

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

    cntrs, _ = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    debug = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug, cntrs, -1, (0, 255, 255), 1)
    show_image("Contours", debug, cmap=None)

    char_imgs, bboxes, centroids = [], [], []
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    for cnt in cntrs:
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
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 1, H, W)
    return tensor


def correct_plate_text(pred):
    STATE_CODES = {
        'AN': 'Andaman and Nicobar Islands', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
        'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh', 'DD': 'Daman and Diu',
        'DL': 'Delhi', 'DN': 'Dadra and Nagar Haveli', 'GA': 'Goa', 'GJ': 'Gujarat', 'HP': 'Himachal Pradesh',
        'HR': 'Haryana', 'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala',
        'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya', 'MN': 'Manipur',
        'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab',
        'PY': 'Puducherry', 'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura',
        'TS': 'Telangana', 'UK': 'Uttarakhand', 'UP': 'Uttar Pradesh', 'WB': 'West Bengal'
    }

    COMMON_CONFUSIONS = {
        'O': '0',
        'Q': '0', 
        'D': '0',
        '0': 'O',
        'I': '1',
        'L': '4',
        'T': '1',
        '1': 'I',
        'Z': '2',
        '2': 'Z', 
        'S': '5', 
        '5': 'S',
        'B': '8',
        '8': 'B', 
        'G': '0', 
        '6': 'G',
        'J': '3',
    }

    pred = pred.strip().upper()
    original = pred

    if not (9 <= len(pred) <= 10):
        print(f"❌ Invalid length: {pred}")
        return pred

    chars = list(pred)
    pattern = ['A', 'A', 'N', 'N', 'A', 'A', 'N', 'N', 'N', 'N'] if len(chars) == 10 else ['A', 'A', 'N', 'N', 'A', 'N', 'N', 'N', 'N']

    for i in range(len(chars)):
        expected = pattern[i]
        c = chars[i]
        if expected == 'A' and not c.isalpha():
            corrected = COMMON_CONFUSIONS.get(c, 'A')
            print(f"🔠 Position {i+1}: {c} → {corrected} (expected letter)")
            chars[i] = corrected
        elif expected == 'N' and not c.isdigit():
            corrected = COMMON_CONFUSIONS.get(c, '0')
            print(f"🔢 Position {i+1}: {c} → {corrected} (expected digit)")
            chars[i] = corrected

    c1, c2 = chars[0], chars[1]
    state_code = c1 + c2

    if state_code not in STATE_CODES:
        if c2 == 'B':
            if c1 in {'H', 'M', 'N'}:
                chars[0] = 'W'; state_code = 'WB'
                print(f"⚠️ State code {c1+c2} invalid → correcting to WB")
            elif c1 not in {'P', 'W'}:
                chars[0] = 'P'; state_code = 'PB'
                print(f"⚠️ State code {c1+c2} invalid → correcting to PB")
        elif c1 == 'D':
            if c2 in {'0', 'O', 'U'}:
                chars[1] = 'D'; state_code = 'DD'
                print(f"⚠️ State code {c1+c2} likely → DD (Daman and Diu)")
            elif c2 in {'1', '2', 'I', 'Z'}:
                chars[1] = 'L'; state_code = 'DL'
                print(f"⚠️ State code {c1+c2} likely → DL (Delhi)")
            elif c2 in {'W', 'V', 'M', 'N'}:
                chars[1] = 'N'; state_code = 'DN'
                print(f"⚠️ State code {c1+c2} likely → DN (Dadra and Nagar Haveli)")
        elif c2 == 'P' and c1 in {'N', 'H'}:
            chars[0] = 'M'; state_code = 'MP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to MP")
        elif c2 == 'H' and c1 in {'N', 'W'}:
            chars[0] = 'M'; state_code = 'MH'
            print(f"⚠️ State code {c1+c2} invalid → correcting to MH")
        elif c2 == 'P' and c1 in {'V', 'O'}:
            chars[0] = 'U'; state_code = 'UP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to UP")
        elif c1 == 'A' and c2 != 'P':
            chars[1] = 'P'; state_code = 'AP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to AP")
        elif c1 == 'T' and c2 != 'N':
            chars[1] = 'N'; state_code = 'TN'
            print(f"⚠️ State code {c1+c2} invalid → correcting to TN")
        elif c1 == 'G' and c2 not in {'J'}:
            chars[1] = 'J'; state_code = 'GJ'
            print(f"⚠️ State code {c1+c2} invalid → correcting to GJ")
        elif c2 == 'P' and c1  in {'W','V'}:
            chars[1] = 'MP'; state_code = 'MP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to MP")    

    corrected = ''.join(chars)

    if corrected != original:
        print(f"🔁 Predicted: {original} → Corrected: {corrected}")
    else:
        print(f"✅ Plate Format Valid: {corrected}")

    state_code = corrected[:2]
    state_name = STATE_CODES.get(state_code, "Unknown State Code")
    print(f"🌐 Vehicle registered in: {state_name} ({state_code})")

    return corrected, state_name


def predict_plate(chars, bboxes, plate_img):
    prediction = ""
    debug_img = plate_img.copy()

    for char_img, bbox in zip(chars, bboxes):
        proc = pad_and_prepare(char_img)
        with torch.no_grad():
            pred = ocr_model(proc)
            label = int_to_char[int(torch.argmax(pred, dim=1).item())]

        prediction += label
        x, y, w, h = bbox
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    show_image("Final Prediction", debug_img, cmap=None)
    corrected_prediction = correct_plate_text(prediction)
    return prediction, corrected_prediction


# --------------------------------
# Streamlit UI
# --------------------------------
st.title("🚗 Vehicle License Plate Detection and Recognition")
option = st.sidebar.radio("Select Mode", ["Upload Images", "Live Webcam Detection"])

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

