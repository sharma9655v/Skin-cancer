import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import uuid
import qrcode
import os
import google.generativeai as genai
import json
import traceback
import sqlite3
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance
import folium
from streamlit_folium import st_folium
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Skin Cancer Detection", page_icon="🧬", layout="wide")

# ================= APP INFO =================
APP_VERSION = "6.0"

HAM10000_CLASSES = [
    "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
    "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesion",
    "Fungal Infection", "Other"
]

BODY_LOCATIONS = [
    "scalp", "ear", "face", "back", "trunk", "chest", "abdomen",
    "upper extremity", "lower extremity", "hand", "foot", "neck", "genital"
]

FITZPATRICK_TYPES = {
    "Type I — Very Fair": 1, "Type II — Fair": 2, "Type III — Medium": 3,
    "Type IV — Olive": 4, "Type V — Brown": 5, "Type VI — Dark Brown/Black": 6
}

SUPPORTED_LANGUAGES = {
    "English": "en", "हिन्दी (Hindi)": "hi", "Español (Spanish)": "es",
    "Français (French)": "fr", "Deutsch (German)": "de", "中文 (Chinese)": "zh",
}

TRIAGE_LEVELS = {
    "LOW": {"color": "#51cf66", "icon": "🟢", "label": "Low Risk",
            "action": "Routine monitoring. Re-check in 3–6 months. Continue regular self-examinations."},
    "MODERATE": {"color": "#ffd43b", "icon": "🟡", "label": "Moderate Risk",
                 "action": "Schedule a dermatologist appointment within 2 weeks. Document any changes with photos."},
    "HIGH": {"color": "#ff6b6b", "icon": "🔴", "label": "High Risk",
             "action": "URGENT: Seek immediate dermatological evaluation. A biopsy may be recommended."},
}

# ================= DATABASE SETUP =================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age INTEGER, gender TEXT, patient_id TEXT,
            contact TEXT, exam_date TEXT, predicted_class TEXT,
            confidence REAL, cancer_prob REAL, risk_level TEXT,
            reasoning TEXT, recommendation TEXT, scan_timestamp TEXT,
            body_location TEXT, skin_type TEXT, lesion_size TEXT,
            lesion_changed TEXT, uncertainty TEXT, triage_level TEXT,
            inference_mode TEXT
        )
    """)
    # Add new columns if they don't exist (for existing DBs)
    for col, ctype in [("body_location","TEXT"),("skin_type","TEXT"),("lesion_size","TEXT"),
                       ("lesion_changed","TEXT"),("uncertainty","TEXT"),("triage_level","TEXT"),
                       ("inference_mode","TEXT"),("feature_vector","BLOB"),("image_thumbnail","BLOB")]:
        try:
            c.execute(f"ALTER TABLE scans ADD COLUMN {col} {ctype}")
        except Exception:
            pass
    conn.commit()
    conn.close()

def save_scan(patient_data, result, feature_vector=None, image_thumbnail=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cancer_prob = result.get("cancer_prob", 0.0)
    risk_percent = cancer_prob * 100
    risk_level = "LOW" if risk_percent < 30 else ("MODERATE" if risk_percent < 60 else "HIGH")
    fv_blob = None
    if feature_vector is not None:
        fv_blob = feature_vector.tobytes()
    thumb_blob = None
    if image_thumbnail is not None:
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(np.array(image_thumbnail.resize((112,112))), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        thumb_blob = buf.tobytes()
    c.execute("""
        INSERT INTO scans (name, age, gender, patient_id, contact, exam_date,
                          predicted_class, confidence, cancer_prob, risk_level,
                          reasoning, recommendation, scan_timestamp,
                          body_location, skin_type, lesion_size, lesion_changed,
                          uncertainty, triage_level, inference_mode, feature_vector, image_thumbnail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_data["name"], patient_data["age"], patient_data["gender"],
        patient_data["patient_id"], patient_data["contact"], patient_data["exam_date"],
        result.get("predicted_class", "Unknown"), result.get("confidence", 0.0),
        cancer_prob, risk_level,
        result.get("reasoning", ""), result.get("recommendation", ""),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient_data.get("body_location", ""), patient_data.get("skin_type", ""),
        patient_data.get("lesion_size", ""), patient_data.get("lesion_changed", ""),
        result.get("uncertainty", ""), risk_level, result.get("inference_mode", "cloud"),
        fv_blob, thumb_blob
    ))
    conn.commit()
    conn.close()

def get_all_scans():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM scans ORDER BY scan_timestamp DESC", conn)
    conn.close()
    return df

init_db()

# ================= SESSION STATE =================
for key, default in [("chat_history", []), ("last_result", None),
                     ("selected_language", "English"), ("active_page", "🏠 Home"),
                     ("use_local_model", False), ("enable_tta", False),
                     ("enable_hair_removal", True)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ================= TITLE =================
st.markdown("""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/2785/2785819.png' width='80'>
    <h1>🧬 AI Skin Disease & Cancer Detection System</h1>
    <p style='color: grey; font-size: 14px;'>v6.0 — Grad-CAM • TTA • Hair Removal • CBIR • Offline OOD • Tracking • Multi-Language</p>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
GEMINI_API_KEY = st.sidebar.text_input("Enter Gemini API Key", type="password")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.sidebar.warning("Enter Gemini API Key for cloud AI features.")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
st.sidebar.title("Navigation")
PAGE_OPTIONS = ["🏠 Home", "🔬 Prediction", "📊 Dashboard", "📈 Tracking", "🗺️ Find Clinics", "💬 Ask AI"]

def on_nav_change():
    st.session_state.active_page = st.session_state._nav_radio_widget

current_idx = PAGE_OPTIONS.index(st.session_state.active_page) if st.session_state.active_page in PAGE_OPTIONS else 0
st.sidebar.radio("Go to", PAGE_OPTIONS, index=current_idx, key="_nav_radio_widget", on_change=on_nav_change)
page = st.session_state.active_page

# Inference Mode Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Inference Mode")
use_local = st.sidebar.toggle("🔒 Use Local Model (Offline/Private)", value=st.session_state.use_local_model)
st.session_state.use_local_model = use_local
if use_local:
    st.sidebar.success("🔒 Images processed locally. No data sent to cloud.")
else:
    st.sidebar.info("☁️ Using Gemini Cloud AI for analysis.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Advanced Options")
enable_tta = st.sidebar.toggle("🎯 Enable TTA (Higher Accuracy)", value=st.session_state.enable_tta,
                               help="Test-Time Augmentation: runs 8 augmented versions for more robust predictions. ~8x slower.")
st.session_state.enable_tta = enable_tta
enable_hair = st.sidebar.toggle("✂️ Hair Removal Preprocessing", value=st.session_state.enable_hair_removal,
                                help="Remove hair artifacts from dermoscopy images before analysis.")
st.session_state.enable_hair_removal = enable_hair

theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Language")
selected_lang = st.sidebar.selectbox(
    "Translate results to:", list(SUPPORTED_LANGUAGES.keys()),
    index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.selected_language)
)
st.session_state.selected_language = selected_lang

if theme_mode == "Dark":
    st.markdown("<style>.stApp {background-color: #0E1117; color: white;}</style>", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================

def translate_text(text, target_language):
    if target_language == "English" or not text or not GEMINI_API_KEY:
        return text
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Translate the following medical text to {target_language}. Return ONLY the translated text:\n\n{text}"
        return model.generate_content(prompt).text.strip()
    except Exception:
        return text

# ================= HAIR REMOVAL (Feature 5) =================
def remove_hair(image_array):
    """Remove dark hair from dermoscopy images using morphological blackhat filter."""
    was_float = image_array.dtype == np.float32 or image_array.dtype == np.float64
    if was_float:
        img = (image_array * 255).astype(np.uint8)
    else:
        img = image_array.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cleaned = cv2.inpaint(img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    if was_float:
        return cleaned.astype(np.float32) / 255.0
    return cleaned

# ================= TEST-TIME AUGMENTATION (Feature 1) =================
def tta_predict(model, img_array, meta_array=None, n_augments=8):
    """Run multiple augmented versions through the model and average predictions."""
    augmented = [img_array]
    img_np = img_array[0]
    # Horizontal flip
    augmented.append(np.expand_dims(np.fliplr(img_np), 0))
    # Vertical flip
    augmented.append(np.expand_dims(np.flipud(img_np), 0))
    # 90 degree rotation
    augmented.append(np.expand_dims(np.rot90(img_np, 1), 0))
    # 180 degree rotation
    augmented.append(np.expand_dims(np.rot90(img_np, 2), 0))
    # 270 degree rotation
    augmented.append(np.expand_dims(np.rot90(img_np, 3), 0))
    # Slight brightness increase
    augmented.append(np.expand_dims(np.clip(img_np * 1.1, 0, 1), 0))
    # Slight brightness decrease
    augmented.append(np.expand_dims(np.clip(img_np * 0.9, 0, 1), 0))

    preds = []
    for aug_img in augmented[:n_augments]:
        if meta_array is not None:
            pred = model.predict([aug_img, meta_array], verbose=0)[0]
        else:
            pred = model.predict(aug_img, verbose=0)[0]
        preds.append(pred)
    return np.mean(preds, axis=0)

# ================= OFFLINE OOD DETECTION (Feature 7) =================
def check_ood_local(image):
    """Offline Out-of-Distribution detection using image heuristics."""
    img = np.array(image.resize((224, 224)))
    # Check 1: Skin-tone pixel ratio using HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    # Check 2: Texture check (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Check 3: Color diversity
    std_color = np.std(img.astype(float))
    if skin_ratio < 0.10:
        return False, "Image has very few skin-toned pixels. Likely not a skin lesion."
    if lap_var < 10:
        return False, "Image appears too uniform/smooth. May not be a valid photograph."
    if std_color < 8:
        return False, "Image lacks color variation. May be a solid color or blank image."
    return True, "Image passes local OOD checks."

# ================= FEATURE EXTRACTION FOR CBIR (Feature 6) =================
def extract_feature_vector(model, img_array, meta_array=None):
    """Extract the feature vector from the penultimate dense layer for CBIR."""
    try:
        # Find the GlobalAveragePooling2D or last dense layer before output
        feat_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                feat_layer = layer.name
                break
        if feat_layer is None:
            return None
        feat_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(feat_layer).output)
        if meta_array is not None:
            vec = feat_model.predict([img_array, meta_array], verbose=0)[0]
        else:
            vec = feat_model.predict(img_array, verbose=0)[0]
        return vec.astype(np.float32)
    except Exception:
        return None

def find_similar_cases(feature_vector, top_k=3):
    """Find the most similar past scans using cosine similarity."""
    if feature_vector is None:
        return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, patient_id, predicted_class, confidence, cancer_prob, risk_level, scan_timestamp, feature_vector, image_thumbnail FROM scans WHERE feature_vector IS NOT NULL ORDER BY id DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    similarities = []
    for row in rows:
        fv_blob = row[8]
        if fv_blob is None:
            continue
        stored_vec = np.frombuffer(fv_blob, dtype=np.float32)
        if stored_vec.shape != feature_vector.shape:
            continue
        try:
            sim = 1 - cosine_distance(feature_vector, stored_vec)
        except Exception:
            continue
        thumb_bytes = row[9]
        similarities.append({
            "id": row[0], "name": row[1], "patient_id": row[2],
            "predicted_class": row[3], "confidence": row[4],
            "cancer_prob": row[5], "risk_level": row[6],
            "timestamp": row[7], "similarity": sim,
            "thumbnail": thumb_bytes
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    # Skip the most similar if it's basically identical (self-match)
    results = [s for s in similarities if s["similarity"] < 0.9999]
    return results[:top_k]

# ================= GRAD-CAM =================
@st.cache_resource
def load_local_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin_cancer_model_v2.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    model_path_v1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin_cancer_model.h5")
    if os.path.exists(model_path_v1):
        return tf.keras.models.load_model(model_path_v1, compile=False)
    return None

@st.cache_resource
def load_model_artifacts():
    """Load model_artifacts.json to get metadata column info."""
    artifacts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.json")
    if os.path.exists(artifacts_path):
        with open(artifacts_path, "r") as f:
            return json.load(f)
    return None

def is_dual_input_model(model):
    """Check if the model expects dual inputs (image + metadata)."""
    try:
        return isinstance(model.input, list) and len(model.input) >= 2
    except Exception:
        try:
            return len(model.inputs) >= 2
        except Exception:
            return False

def build_metadata_tensor(metadata, model_artifacts=None):
    """Build a metadata tensor matching the model's expected metadata input shape."""
    if model_artifacts and "metadata_cols" in model_artifacts:
        meta_cols = model_artifacts["metadata_cols"]
        num_features = len(meta_cols)
    else:
        # Fallback: infer from model or use default HAM10000 metadata layout
        num_features = 17  # age_norm + sex_encoded + 15 localization dummies (typical)

    meta_values = np.zeros(num_features, dtype=np.float32)

    if metadata:
        # Age normalized (0-1)
        try:
            meta_values[0] = float(metadata.get("age", 50)) / 100.0
        except (ValueError, TypeError):
            meta_values[0] = 0.5

        # Sex encoded
        gender = metadata.get("gender", "").lower()
        if gender == "male":
            meta_values[1] = 0.0
        elif gender == "female":
            meta_values[1] = 1.0
        else:
            meta_values[1] = 0.5

        # Body location one-hot (if model_artifacts has the column names)
        if model_artifacts and "metadata_cols" in model_artifacts:
            body_loc = metadata.get("body_location", "").lower().replace(" ", "_")
            for i, col in enumerate(meta_cols):
                if col.startswith("loc_") and body_loc in col.lower():
                    meta_values[i] = 1.0

    return np.expand_dims(meta_values, axis=0)

def generate_gradcam(model, img_array, meta_array=None, class_index=None):
    """Generate real Grad-CAM heatmap from the model."""
    try:
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                last_conv = layer.name
                break
        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        # Build the correct input based on model type
        if is_dual_input_model(model) and meta_array is not None:
            model_input = [img_array, meta_array]
        else:
            model_input = img_array

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(model_input)
            if class_index is None:
                if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                    class_index = tf.argmax(predictions[0])
                else:
                    class_index = 0
            loss = predictions[:, class_index] if len(predictions.shape) > 1 else predictions

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    img = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(heatmap_colored * alpha + img * (1 - alpha))
    return superimposed

# ================= OOD DETECTION =================
def check_ood(image):
    """Out-of-Distribution detection — verify image is a skin lesion."""
    if not GEMINI_API_KEY:
        return True, "API key not available, skipping OOD check."
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = """Look at this image. Is it a clear photograph of a human skin lesion or skin condition?
Return ONLY a JSON object: {"is_skin": true/false, "reason": "brief explanation"}"""
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        result = json.loads(text.strip())
        return result.get("is_skin", True), result.get("reason", "")
    except Exception:
        return True, "OOD check unavailable."

# ================= MC DROPOUT INFERENCE =================
def mc_dropout_predict(model, img_array, meta_array=None, n_iter=30):
    """Monte Carlo Dropout: run n_iter forward passes with dropout ON."""
    predictions = []
    for _ in range(n_iter):
        if meta_array is not None:
            pred = model([img_array, meta_array], training=True)
        else:
            pred = model(img_array, training=True)
        predictions.append(pred.numpy())
    preds = np.array(predictions)
    mean_pred = preds.mean(axis=0)[0]
    std_pred = preds.std(axis=0)[0]
    return mean_pred, std_pred

# ================= PREDICT (CLOUD) =================
def predict_image_cloud(image, metadata=None):
    if not GEMINI_API_KEY:
        st.error("API Key is missing!")
        return None, None, None, None
    model = genai.GenerativeModel('gemini-2.5-flash')
    meta_context = ""
    if metadata:
        meta_context = f"""
Patient metadata:
- Age: {metadata.get('age', 'Unknown')}
- Gender: {metadata.get('gender', 'Unknown')}
- Body Location: {metadata.get('body_location', 'Unknown')}
- Fitzpatrick Skin Type: {metadata.get('skin_type', 'Unknown')}
- Lesion Size: {metadata.get('lesion_size', 'Unknown')} mm
- Lesion Changed Recently: {metadata.get('lesion_changed', 'Unknown')}
"""
    prompt = f"""You are an expert dermatologist AI. Analyze this skin image.
{meta_context}
Classify into one of: Melanoma, Basal Cell Carcinoma, Actinic Keratoses, Melanocytic Nevi,
Benign Keratosis, Dermatofibroma, Vascular Lesion, Fungal Infection, Other.

Return ONLY a JSON object (no markdown):
{{
  "predicted_class": "category name",
  "confidence": 85.5,
  "cancer_prob": 0.1,
  "all_probs": {{"Melanoma": 0.05, "Basal Cell Carcinoma": 0.03, "Actinic Keratoses": 0.02,
                 "Melanocytic Nevi": 0.5, "Benign Keratosis": 0.2, "Dermatofibroma": 0.05,
                 "Vascular Lesion": 0.05, "Fungal Infection": 0.05, "Other": 0.05}},
  "reasoning": "Detailed 3-5 sentence clinical explanation of the diagnosis.",
  "recommendation": "Detailed action recommendation with follow-up steps.",
  "lesion_description": "Detailed clinical description of the lesion appearance, texture, and surface characteristics.",
  "morphology": "Describe the shape, symmetry, elevation, and structural features of the lesion.",
  "color_pattern": "Describe the color distribution including any pigment variations, shading, or multi-toned areas.",
  "border_analysis": "Describe the border characteristics — regular/irregular, well-defined/diffuse, notched/smooth.",
  "differential_diagnosis": ["Condition 1", "Condition 2", "Condition 3"],
  "risk_factors": "Relevant risk factors based on lesion features and patient metadata."
}}"""
    try:
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        result = json.loads(text.strip())
        result["inference_mode"] = "cloud"
        st.session_state.last_result = result
        return result, result.get("predicted_class"), result.get("cancer_prob", 0.0), result.get("confidence", 0.0)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, "Error", 0.0, 0.0

# ================= PREDICT (LOCAL) =================
def predict_image_local(image, metadata=None):
    local_model = load_local_model()
    if local_model is None:
        st.error("No local model found! Train the model first or switch to Cloud AI.")
        return None, None, None, None
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    # Feature 5: Hair Removal preprocessing
    if st.session_state.get("enable_hair_removal", False):
        img_array = remove_hair(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # Build metadata tensor if model expects dual inputs
    meta_array = None
    if is_dual_input_model(local_model):
        model_artifacts = load_model_artifacts()
        meta_array = build_metadata_tensor(metadata, model_artifacts)

    # Try MC Dropout
    try:
        output_shape = local_model.output_shape
        num_outputs = output_shape[-1] if isinstance(output_shape[-1], int) else 1
    except Exception:
        num_outputs = 1

    # Feature 1: TTA or MC Dropout prediction
    if st.session_state.get("enable_tta", False):
        try:
            mean_pred = tta_predict(local_model, img_array, meta_array=meta_array, n_augments=8)
            std_pred = np.zeros_like(mean_pred)
        except Exception:
            if meta_array is not None:
                mean_pred = local_model.predict([img_array, meta_array], verbose=0)[0]
            else:
                mean_pred = local_model.predict(img_array, verbose=0)[0]
            std_pred = np.zeros_like(mean_pred)
    else:
        try:
            mean_pred, std_pred = mc_dropout_predict(local_model, img_array, meta_array=meta_array, n_iter=30)
        except Exception:
            if meta_array is not None:
                mean_pred = local_model.predict([img_array, meta_array], verbose=0)[0]
            else:
                mean_pred = local_model.predict(img_array, verbose=0)[0]
            std_pred = np.zeros_like(mean_pred)

    # Build result
    if num_outputs > 1:
        classes = HAM10000_CLASSES[:num_outputs]
        pred_idx = int(np.argmax(mean_pred))
        predicted_class = classes[pred_idx]
        confidence = float(mean_pred[pred_idx] * 100)
        cancer_classes = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
        cancer_prob = float(sum(mean_pred[i] for i, c in enumerate(classes) if c in cancer_classes))
        all_probs = {c: float(mean_pred[i]) for i, c in enumerate(classes)}
        uncertainty_str = "; ".join([f"{c}: {mean_pred[i]*100:.1f}% ± {std_pred[i]*100:.1f}%" for i, c in enumerate(classes)])
    else:
        cancer_prob = float(mean_pred[0]) if len(mean_pred.shape) == 0 or mean_pred.shape[0] == 1 else float(mean_pred[0])
        predicted_class = "Malignant" if cancer_prob > 0.5 else "Benign"
        confidence = float(max(cancer_prob, 1 - cancer_prob) * 100)
        all_probs = {"Malignant": cancer_prob, "Benign": 1 - cancer_prob}
        uncertainty_str = f"Cancer: {cancer_prob*100:.1f}% ± {float(std_pred)*100:.1f}%"

    result = {
        "predicted_class": predicted_class, "confidence": confidence,
        "cancer_prob": cancer_prob, "all_probs": all_probs,
        "reasoning": "Classification based on local CNN model with MC Dropout uncertainty estimation.",
        "recommendation": "Please consult a dermatologist for clinical confirmation.",
        "uncertainty": uncertainty_str, "inference_mode": "local"
    }
    st.session_state.last_result = result
    return result, predicted_class, cancer_prob, confidence

def get_triage(cancer_prob):
    pct = cancer_prob * 100
    if pct < 30: return TRIAGE_LEVELS["LOW"]
    elif pct < 60: return TRIAGE_LEVELS["MODERATE"]
    else: return TRIAGE_LEVELS["HIGH"]

# ================= PDF GENERATOR (Professional Lab Report Style) =================
def generate_pdf(patient_data, image, result, gradcam_overlay=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch, mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    from io import BytesIO

    predicted = result.get("predicted_class", "Unknown")
    cancer_prob = result.get("cancer_prob", 0.0)
    confidence = result.get("confidence", 0.0)
    reasoning = result.get("reasoning", "")
    recommendation = result.get("recommendation", "")
    all_probs = result.get("all_probs", {})
    uncertainty = result.get("uncertainty", "N/A")
    triage = get_triage(cancer_prob)
    risk_level = "LOW" if cancer_prob * 100 < 30 else ("MODERATE" if cancer_prob * 100 < 60 else "HIGH")
    report_id = str(uuid.uuid4())[:12].upper()
    now = datetime.now()
    collected_time = now.strftime("%d/%m/%Y  %I:%M:%S %p")
    reported_time = now.strftime("%d/%m/%Y  %I:%M:%S %p")

    filename = "AI_Skin_Cancer_Report.pdf"

    # ── Color Palette (Dr Lal PathLabs inspired) ──
    GOLD = colors.HexColor("#D4A017")
    DARK_GOLD = colors.HexColor("#B8860B")
    HEADER_BG = colors.HexColor("#D4A017")
    LIGHT_GRAY = colors.HexColor("#F5F5F5")
    SECTION_BG = colors.HexColor("#E8E8E8")
    WHITE = colors.white
    BLACK = colors.black
    RED_FLAG = colors.HexColor("#CC0000")
    GREEN_OK = colors.HexColor("#228B22")

    # ── Custom Styles ──
    styles = getSampleStyleSheet()

    style_header_title = ParagraphStyle(
        'HeaderTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=16,
        textColor=WHITE, alignment=TA_LEFT
    )
    style_header_sub = ParagraphStyle(
        'HeaderSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7,
        textColor=WHITE, alignment=TA_LEFT, leading=9
    )
    style_section_header = ParagraphStyle(
        'SectionHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9,
        textColor=BLACK, backColor=SECTION_BG, leading=14,
        spaceBefore=4, spaceAfter=2
    )
    style_label = ParagraphStyle(
        'Label', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10
    )
    style_value = ParagraphStyle(
        'Value', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=10
    )
    style_value_bold = ParagraphStyle(
        'ValueBold', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=RED_FLAG
    )
    style_small = ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7, leading=9,
        textColor=colors.HexColor("#555555")
    )
    style_footer = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6, leading=8,
        textColor=colors.HexColor("#666666"), alignment=TA_CENTER
    )
    style_note = ParagraphStyle(
        'Note', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7, leading=9,
        textColor=BLACK, leftIndent=15
    )
    style_table_header = ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10
    )
    style_disclaimer = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=5.5, leading=7,
        textColor=colors.HexColor("#333333"), alignment=TA_CENTER
    )

    # ── Helper: Check if value is abnormal ──
    cancer_classes = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
    def is_abnormal_class(cls_name, prob_val):
        if cls_name in cancer_classes and prob_val > 0.1:
            return True
        return prob_val > 0.5

    # ── Save temporary images ──
    original = np.array(image)
    resized = cv2.resize(original, (224, 224))
    cv2.imwrite("original.png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

    if gradcam_overlay is not None:
        cv2.imwrite("gradcam.png", cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
    else:
        heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        cv2.imwrite("gradcam.png", heatmap)

    # Probability chart
    if all_probs:
        plt.figure(figsize=(5.5, 2.5))
        classes_list = list(all_probs.keys())
        probs = [v * 100 for v in all_probs.values()]
        bar_colors = ['#CC0000' if c in cancer_classes else '#228B22' for c in classes_list]
        plt.barh(classes_list, probs, color=bar_colors, height=0.6)
        plt.xlabel("Probability (%)", fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig("prob.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(4, 2))
        plt.bar(["Cancer", "Benign"], [cancer_prob*100, (1-cancer_prob)*100],
                color=['#CC0000', '#228B22'])
        plt.ylabel("Probability (%)", fontsize=8)
        plt.tight_layout()
        plt.savefig("prob.png", dpi=150)
        plt.close()

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=4, border=1)
    qr.add_data(f"REPORT-{report_id}|{patient_data.get('name','')}|{predicted}|{confidence:.1f}%")
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save("report_qr.png")

    # ── Page Template with header/footer ──
    PAGE_W, PAGE_H = A4
    page_num_storage = [0]

    def draw_header_footer(canvas_obj, doc_obj):
        page_num_storage[0] += 1
        canvas_obj.saveState()

        # ── HEADER: Gold banner ──
        canvas_obj.setFillColor(HEADER_BG)
        canvas_obj.rect(0, PAGE_H - 70, PAGE_W, 70, fill=True, stroke=False)

        # Clinic/System name
        canvas_obj.setFillColor(WHITE)
        canvas_obj.setFont("Helvetica-Bold", 18)
        canvas_obj.drawString(25, PAGE_H - 30, "AI DermaScan")
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawString(25, PAGE_H - 42, "Advanced AI-Powered Skin Disease Detection System")
        canvas_obj.setFont("Helvetica", 6.5)
        canvas_obj.drawString(25, PAGE_H - 53, "Powered by EfficientNet Deep Learning | Grad-CAM Explainability | Multi-Modal Analysis")
        canvas_obj.drawString(25, PAGE_H - 63, "Web: ai-dermascan.health  |  Version 6.0  |  Gautam Buddha University")

        # Right side: Accreditation-style badges
        canvas_obj.setFont("Helvetica-Bold", 7)
        canvas_obj.drawRightString(PAGE_W - 25, PAGE_H - 28, "AI-VERIFIED")
        canvas_obj.setFont("Helvetica", 6)
        canvas_obj.drawRightString(PAGE_W - 25, PAGE_H - 38, "Deep Learning")
        canvas_obj.drawRightString(PAGE_W - 25, PAGE_H - 47, "Certified Analysis")

        # Thin gold line below header
        canvas_obj.setStrokeColor(DARK_GOLD)
        canvas_obj.setLineWidth(2)
        canvas_obj.line(0, PAGE_H - 72, PAGE_W, PAGE_H - 72)

        # ── FOOTER ──
        # Gold line above footer
        canvas_obj.setStrokeColor(DARK_GOLD)
        canvas_obj.setLineWidth(1)
        canvas_obj.line(25, 50, PAGE_W - 25, 50)

        # Disclaimer
        canvas_obj.setFillColor(colors.HexColor("#333333"))
        canvas_obj.setFont("Helvetica", 5.5)
        canvas_obj.drawCentredString(PAGE_W / 2, 40,
            "If test results are alarming or unexpected, the patient is advised to consult a dermatologist immediately for clinical evaluation.")
        canvas_obj.drawCentredString(PAGE_W / 2, 33,
            "This report is generated by an AI system and should be used as a screening aid only. It is NOT a substitute for professional medical diagnosis.")
        canvas_obj.drawCentredString(PAGE_W / 2, 26,
            f"AI DermaScan v6.0 | Gautam Buddha University | Report ID: {report_id}")

        # Page number
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawRightString(PAGE_W - 25, 26, f"Page {page_num_storage[0]}")

        canvas_obj.restoreState()

    # ── Build Document ──
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        topMargin=85, bottomMargin=60,
        leftMargin=25, rightMargin=25
    )
    elements = []

    # ── LAB INFO LINE ──
    lab_info = Table([
        [Paragraph("<b>AI DERMASCAN — SKIN DISEASE DETECTION LABORATORY</b>", style_label),
         Paragraph("", style_value)],
    ], colWidths=[4.5*inch, 2.5*inch])
    lab_info.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(lab_info)
    elements.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════
    # SECTION 1: PATIENT INFORMATION (Lab-style block)
    # ══════════════════════════════════════════════════

    patient_info_data = [
        [Paragraph("<b>Name</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{patient_data.get('name', 'N/A')}", style_value),
         Paragraph("<b>Collected</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{collected_time}", style_value)],
        [Paragraph("<b>Lab No.</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{report_id}", style_value),
         Paragraph("<b>Received</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{collected_time}", style_value)],
        [Paragraph(f"<b>Age: {patient_data.get('age', 'N/A')} Years</b>", style_label),
         Paragraph(f"<b>Gender:&nbsp;&nbsp;{patient_data.get('gender', 'N/A')}</b>", style_value),
         Paragraph("<b>Reported</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{reported_time}", style_value)],
        [Paragraph("<b>Ref By</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{result.get('inference_mode', 'Cloud AI').upper()}", style_value),
         Paragraph("<b>Report Status</b>", style_label),
         Paragraph(":&nbsp;&nbsp;&nbsp;<b>Final</b>", style_value_bold if risk_level == "HIGH" else style_value)],
    ]

    patient_table = Table(patient_info_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    patient_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.5, BLACK),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════
    # SECTION 2: SKIN DISEASE CLASSIFICATION PANEL
    # ══════════════════════════════════════════════════

    # Section header (yellow background like Dr Lal PathLabs)
    panel_header = Table(
        [[Paragraph("<b>Test Name</b>", style_table_header),
          Paragraph("<b>Results</b>", style_table_header),
          Paragraph("<b>Units</b>", style_table_header),
          Paragraph("<b>Bio. Ref. Interval</b>", style_table_header)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    panel_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECTION_BG),
        ('BOX', (0, 0), (-1, -1), 1, BLACK),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(panel_header)

    # Sub-header: Panel Name
    skin_panel_title = Table(
        [[Paragraph("<b>SKIN DISEASE CLASSIFICATION PANEL</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    skin_panel_title.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(skin_panel_title)

    # Classification results rows
    ref_intervals = {
        "Melanoma": "0.00 - 5.00",
        "Basal Cell Carcinoma": "0.00 - 5.00",
        "Actinic Keratoses": "0.00 - 5.00",
        "Melanocytic Nevi": "0.00 - 100.00",
        "Benign Keratosis": "0.00 - 100.00",
        "Dermatofibroma": "0.00 - 100.00",
        "Vascular Lesion": "0.00 - 100.00",
        "Fungal Infection": "0.00 - 100.00",
        "Other": "0.00 - 100.00",
    }

    results_data = []
    abnormal_rows = []

    if all_probs:
        for idx, (cls_name, prob_val) in enumerate(all_probs.items()):
            pct = prob_val * 100
            is_abnormal = is_abnormal_class(cls_name, prob_val)
            if is_abnormal:
                abnormal_rows.append(idx)

            val_style = style_value_bold if is_abnormal else style_value
            marker = " ▲" if is_abnormal and cls_name in cancer_classes else ""

            results_data.append([
                Paragraph(cls_name, style_value),
                Paragraph(f"<b>{pct:.2f}</b>{marker}" if is_abnormal else f"{pct:.2f}", val_style),
                Paragraph("%", style_value),
                Paragraph(ref_intervals.get(cls_name, "0.00 - 100.00"), style_value),
            ])

    if results_data:
        results_table = Table(results_data, colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch])
        table_style_cmds = [
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ]
        # Highlight abnormal rows
        for row_idx in abnormal_rows:
            table_style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor("#FFF0F0")))
        results_table.setStyle(TableStyle(table_style_cmds))
        elements.append(results_table)

    elements.append(Spacer(1, 6))

    # ── Summary Row ──
    summary_data = [
        [Paragraph("<b>Primary Diagnosis</b>", style_label),
         Paragraph(f"<b>{predicted}</b>", style_value_bold if risk_level == "HIGH" else ParagraphStyle('GreenBold', parent=style_value, fontName='Helvetica-Bold', textColor=GREEN_OK)),
         Paragraph("<b>Confidence</b>", style_label),
         Paragraph(f"<b>{confidence:.1f}%</b>", style_value)],
        [Paragraph("<b>Cancer Probability</b>", style_label),
         Paragraph(f"<b>{cancer_prob*100:.1f}%</b>", style_value_bold if cancer_prob > 0.3 else style_value),
         Paragraph("<b>Risk Level</b>", style_label),
         Paragraph(f"<b>{risk_level}</b>", style_value_bold if risk_level == "HIGH" else style_value)],
    ]
    summary_table = Table(summary_data, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
    summary_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.5, DARK_GOLD),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D4A017")),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6))

    # ── Triage / Risk Assessment section header ──
    triage_header = Table(
        [[Paragraph("<b>RISK ASSESSMENT &amp; TRIAGE</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    triage_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(triage_header)

    triage_color = colors.HexColor("#FFEBEE") if risk_level == "HIGH" else (
        colors.HexColor("#FFF8E1") if risk_level == "MODERATE" else colors.HexColor("#E8F5E9"))

    triage_data = [
        [Paragraph("<b>Triage Level</b>", style_label),
         Paragraph(f"{triage['label']}", style_value_bold if risk_level == "HIGH" else style_value),
         Paragraph("Assessment", style_value),
         Paragraph(ref_intervals.get(predicted, "Screening"), style_value)],
        [Paragraph("<b>Recommended Action</b>", style_label),
         Paragraph(triage['action'], style_value),
         Paragraph("", style_value),
         Paragraph("", style_value)],
    ]
    triage_table = Table(triage_data, colWidths=[1.5*inch, 2.5*inch, 1.0*inch, 2.0*inch])
    triage_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ('BACKGROUND', (0, 0), (-1, -1), triage_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('SPAN', (1, 1), (3, 1)),
    ]))
    elements.append(triage_table)

    if uncertainty and uncertainty != "N/A":
        elements.append(Spacer(1, 3))
        elements.append(Paragraph(f"<i>Uncertainty (MC Dropout): {uncertainty}</i>", style_small))

    elements.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════
    # SECTION 3: IMAGE ANALYSIS (Original + Grad-CAM)
    # ══════════════════════════════════════════════════

    img_header = Table(
        [[Paragraph("<b>IMAGE ANALYSIS — Submitted Specimen &amp; AI Heatmap</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    img_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
    ]))
    elements.append(img_header)
    elements.append(Spacer(1, 4))

    # Images side by side
    try:
        img_table = Table([
            [RLImage("original.png", width=2.2*inch, height=2.2*inch),
             RLImage("gradcam.png", width=2.2*inch, height=2.2*inch),
             RLImage("report_qr.png", width=0.9*inch, height=0.9*inch)],
            [Paragraph("<b>Original Image</b>", ParagraphStyle('ImgCap', parent=style_small, alignment=TA_CENTER)),
             Paragraph("<b>Grad-CAM Heatmap</b>", ParagraphStyle('ImgCap2', parent=style_small, alignment=TA_CENTER)),
             Paragraph("<b>Scan QR</b>", ParagraphStyle('ImgCap3', parent=style_small, alignment=TA_CENTER))],
        ], colWidths=[2.5*inch, 2.5*inch, 1.2*inch])
        img_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(img_table)
    except Exception:
        elements.append(Paragraph("Image rendering unavailable.", style_value))

    elements.append(Spacer(1, 6))

    # Probability Chart
    try:
        elements.append(RLImage("prob.png", width=5.0*inch, height=2.2*inch))
    except Exception:
        pass

    elements.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════
    # SECTION 4: CLINICAL ANALYSIS TABLE
    # ══════════════════════════════════════════════════

    clinical_header = Table(
        [[Paragraph("<b>AI CLINICAL ANALYSIS</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    clinical_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
    ]))
    elements.append(clinical_header)

    clinical_rows = []

    # Extract detailed fields
    lesion_desc = result.get("lesion_description", "")
    morphology = result.get("morphology", "")
    color_pattern = result.get("color_pattern", "")
    border_analysis = result.get("border_analysis", "")
    differential = result.get("differential_diagnosis", [])
    risk_factors = result.get("risk_factors", "")

    analysis_items = [
        ("Clinical Observations", reasoning),
        ("Clinical Description", lesion_desc),
        ("Morphology (Shape & Structure)", morphology),
        ("Color Pattern", color_pattern),
        ("Border Analysis", border_analysis),
        ("Differential Diagnoses", ", ".join(differential) if isinstance(differential, list) else str(differential)),
        ("Risk Factors", risk_factors),
        ("Anatomical Site", patient_data.get("body_location", "N/A")),
        ("Measured Size", f"{patient_data.get('lesion_size', 'N/A')} mm"),
        ("Skin Type (Fitzpatrick)", patient_data.get("skin_type", "N/A")),
        ("Recent Change Reported", patient_data.get("lesion_changed", "N/A")),
        ("Recommendation", recommendation),
    ]

    for label_text, value_text in analysis_items:
        if value_text and str(value_text).strip():
            clinical_rows.append([
                Paragraph(f"<b>{label_text}</b>", style_label),
                Paragraph(str(value_text), style_value),
            ])

    if clinical_rows:
        clinical_table = Table(clinical_rows, colWidths=[2.0*inch, 5.0*inch])
        clinical_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(clinical_table)
    else:
        elements.append(Paragraph(
            "<i>Detailed clinical analysis is available when using Cloud AI inference mode.</i>",
            style_small))

    elements.append(Spacer(1, 15))

    # ══════════════════════════════════════════════════
    # SECTION 5: NOTES & IMPORTANT INSTRUCTIONS
    # ══════════════════════════════════════════════════

    notes_header = Table(
        [[Paragraph("<b>Note</b>", style_label)]],
        colWidths=[7.0*inch]
    )
    notes_header.setStyle(TableStyle([
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(notes_header)

    notes = [
        "1.  Analysis conducted on submitted digital dermoscopy / clinical photograph using EfficientNet AI model.",
        "2.  Results are probability-based screening and do NOT constitute a definitive medical diagnosis.",
        "3.  All suspicious findings (cancer probability > 10%) should be clinically correlated with biopsy.",
        "4.  Grad-CAM heatmap indicates regions of interest — does not define lesion boundaries.",
        "5.  MC Dropout uncertainty quantification provides confidence intervals for the AI prediction.",
    ]
    for note in notes:
        elements.append(Paragraph(note, style_note))
    elements.append(Spacer(1, 10))

    # ── Signature Area ──
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999")))
    elements.append(Spacer(1, 5))

    sig_data = [
        [Paragraph("", style_value),
         Paragraph("", style_value),
         Paragraph("", style_value)],
        [Paragraph("___________________", style_small),
         Paragraph("___________________", style_small),
         Paragraph("___________________", style_small)],
        [Paragraph("<b>AI DermaScan System</b>", ParagraphStyle('SigL', parent=style_small, alignment=TA_CENTER)),
         Paragraph("<b>Reviewing Physician</b>", ParagraphStyle('SigC', parent=style_small, alignment=TA_CENTER)),
         Paragraph("<b>Consulting Dermatologist</b>", ParagraphStyle('SigR', parent=style_small, alignment=TA_CENTER))],
        [Paragraph("Deep Learning Engine", ParagraphStyle('SigL2', parent=style_small, alignment=TA_CENTER)),
         Paragraph("MD, Pathology", ParagraphStyle('SigC2', parent=style_small, alignment=TA_CENTER)),
         Paragraph("MD, Dermatology", ParagraphStyle('SigR2', parent=style_small, alignment=TA_CENTER))],
    ]
    sig_table = Table(sig_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(sig_table)

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "——————————— End of report ———————————",
        ParagraphStyle('EndReport', parent=style_small, alignment=TA_CENTER, fontName='Helvetica-Bold')
    ))

    elements.append(Spacer(1, 8))

    # ── IMPORTANT INSTRUCTIONS ──
    elements.append(HRFlowable(width="100%", thickness=1, color=DARK_GOLD))
    elements.append(Spacer(1, 3))

    instructions_title = Paragraph("<b>IMPORTANT INSTRUCTIONS</b>", ParagraphStyle(
        'InstrTitle', parent=style_small, alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=7))
    elements.append(instructions_title)
    elements.append(Spacer(1, 3))

    instructions = [
        "*All results pertain to the image submitted. All AI analyses are dependent on the quality of the image provided.",
        "*AI investigations are only a tool to facilitate arriving at a diagnosis and should be clinically correlated by the Referring Physician.",
        "*Report delivery may be delayed due to unforeseen system load. Kindly submit request within 72 hours post analysis.",
        "*AI results may show inter-analysis variations based on image quality, lighting, and resolution.",
        "*The Courts/Forum at Greater Noida shall have exclusive jurisdiction in all disputes regarding the report.",
        "*Results are not valid for medico legal purposes. Contact support for all queries related to reports.",
    ]
    for instr in instructions:
        elements.append(Paragraph(instr, style_disclaimer))

    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=1, color=DARK_GOLD))

    # Build the PDF with header/footer
    doc.build(elements, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    return filename
# ===================================================================
#                           PAGE: HOME
# ===================================================================
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h2>Welcome to the AI Skin Disease Detection Platform</h2>
        <p style='font-size: 18px; color: grey;'>
            Upload or capture a skin image for instant AI-powered diagnosis,<br>
            with explainable AI, triage recommendations, and privacy-first design.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px; border-radius: 15px; text-align: center; color: white;'>
            <h1 style='font-size: 40px; margin: 0;'>🔬</h1>
            <h3>AI Diagnosis</h3>
            <p>Multi-class detection with Grad-CAM explainability and triage.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Prediction →", key="card_prediction", use_container_width=True):
            st.session_state.active_page = "🔬 Prediction"
            st.rerun()
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 25px; border-radius: 15px; text-align: center; color: white;'>
            <h1 style='font-size: 40px; margin: 0;'>💬</h1>
            <h3>AI Chatbot</h3>
            <p>Context-aware medical Q&A with multilingual support.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Chatbot →", key="card_chatbot", use_container_width=True):
            st.session_state.active_page = "💬 Ask AI"
            st.rerun()
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    padding: 25px; border-radius: 15px; text-align: center; color: white;'>
            <h1 style='font-size: 40px; margin: 0;'>📊</h1>
            <h3>Dashboard</h3>
            <p>Track patient scans, view analytics, and monitor trends.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Dashboard →", key="card_dashboard", use_container_width=True):
            st.session_state.active_page = "📊 Dashboard"
            st.rerun()

    st.markdown("---")
    st.markdown("### ✨ What Makes This Special")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🧠 Grad-CAM XAI**\nSee exactly where the AI looked to make its decision.")
    c2.markdown("**🎯 Test-Time Augmentation**\n8 augmented views averaged for robust predictions.")
    c3.markdown("**✂️ Hair Removal**\nDullRazor preprocessing removes hair artifacts automatically.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🔍 Similar Cases (CBIR)**\nFind visually similar past cases using cosine similarity.")
    c2.markdown("**📈 Longitudinal Tracking**\nMonitor lesion progression over time per patient.")
    c3.markdown("**🛡️ Offline OOD Detection**\nValidate images even without internet connection.")

    scans_df = get_all_scans()
    if len(scans_df) > 0:
        st.markdown("---")
        st.markdown("### 📈 Platform Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans", len(scans_df))
        high_risk = len(scans_df[scans_df["risk_level"] == "HIGH"]) if "risk_level" in scans_df.columns else 0
        c2.metric("High Risk", high_risk)
        avg_conf = scans_df["confidence"].mean() if "confidence" in scans_df.columns else 0
        c3.metric("Avg Confidence", f"{avg_conf:.1f}%")
        c4.metric("Languages", len(SUPPORTED_LANGUAGES))


# ===================================================================
#                        PAGE: PREDICTION
# ===================================================================
if page == "🔬 Prediction":
    st.markdown("## 🔬 Skin Lesion Analysis")

    # Patient Info
    st.markdown("### 👤 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        patient_id = st.text_input("Patient ID")
        contact = st.text_input("Contact Number")
        exam_date = st.date_input("Date of Examination")

    # Multi-Modal Metadata (Feature 2)
    st.markdown("### 📋 Clinical Metadata (Multi-Modal Input)")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        body_location = st.selectbox("Body Location", BODY_LOCATIONS)
    with mc2:
        skin_type = st.selectbox("Fitzpatrick Skin Type", list(FITZPATRICK_TYPES.keys()))
    with mc3:
        lesion_size = st.number_input("Lesion Size (mm)", 0.0, 100.0, 5.0)
    with mc4:
        lesion_changed = st.selectbox("Lesion Changed Recently?", ["No", "Yes — Growing", "Yes — Color Change", "Yes — Shape Change"])

    # Fairness Disclaimer (Feature 4)
    if "V" in skin_type or "VI" in skin_type:
        st.warning("⚖️ **Fairness Notice:** This model has limited training data for darker skin tones "
                   "(Fitzpatrick V–VI). Results should be verified by a specialist experienced with diverse skin types.")

    # Image Upload
    st.markdown("### 📸 Upload Skin Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    camera_image = st.camera_input("Or Capture Image")
    img = uploaded_file if uploaded_file else camera_image

    if img:
        image = Image.open(img).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze", use_container_width=True):
            metadata = {
                "age": age, "gender": gender, "body_location": body_location,
                "skin_type": skin_type, "lesion_size": str(lesion_size),
                "lesion_changed": lesion_changed
            }

            # OOD Detection — cloud or local (Feature 7)
            if not st.session_state.use_local_model and GEMINI_API_KEY:
                with st.spinner("🛡️ Validating image (Cloud OOD Detection)..."):
                    is_skin, ood_reason = check_ood(image)
                if not is_skin:
                    st.error(f"⚠️ **Invalid Image Detected!** This doesn't appear to be a skin lesion.\n\nReason: {ood_reason}")
                    st.stop()
                else:
                    st.success("✅ Image validated as skin lesion.")
            elif st.session_state.use_local_model:
                with st.spinner("🛡️ Validating image (Offline OOD Detection)..."):
                    is_skin, ood_reason = check_ood_local(image)
                if not is_skin:
                    st.error(f"⚠️ **Invalid Image Detected (Local Check)!**\n\nReason: {ood_reason}")
                    st.stop()
                else:
                    st.success("✅ Image passes local validation.")

            # Hair Removal Preview (Feature 5)
            if st.session_state.get("enable_hair_removal", False):
                hr_img = np.array(image.resize((224, 224)))
                cleaned = remove_hair(hr_img)
                st.session_state.last_hair_removed = cleaned

            # Run Prediction
            with st.spinner("🧠 Analyzing image..."):
                if st.session_state.use_local_model:
                    result, predicted, cancer_prob, confidence = predict_image_local(image, metadata)
                else:
                    result, predicted, cancer_prob, confidence = predict_image_cloud(image, metadata)

            if result:
                st.session_state.last_result = result
                st.session_state.last_patient_data = {
                    "name": name, "age": age, "gender": gender,
                    "patient_id": patient_id, "contact": contact,
                    "exam_date": str(exam_date), "body_location": body_location,
                    "skin_type": skin_type, "lesion_size": str(lesion_size),
                    "lesion_changed": lesion_changed
                }
                st.session_state.last_image = image

                # Generate Grad-CAM
                local_model = load_local_model()
                feature_vec = None
                if local_model:
                    img_arr = np.expand_dims(np.array(image.resize((224,224)))/255.0, axis=0)
                    gcam_meta = None
                    if is_dual_input_model(local_model):
                        model_artifacts = load_model_artifacts()
                        gcam_meta = build_metadata_tensor(metadata, model_artifacts)
                    heatmap = generate_gradcam(local_model, img_arr, meta_array=gcam_meta)
                    if heatmap is not None:
                        gradcam_img = overlay_gradcam(image, heatmap)
                        st.session_state.last_gradcam = gradcam_img
                    else:
                        st.session_state.last_gradcam = None

                    # Feature 6: Extract feature vector for CBIR
                    feature_vec = extract_feature_vector(local_model, img_arr, meta_array=gcam_meta)
                    st.session_state.last_feature_vector = feature_vec
                else:
                    st.session_state.last_gradcam = None
                    st.session_state.last_feature_vector = None

                save_scan(st.session_state.last_patient_data, result,
                         feature_vector=feature_vec, image_thumbnail=image)
                st.toast("✅ Scan saved!", icon="💾")

    # Display Results
    if st.session_state.last_result:
        result = st.session_state.last_result
        predicted = result.get("predicted_class", "Unknown")
        confidence = result.get("confidence", 0.0)
        cancer_prob = result.get("cancer_prob", 0.0)
        all_probs = result.get("all_probs", {})
        reasoning_en = result.get("reasoning", "")
        recommendation_en = result.get("recommendation", "")
        uncertainty = result.get("uncertainty", "")
        triage = get_triage(cancer_prob)

        st.markdown("---")

        # Inference Mode Badge (Feature 8)
        mode = result.get("inference_mode", "cloud")
        if mode == "local":
            st.markdown("🔒 **Processed locally — no data sent to cloud.**")
        else:
            st.markdown("☁️ **Processed via Gemini Cloud AI.**")

        # Triage Panel (Feature 6)
        st.markdown(f"""
        <div style='background: {triage['color']}22; border-left: 5px solid {triage['color']};
                    padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <h3>{triage['icon']} {triage['label']}</h3>
            <p><b>Diagnosis:</b> {predicted} ({confidence:.1f}% confidence)</p>
            <p><b>Action:</b> {triage['action']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Multi-Class Probability Chart (Feature 7)
        if all_probs:
            st.markdown("### 📊 Classification Probabilities")
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = list(all_probs.keys())
            probs = [v * 100 for v in all_probs.values()]
            cancer_list = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
            bar_colors = ['#ff6b6b' if c in cancer_list else '#51cf66' for c in classes]
            ax.barh(classes, probs, color=bar_colors)
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Uncertainty (Feature 5)
        if uncertainty:
            st.markdown("### 🎯 Uncertainty Estimation (MC Dropout)")
            st.info(f"**{uncertainty}**")

        # Grad-CAM Display (Feature 1)
        gcam = st.session_state.get("last_gradcam")
        if gcam is not None:
            st.markdown("### 🧠 Explainable AI — Grad-CAM Heatmap")
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(st.session_state.get("last_image"), caption="Original", use_container_width=True)
            with gc2:
                st.image(gcam, caption="Grad-CAM Overlay", use_container_width=True)
            st.caption("The heatmap highlights the regions the AI focused on for its prediction.")

        # Hair Removal Preview (Feature 5)
        hr_img = st.session_state.get("last_hair_removed")
        if hr_img is not None:
            st.markdown("### ✂️ Hair Removal Preprocessing")
            hr1, hr2 = st.columns(2)
            with hr1:
                st.image(np.array(st.session_state.get("last_image", image).resize((224,224))), caption="Before", use_container_width=True)
            with hr2:
                st.image(hr_img, caption="After Hair Removal", use_container_width=True)

        # TTA Badge
        if st.session_state.get("enable_tta", False) and mode == "local":
            st.success("🎯 **Test-Time Augmentation (TTA) was used** — 8 augmented views averaged for robust prediction.")

        # Reasoning
        st.info(f"**Reasoning:** {reasoning_en}")
        st.warning(f"**Recommendation:** {recommendation_en}")

        # Similar Cases (Feature 6)
        fv = st.session_state.get("last_feature_vector")
        if fv is not None:
            similar = find_similar_cases(fv, top_k=3)
            if similar:
                st.markdown("### 🔍 Similar Past Cases (CBIR)")
                sim_cols = st.columns(len(similar))
                for i, case in enumerate(similar):
                    with sim_cols[i]:
                        if case.get("thumbnail"):
                            try:
                                thumb_arr = cv2.imdecode(np.frombuffer(case["thumbnail"], np.uint8), cv2.IMREAD_COLOR)
                                thumb_arr = cv2.cvtColor(thumb_arr, cv2.COLOR_BGR2RGB)
                                st.image(thumb_arr, caption=f"Case #{case['id']}", use_container_width=True)
                            except Exception:
                                st.write(f"Case #{case['id']}")
                        st.markdown(f"**{case['predicted_class']}**")
                        st.markdown(f"Confidence: {case['confidence']:.1f}%")
                        st.markdown(f"Similarity: {case['similarity']*100:.1f}%")
                        st.markdown(f"Risk: {case['risk_level']}")

        # Translation
        lang = st.session_state.selected_language
        if lang != "English":
            st.markdown(f"### 🌐 Translation ({lang})")
            with st.spinner(f"Translating to {lang}..."):
                tr_reasoning = translate_text(reasoning_en, lang)
                tr_recommendation = translate_text(recommendation_en, lang)
            st.info(f"**Reasoning ({lang}):** {tr_reasoning}")
            st.warning(f"**Recommendation ({lang}):** {tr_recommendation}")

        # PDF Report
        if "last_patient_data" in st.session_state and "last_image" in st.session_state:
            gradcam_img = st.session_state.get("last_gradcam")
            pdf = generate_pdf(st.session_state.last_patient_data, st.session_state.last_image, result, gradcam_img)
            with open(pdf, "rb") as f:
                st.download_button("📄 Download Full Report", f, file_name=pdf, use_container_width=True)
# ===================================================================
#                        PAGE: DASHBOARD
# ===================================================================
if page == "📊 Dashboard":
    st.markdown("## 📊 Patient History Dashboard")
    scans_df = get_all_scans()

    if len(scans_df) == 0:
        st.info("No scans recorded yet. Go to 🔬 Prediction to analyze your first image!")
    else:
        st.markdown("### 📈 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans", len(scans_df))
        high_risk = len(scans_df[scans_df["risk_level"] == "HIGH"]) if "risk_level" in scans_df.columns else 0
        c2.metric("🔴 High Risk", high_risk)
        medium_risk = len(scans_df[scans_df["risk_level"] == "MODERATE"]) if "risk_level" in scans_df.columns else 0
        c3.metric("🟡 Moderate", medium_risk)
        low_risk = len(scans_df[scans_df["risk_level"] == "LOW"]) if "risk_level" in scans_df.columns else 0
        c4.metric("🟢 Low Risk", low_risk)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧪 Diagnosis Distribution")
            diagnosis_counts = scans_df["predicted_class"].value_counts()
            fig1, ax1 = plt.subplots()
            colors_list = ['#ff6b6b','#51cf66','#ffd43b','#339af0','#845ef7','#ff922b','#20c997','#e64980','#adb5bd']
            ax1.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%',
                    colors=colors_list[:len(diagnosis_counts)], startangle=90)
            ax1.set_title("Diagnosis Breakdown")
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.markdown("### 📅 Scans Over Time")
            scans_df["scan_date"] = pd.to_datetime(scans_df["scan_timestamp"]).dt.date
            daily_counts = scans_df.groupby("scan_date").size().reset_index(name="count")
            fig2, ax2 = plt.subplots()
            ax2.bar(daily_counts["scan_date"].astype(str), daily_counts["count"], color="#667eea")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Scans")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 🎯 Confidence Distribution")
            fig3, ax3 = plt.subplots()
            ax3.hist(scans_df["confidence"], bins=20, color="#f093fb", edgecolor="white")
            ax3.set_xlabel("Confidence (%)")
            ax3.set_ylabel("Frequency")
            st.pyplot(fig3)
            plt.close()

        with col4:
            st.markdown("### ⚠️ Risk Level Distribution")
            risk_counts = scans_df["risk_level"].value_counts()
            fig4, ax4 = plt.subplots()
            risk_colors = {"LOW": "#51cf66", "MODERATE": "#ffd43b", "HIGH": "#ff6b6b"}
            bar_colors = [risk_colors.get(r, "#aaa") for r in risk_counts.index]
            ax4.bar(risk_counts.index, risk_counts.values, color=bar_colors)
            ax4.set_xlabel("Risk Level")
            ax4.set_ylabel("Count")
            st.pyplot(fig4)
            plt.close()

        st.markdown("---")
        st.markdown("### 🗂️ Scan History")
        search_query = st.text_input("🔍 Search by Patient Name or ID")
        filtered = scans_df
        if search_query:
            filtered = scans_df[
                scans_df["name"].str.contains(search_query, case=False, na=False) |
                scans_df["patient_id"].str.contains(search_query, case=False, na=False)
            ]
        display_cols = ["name","age","gender","patient_id","predicted_class",
                       "confidence","risk_level","body_location","skin_type","inference_mode","scan_timestamp"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True)

# ===================================================================
#                        PAGE: TRACKING (Feature 8)
# ===================================================================
if page == "📈 Tracking":
    st.markdown("## 📈 Longitudinal Lesion Tracking")
    st.markdown("Track a patient's skin lesion history over time to monitor changes.")

    scans_df = get_all_scans()
    if len(scans_df) == 0:
        st.info("No scans recorded yet. Go to 🔬 Prediction to analyze your first image!")
    else:
        # Get unique patient IDs
        patient_ids = scans_df[scans_df["patient_id"].str.strip() != ""]["patient_id"].unique().tolist()
        patient_names = scans_df[scans_df["name"].str.strip() != ""]["name"].unique().tolist()

        search_method = st.radio("Search by:", ["Patient ID", "Patient Name"], horizontal=True)
        if search_method == "Patient ID":
            if patient_ids:
                selected = st.selectbox("Select Patient ID", patient_ids)
                patient_scans = scans_df[scans_df["patient_id"] == selected].sort_values("scan_timestamp")
            else:
                st.warning("No patient IDs found in scan history.")
                patient_scans = pd.DataFrame()
        else:
            if patient_names:
                selected = st.selectbox("Select Patient Name", patient_names)
                patient_scans = scans_df[scans_df["name"] == selected].sort_values("scan_timestamp")
            else:
                st.warning("No patient names found in scan history.")
                patient_scans = pd.DataFrame()

        if len(patient_scans) > 0:
            st.markdown(f"### Found **{len(patient_scans)}** scans")

            # Trend indicators
            if len(patient_scans) >= 2:
                first_prob = patient_scans.iloc[0]["cancer_prob"]
                last_prob = patient_scans.iloc[-1]["cancer_prob"]
                diff = last_prob - first_prob
                if diff > 0.05:
                    trend = "↗️ **Increasing Risk** — Cancer probability is trending upward."
                    trend_color = "#ff6b6b"
                elif diff < -0.05:
                    trend = "↘️ **Decreasing Risk** — Cancer probability is trending downward."
                    trend_color = "#51cf66"
                else:
                    trend = "➡️ **Stable** — Cancer probability is relatively stable."
                    trend_color = "#ffd43b"
                st.markdown(f"<div style='background: {trend_color}22; border-left: 4px solid {trend_color}; padding: 15px; border-radius: 8px; margin: 10px 0;'>{trend}</div>", unsafe_allow_html=True)

            # Metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Scans", len(patient_scans))
            mc2.metric("Latest Diagnosis", patient_scans.iloc[-1]["predicted_class"])
            mc3.metric("Latest Risk", patient_scans.iloc[-1]["risk_level"])
            mc4.metric("Avg Confidence", f"{patient_scans['confidence'].mean():.1f}%")

            # Cancer probability over time chart
            st.markdown("### 📉 Cancer Probability Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            timestamps = pd.to_datetime(patient_scans["scan_timestamp"])
            ax.plot(timestamps, patient_scans["cancer_prob"] * 100, 'o-', color='#ff6b6b', linewidth=2, markersize=8, label="Cancer Prob %")
            ax.axhline(y=30, color='#ffd43b', linestyle='--', alpha=0.7, label="Moderate Threshold (30%)")
            ax.axhline(y=60, color='#ff6b6b', linestyle='--', alpha=0.7, label="High Risk Threshold (60%)")
            ax.fill_between(timestamps, 0, 30, alpha=0.05, color='green')
            ax.fill_between(timestamps, 30, 60, alpha=0.05, color='yellow')
            ax.fill_between(timestamps, 60, 100, alpha=0.05, color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Cancer Probability (%)")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Confidence over time
            st.markdown("### 🎯 Model Confidence Over Time")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.bar(timestamps.dt.strftime("%Y-%m-%d %H:%M"), patient_scans["confidence"], color="#667eea", alpha=0.8)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Confidence (%)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Scan history table
            st.markdown("### 🗂️ Scan Details")
            display_cols = ["scan_timestamp", "predicted_class", "confidence", "cancer_prob", "risk_level", "body_location", "inference_mode"]
            available = [c for c in display_cols if c in patient_scans.columns]
            st.dataframe(patient_scans[available], use_container_width=True)

            # Visual progression with thumbnails
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            scan_ids = patient_scans["id"].tolist() if "id" in patient_scans.columns else []
            if scan_ids:
                st.markdown("### 🖼️ Visual Progression")
                thumb_cols = st.columns(min(len(scan_ids), 6))
                for i, sid in enumerate(scan_ids[:6]):
                    c.execute("SELECT image_thumbnail, scan_timestamp, predicted_class FROM scans WHERE id=?", (int(sid),))
                    row = c.fetchone()
                    if row and row[0]:
                        try:
                            thumb = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
                            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                            with thumb_cols[i % len(thumb_cols)]:
                                st.image(thumb, caption=f"{row[1][:10]}\n{row[2]}", use_container_width=True)
                        except Exception:
                            pass
            conn.close()
        elif search_method == "Patient ID" and patient_ids or search_method == "Patient Name" and patient_names:
            st.info("No scans found for this patient.")


# ===================================================================
#                        PAGE: AI CHATBOT
# ===================================================================
if page == "💬 Ask AI":
    st.markdown("## 💬 AI Medical Assistant")
    st.markdown("Ask follow-up questions about your diagnosis or general skin health.")

    if not GEMINI_API_KEY:
        st.warning("Please enter your Gemini API Key in the sidebar to use the chatbot.")
    else:
        if st.session_state.last_result:
            last = st.session_state.last_result
            st.markdown(
                f"""<div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                            padding: 15px; border-radius: 10px; border-left: 4px solid #667eea; margin-bottom: 20px;'>
                    <b>📋 Last Diagnosis Context:</b><br>
                    <b>Condition:</b> {last.get('predicted_class', 'N/A')}<br>
                    <b>Confidence:</b> {last.get('confidence', 0):.1f}%<br>
                    <b>Reasoning:</b> {last.get('reasoning', 'N/A')}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("💡 No diagnosis yet. Go to 🔬 Prediction first for context-aware responses.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask about your diagnosis or skin health...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            context = ""
            if st.session_state.last_result:
                r = st.session_state.last_result
                context = f"""
Latest AI skin analysis:
- Condition: {r.get('predicted_class', 'Unknown')}
- Confidence: {r.get('confidence', 0)}%
- Cancer Probability: {r.get('cancer_prob', 0)*100:.1f}%
- Reasoning: {r.get('reasoning', '')}
- Recommendation: {r.get('recommendation', '')}
- Uncertainty: {r.get('uncertainty', 'N/A')}
"""
            system_prompt = f"""You are a helpful, empathetic AI dermatology assistant.
{context}
Rules: Be compassionate, use simple language, always disclaim you are AI not a doctor,
use diagnosis context if relevant, redirect non-medical questions politely."""

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        chat_messages = [system_prompt]
                        for msg in st.session_state.chat_history[-6:]:
                            chat_messages.append(f"{msg['role']}: {msg['content']}")
                        response = model.generate_content("\n\n".join(chat_messages))
                        ai_response = response.text.strip()

                        lang = st.session_state.selected_language
                        if lang != "English":
                            translated = translate_text(ai_response, lang)
                            st.markdown(ai_response)
                            st.markdown(f"---\n**🌐 {lang}:**\n{translated}")
                            ai_response = ai_response + f"\n\n---\n🌐 {lang}:\n{translated}"
                        else:
                            st.markdown(ai_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ===================================================================
#                        PAGE: FIND CLINICS
# ===================================================================
if page == "🗺️ Find Clinics":
    st.markdown("## 🗺️ Find Nearby Dermatology Clinics")
    st.markdown("Locate dermatologists, skin specialists, and derma clinics near you.")

    # Location Input
    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        city_input = st.text_input("🔍 Enter your City or Area", placeholder="e.g. Greater Noida, Delhi, Mumbai...")
    with loc_col2:
        search_radius = st.selectbox("Radius", ["5 km", "10 km", "20 km", "50 km"], index=1)

    if city_input and st.button("🔎 Search Dermatology Clinics", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("Please enter your Gemini API Key in the sidebar to search for clinics.")
        else:
            with st.spinner(f"🔍 Searching for dermatology clinics near {city_input}..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""Find 8-12 real dermatology clinics, skin specialists, or skin cancer hospitals near {city_input}, India within {search_radius}.

Return ONLY a valid JSON array. Each object must have:
- "name": clinic/hospital name
- "address": full address
- "lat": latitude (float)
- "lon": longitude (float)
- "type": one of "Dermatology Clinic", "Skin Hospital", "Multi-Specialty Hospital", "Cosmetic Dermatology"
- "phone": phone number or "N/A"
- "rating": rating out of 5 (float) or null
- "specialties": short list of specialties like "Skin Cancer, Moles, Psoriasis"

Return ONLY the JSON array, no markdown, no explanation. Use real, accurate coordinates for the {city_input} area."""
                    response = model.generate_content(prompt)
                    text = response.text.strip()
                    if text.startswith("```json"): text = text[7:]
                    if text.startswith("```"): text = text[3:]
                    if text.endswith("```"): text = text[:-3]
                    clinics = json.loads(text.strip())
                    st.session_state.found_clinics = clinics
                    st.session_state.clinic_city = city_input
                except Exception as e:
                    st.error(f"Error searching for clinics: {str(e)}")
                    st.session_state.found_clinics = []

    # Display Results
    if "found_clinics" in st.session_state and st.session_state.found_clinics:
        clinics = st.session_state.found_clinics
        city = st.session_state.get("clinic_city", "your area")

        st.success(f"Found **{len(clinics)}** dermatology clinics near {city}")

        # Calculate map center
        avg_lat = sum(c.get("lat", 28.47) for c in clinics) / len(clinics)
        avg_lon = sum(c.get("lon", 77.50) for c in clinics) / len(clinics)

        # Create Folium Map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="OpenStreetMap")

        # Color by type
        type_colors = {
            "Dermatology Clinic": "blue",
            "Skin Hospital": "red",
            "Multi-Specialty Hospital": "green",
            "Cosmetic Dermatology": "purple",
        }
        type_icons = {
            "Dermatology Clinic": "stethoscope",
            "Skin Hospital": "hospital-o",
            "Multi-Specialty Hospital": "plus-square",
            "Cosmetic Dermatology": "star",
        }

        for clinic in clinics:
            lat = clinic.get("lat", avg_lat)
            lon = clinic.get("lon", avg_lon)
            name = clinic.get("name", "Unknown")
            address = clinic.get("address", "")
            phone = clinic.get("phone", "N/A")
            c_type = clinic.get("type", "Dermatology Clinic")
            rating = clinic.get("rating")
            specialties = clinic.get("specialties", "")

            rating_str = f"⭐ {rating}/5" if rating else "No rating"
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <h4 style='margin: 0; color: #1a73e8;'>{name}</h4>
                <p style='margin: 4px 0; font-size: 12px; color: #555;'>{address}</p>
                <p style='margin: 4px 0; font-size: 12px;'>📞 {phone}</p>
                <p style='margin: 4px 0; font-size: 12px;'>🏥 {c_type}</p>
                <p style='margin: 4px 0; font-size: 12px;'>{rating_str}</p>
                <p style='margin: 4px 0; font-size: 11px; color: #777;'>🔬 {specialties}</p>
                <a href='https://www.google.com/maps/dir/?api=1&destination={lat},{lon}' target='_blank'
                   style='font-size: 12px; color: #1a73e8;'>📍 Get Directions</a>
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=name,
                icon=folium.Icon(
                    color=type_colors.get(c_type, "gray"),
                    icon=type_icons.get(c_type, "info-sign"),
                    prefix="fa"
                )
            ).add_to(m)

        # Display the map
        st_folium(m, width=None, height=500, use_container_width=True)

        # Legend
        st.markdown("""        <div style='display: flex; gap: 20px; justify-content: center; margin: 10px 0; flex-wrap: wrap;'>
            <span>🔵 Dermatology Clinic</span>
            <span>🔴 Skin Hospital</span>
            <span>🟢 Multi-Specialty Hospital</span>
            <span>🟣 Cosmetic Dermatology</span>
        </div>
        """, unsafe_allow_html=True)

        # Clinic Cards
        st.markdown("### 🏥 Clinic Details")
        for i, clinic in enumerate(clinics):
            c_type = clinic.get("type", "Clinic")
            rating = clinic.get("rating")
            rating_str = f"⭐ {rating}/5" if rating else ""
            type_color = {"Dermatology Clinic": "#339af0", "Skin Hospital": "#ff6b6b",
                         "Multi-Specialty Hospital": "#51cf66", "Cosmetic Dermatology": "#845ef7"}.get(c_type, "#aaa")

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {type_color}11, {type_color}22);
                        border-left: 4px solid {type_color}; padding: 15px; border-radius: 10px; margin: 8px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <div>
                        <h4 style='margin: 0;'>{clinic.get('name', 'Unknown')}</h4>
                        <p style='margin: 4px 0; color: #666; font-size: 14px;'>📍 {clinic.get('address', '')}</p>
                        <p style='margin: 4px 0; font-size: 13px;'>📞 {clinic.get('phone', 'N/A')} &nbsp;|&nbsp; 🏥 {c_type}</p>
                        <p style='margin: 4px 0; font-size: 13px; color: #888;'>🔬 {clinic.get('specialties', '')}</p>
                    </div>
                    <div style='text-align: right;'>
                        <span style='font-size: 16px;'>{rating_str}</span><br>
                        <a href='https://www.google.com/maps/dir/?api=1&destination={clinic.get("lat",0)},{clinic.get("lon",0)}'
                           target='_blank' style='font-size: 13px; color: {type_color};'>Get Directions →</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif "found_clinics" in st.session_state and not st.session_state.found_clinics:
        st.info("No clinics found. Try a different city or area.")
    else:
        # Default helpful info when no search performed
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                    padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h3>🔍 How to use</h3>
            <p>Enter your city or area name above and click <b>Search</b> to find dermatology clinics near you.</p>
            <p style='color: #888; font-size: 13px;'>Powered by Gemini AI • Results include clinic locations, contact info, and Google Maps directions</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🩺 When should you visit a dermatologist?")
        w1, w2, w3 = st.columns(3)
        w1.markdown("""**🔴 Urgently**\n- Rapidly changing mole\n- Bleeding lesion\n- AI flagged HIGH risk""")
        w2.markdown("""**🟡 Soon (2 weeks)**\n- New or unusual growth\n- Persistent itching\n- AI flagged MODERATE risk""")
        w3.markdown("""**🟢 Routine**\n- Annual skin check\n- Family history of cancer\n- Self-monitoring""")


# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey;'>
    <p>© 2026 AI Medical Assistant v6.0 | Gautam Buddha University</p>
    <p style='font-size: 12px;'>🧬 Multi-Class • 🧠 Grad-CAM • 🎯 TTA • ✂️ Hair Removal • 🔍 CBIR • 🛡️ OOD • 📈 Tracking • 🗺️ Clinic Finder • 🌐 Multi-Language</p>
</div>
""", unsafe_allow_html=True)
