from flask import Flask, render_template, request
import os
import torch
import numpy as np
import cv2

# Local imports
from model import load_models, load_unet, predict
from explain import generate_gradcam
from shap_explaination import generate_shap

app = Flask(__name__)

DEVICE = torch.device("cpu")

# ===== LOAD MODELS =====
mu, mm = load_models()
unet = load_unet()

UPLOAD_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== SCALING CONSTANTS (APPROX ISIC 2019) =====
# These match the StandardScaler used during training
ISIC_MEAN = 51.5
ISIC_STD = 16.5

# ===== CLASS INFO =====
CLASS_INFO = {
    "AK": {"name": "Actinic Keratosis", "desc": "Rough, scaly patch caused by sun exposure.", "risk": "Medium", "action": "Consult dermatologist.", "link": "https://dermnetnz.org/topics/actinic-keratosis"},
    "BCC": {"name": "Basal Cell Carcinoma", "desc": "Slow-growing skin cancer.", "risk": "Low-Medium", "action": "Seek medical evaluation.", "link": "https://dermnetnz.org/topics/basal-cell-carcinoma"},
    "BKL": {"name": "Benign Keratosis", "desc": "Non-cancerous growth.", "risk": "Low", "action": "Usually harmless.", "link": "https://dermnetnz.org/topics/seborrhoeic-keratosis"},
    "DF": {"name": "Dermatofibroma", "desc": "Benign skin nodule.", "risk": "Low", "action": "No treatment needed.", "link": "https://dermnetnz.org/topics/dermatofibroma"},
    "MEL": {"name": "Melanoma", "desc": "Aggressive skin cancer.", "risk": "High", "action": "Immediate consultation required.", "link": "https://dermnetnz.org/topics/melanoma"},
    "NV": {"name": "Melanocytic Nevus", "desc": "Common mole.", "risk": "Low", "action": "Monitor changes.", "link": "https://dermnetnz.org/topics/melanocytic-naevus"},
    "SCC": {"name": "Squamous Cell Carcinoma", "desc": "Can spread if untreated.", "risk": "Medium-High", "action": "Consult dermatologist.", "link": "https://dermnetnz.org/topics/squamous-cell-carcinoma"},
    "VASC": {"name": "Vascular Lesion", "desc": "Blood vessel abnormality.", "risk": "Low", "action": "Usually harmless.", "link": "https://dermnetnz.org/topics/vascular-lesions"}
}

FEATURE_NAMES = ["age", "sex", "torso_ant", "head_neck", "torso_lat", "lower_ext", "oral_genital", "palms_soles", "torso_post", "upper_ext"]
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
SITE_MAP = {"anterior torso": 0, "head/neck": 1, "lateral torso": 2, "lower extremity": 3, "oral/genital": 4, "palms/soles": 5, "posterior torso": 6, "upper extremity": 7}

# ===== HELPERS =====
def save_mask(mask, path):
    img = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def save_gradcam(base_img, gradcam, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(path, overlay)

def save_segmented(img_path, mask, path):
    img = cv2.resize(cv2.imread(img_path), (224, 224))
    m = np.stack([mask.squeeze().cpu().numpy()]*3, axis=-1)
    cv2.imwrite(path, (img * m).astype(np.uint8))

# ===== ROUTE =====
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        age = float(request.form["age"])
        sex = request.form["sex"]
        site = request.form["site"]

        img_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
        file.save(img_path)

        # ===== META PREP (SCALED) =====
        z_age = (age - ISIC_MEAN) / ISIC_STD  # Matches training StandardScaler
        sex_val = 0 if sex == "male" else 1
        one_hot = np.zeros(8)
        one_hot[SITE_MAP[site]] = 1
        meta_input = np.array([z_age, sex_val] + one_hot.tolist(), dtype=np.float32)

        # ===== PREDICT =====
        pred, probs, mask, img_tensor, meta_tensor = predict(
            img_path, age, sex, mu, mm, unet, meta_override=meta_input
        )

        # ===== SAVE VISUALS =====
        mask_path = os.path.join(UPLOAD_FOLDER, "mask.png")
        save_mask(mask, mask_path)

        gradcam = generate_gradcam(mu, img_tensor, meta_tensor)
        base_img = cv2.resize(cv2.imread(img_path), (224, 224))
        gradcam_path = os.path.join(UPLOAD_FOLDER, "gradcam.png")
        save_gradcam(base_img, gradcam, gradcam_path)

        seg_path = os.path.join(UPLOAD_FOLDER, "segmented.png")
        save_segmented(img_path, mask, seg_path)

        # ===== SHAP =====
        shap_vals = generate_shap(mu, mm, img_tensor, mask, meta_tensor)
        shap_vals = np.array(shap_vals).flatten()
        shap_norm = shap_vals / (np.max(np.abs(shap_vals)) + 1e-8)
        
        idx = np.argsort(np.abs(shap_norm))[-5:][::-1]
        top_vals = shap_norm[idx].tolist()
        top_names = [FEATURE_NAMES[i] for i in idx]

        # ===== CONSTRUCT RESULT =====
        result = {
            "prediction": pred,
            "probs": probs.tolist(),
            "input": img_path,
            "mask": mask_path,
            "gradcam": gradcam_path,
            "segmented": seg_path,
            "shap": top_vals,
            "feature_names": top_names,
            "class_names": CLASS_NAMES,
            "info": CLASS_INFO[pred]
        }

        # ===== TERMINAL REPORT =====
        print("\n" + "="*55)
        print("ANALYSIS REPORT ")
        print("="*55)
        print(f"INPUT: Age {int(age)} (Z-score: {z_age:.2f}) | Site: {site}")
        print(f"PREDICTION: {pred} ({CLASS_INFO[pred]['name']})")
        print("-" * 35)
        print("CONFIDENCE SCORES:")
        for c_name, p_val in zip(CLASS_NAMES, result['probs']):
            bar = "█" * int(p_val * 20)
            print(f"  {c_name:4}: {p_val:.4f} {bar}")
        print("-" * 35)
        print("METADATA INFLUENCES (SHAP):")
        for f_name, s_val in zip(top_names, top_vals):
            impact = "(+)" if s_val > 0 else "(-)"
            print(f"  {f_name:12}: {s_val:.4f} {impact}")
        print("="*55 + "\n")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
