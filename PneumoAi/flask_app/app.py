from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==== CONFIG ====
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Dummy login credentials
DUMMY_USER = {
    "email": "admin@example.com",
    "password": "admin123"
}

# ==== MODEL CONFIG ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Pneumo_model.pth"  # ensure your model file is here
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

try:
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model load error:", e)
    raise SystemExit("Model not found or failed to load. Place Pneumo_model.pth correctly.")

# ==== HELPERS ====
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image.")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = torch.argmax(probs, dim=1).item()
    return CLASS_NAMES[idx], float(probs[0][idx].item() * 100), idx

def generate_heatmap(image_path, target_class):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    target_layer = model.features[6]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    heatmap = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    gradcam_filename = "gradcam_" + os.path.basename(image_path)
    gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return gradcam_filename

# ==== ROUTES ====
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")

        if email == DUMMY_USER["email"] and password == DUMMY_USER["password"]:
            session["user"] = email
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid email or password!")

    return render_template("login.html")

# LOGOUT
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# DASHBOARD
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

# UPLOAD & PREDICT
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file uploaded")
        file = request.files["file"]

        if file.filename == "" or not allowed_file(file.filename):
            return render_template("upload.html", error="Invalid file format")

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            label, confidence, idx = predict_image(file_path)
            gradcam_filename = generate_heatmap(file_path, idx)
        except Exception as e:
            os.remove(file_path)
            return render_template("upload.html", error=f"Image error: {e}")

        return render_template("result.html", label=label, confidence=confidence,
                               uploaded_file=filename, gradcam_file=gradcam_filename)

    return render_template("upload.html")

# RUN
if __name__ == "__main__":
    app.run(debug=True)
