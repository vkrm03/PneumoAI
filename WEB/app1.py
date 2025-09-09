import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter

# ==== CONFIG ====
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Pneumo_model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ==== IMAGE TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== LOAD MODEL ====
print("[INFO] Loading model...")
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully!")

# ==== HELPER ====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
    return CLASS_NAMES[predicted_idx], probs[0][predicted_idx].item() * 100, predicted_idx

# ==== LUNG REGION DETECTION ====
def get_lung_mask(img_gray):
    """Rough lung region mask to avoid false highlights outside lungs."""
    _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img_gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

# ==== HEATMAP & MULTI-REGION DETECTION ====
def generate_heatmap_and_boxes(image_path, target_class):
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Grad-CAM++
    target_layer = model.features[6]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Smooth the heatmap to reduce noise
    grayscale_cam = gaussian_filter(grayscale_cam, sigma=2)

    # Create masked heatmap only inside lungs
    img_gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    lung_mask = get_lung_mask(img_gray)
    grayscale_cam = np.where(lung_mask > 0, grayscale_cam, 0)

    # Save Grad-CAM heatmap
    heatmap = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    gradcam_path = os.path.join(GRADCAM_FOLDER, "gradcam_" + os.path.basename(image_path))
    cv2.imwrite(gradcam_path, cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Adaptive threshold for detection
    cam_uint8 = (grayscale_cam * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find multiple regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    # Sort contours by area (largest first) and keep top 3
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    region_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # filter out tiny noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"Region {region_count + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            region_count += 1

    print(f"[INFO] Detected {region_count} pneumonia regions")

    # Save final image with boxes
    box_path = os.path.join(GRADCAM_FOLDER, "box_" + os.path.basename(image_path))
    cv2.imwrite(box_path, img_bgr)

    return gradcam_path, box_path

# ==== ROUTES ====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            label, confidence, pred_idx = predict_image(file_path)

            gradcam_filename = None
            box_filename = None

            if label == "PNEUMONIA":
                gradcam_path, box_path = generate_heatmap_and_boxes(file_path, target_class=pred_idx)
                gradcam_filename = os.path.basename(gradcam_path)
                box_filename = os.path.basename(box_path)

            return redirect(url_for("result",
                                    filename=filename,
                                    label=label,
                                    confidence=confidence,
                                    gradcam=gradcam_filename,
                                    box=box_filename))
    return render_template("index.html")

@app.route("/result")
def result():
    filename = request.args.get("filename")
    label = request.args.get("label")
    confidence = request.args.get("confidence")
    gradcam = request.args.get("gradcam")
    box = request.args.get("box")

    return render_template("result.html",
                           filename=filename,
                           label=label,
                           confidence=confidence,
                           gradcam=gradcam,
                           box=box)

if __name__ == "__main__":
    app.run(debug=True)
