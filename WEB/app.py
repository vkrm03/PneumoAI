import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image



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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()

    return CLASS_NAMES[predicted_idx], probs[0][predicted_idx].item() * 100

def generate_gradcam(image_path, target_class):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    last_conv_layer = model.features[-3][1]
    fwd_handle = last_conv_layer.register_forward_hook(forward_hook)
    bwd_handle = last_conv_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward(retain_graph=True)

    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    original_img = cv2.imread(image_path)
    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    gradcam_path = os.path.join(GRADCAM_FOLDER, os.path.basename(image_path))
    cv2.imwrite(gradcam_path, superimposed)

    fwd_handle.remove()
    bwd_handle.remove()

    return gradcam_path

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

            label, confidence = predict_image(file_path)
            gradcam_filename = None

            if label == "PNEUMONIA":
                gradcam_path = generate_gradcam(file_path, target_class=1)
                gradcam_filename = os.path.basename(gradcam_path)

            return redirect(url_for("result", filename=filename,
                                    label=label, confidence=confidence,
                                    gradcam=gradcam_filename))

    return render_template("index.html")

@app.route("/result")
def result():
    filename = request.args.get("filename")
    label = request.args.get("label")
    confidence = request.args.get("confidence")
    gradcam = request.args.get("gradcam")

    return render_template("result.html", filename=filename, label=label,
                           confidence=confidence, gradcam=gradcam)

if __name__ == "__main__":
    app.run(debug=True)
