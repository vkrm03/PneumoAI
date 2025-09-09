import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter
import gradio as gr

# ==== CONFIG ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Pneumo_model.pth"  # place this file in the same folder
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ==== IMAGE TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== LOAD MODEL ====
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== HELPER FUNCTIONS ====
def predict_image(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
    return CLASS_NAMES[predicted_idx], probs[0][predicted_idx].item() * 100, predicted_idx

def get_lung_mask(img_gray):
    _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

def generate_heatmap_and_boxes(image: Image.Image, target_class):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Grad-CAM++
    target_layer = model.features[6]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    grayscale_cam = gaussian_filter(grayscale_cam, sigma=2)

    # Lung mask
    img_gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    lung_mask = get_lung_mask(img_gray)
    grayscale_cam = np.where(lung_mask > 0, grayscale_cam, 0)

    # Heatmap
    heatmap = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # Adaptive threshold for regions
    cam_uint8 = (grayscale_cam * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    # Draw top 3 regions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"Region {i+1}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return heatmap, img_bgr

# ==== GRADIO INTERFACE ====
def process_image(image: Image.Image):
    label, confidence, pred_idx = predict_image(image)
    if label == "PNEUMONIA":
        heatmap, box_image = generate_heatmap_and_boxes(image, pred_idx)
        return label, f"{confidence:.2f}%", heatmap, box_image
    else:
        return label, f"{confidence:.2f}%", None, None

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Image(label="Detected Regions")
    ],
    title="PneumoAI",
    description="Upload a chest X-ray image. PneumoAI predicts pneumonia and highlights affected regions."
)

if __name__ == "__main__":
    iface.launch()
