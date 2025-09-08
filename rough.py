import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ===== CONFIG =====
MODEL_PATH = "Pneumo_model.pth"  # Your saved model
INPUT_FOLDER = r"E:\imgs_x_rays"  # Folder with chest X-rays
OUTPUT_FOLDER = r"E:\outputs"  # Where to save processed results
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ===== IMAGE TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== LOAD MODEL =====
print("[INFO] Loading model...")
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully!")


# ===== PREDICT FUNCTION =====
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
    return CLASS_NAMES[predicted_idx], probs[0][predicted_idx].item() * 100, predicted_idx


# ===== HEATMAP & MULTI-BBOX FUNCTION =====
def generate_heatmap_and_boxes(image_path, target_class):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    target_layer = model.features[6]

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Generate heatmap
    heatmap = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # Threshold CAM to find areas of high activation
    cam_threshold = (grayscale_cam * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(cam_threshold, 180, 255, cv2.THRESH_BINARY)

    # Find multiple contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    if len(contours) > 0:
        print(f"[INFO] Detected {len(contours)} potential infected regions.")
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:  # Filter out very tiny noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Blend heatmap + bounding boxes
    final_output = cv2.addWeighted(img_bgr, 0.7, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR), 0.5, 0)
    return final_output


# ===== PROCESS FOLDER =====
def process_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        print("[ERROR] No images found in the folder!")
        return

    print(f"[INFO] Found {len(files)} images. Starting analysis...")

    for file_name in files:
        image_path = os.path.join(folder_path, file_name)
        print(f"\n[INFO] Processing: {file_name}")

        label, confidence, pred_idx = predict_image(image_path)
        print(f"[RESULT] Prediction: {label} ({confidence:.2f}%)")

        if label == "PNEUMONIA":
            final_output = generate_heatmap_and_boxes(image_path, target_class=pred_idx)
        else:
            # Just show original for normal case
            final_output = cv2.imread(image_path)
            final_output = cv2.resize(final_output, (224, 224))

        # Save result
        output_path = os.path.join(OUTPUT_FOLDER, f"result_{file_name}")
        cv2.imwrite(output_path, final_output)
        print(f"[INFO] Saved result to {output_path}")

        # Display result
        cv2.imshow("Pneumonia Analysis", final_output)
        print("[INFO] Press any key for next image or 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("[INFO] Exiting early...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_folder(INPUT_FOLDER)
