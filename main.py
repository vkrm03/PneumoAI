import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os

MODEL_PATH = "Pneumo_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("[INFO] Model loaded successfully!")

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()

    predicted_label = class_names[predicted_idx]
    confidence = probs[0][predicted_idx].item() * 100

    plt.imshow(image)
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"[RESULT] The image is predicted as **{predicted_label}** with {confidence:.2f}% confidence")


test_image_path = "E:\DEVELOPMENT\Python_Codings\My_Projects\Large_projects\PneumoAI\evl\evl4.jpeg"
predict_image(test_image_path)
