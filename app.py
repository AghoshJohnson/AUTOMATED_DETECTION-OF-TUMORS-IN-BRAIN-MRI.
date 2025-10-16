from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

app = Flask(__name__, template_folder="templates")

# ✅ Load Model
model_path = "model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🔥 ERROR: {model_path} not found!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Serve Homepage
@app.route("/", methods=["GET"])
def index():
    return render_template("index2.html")

# ✅ Image Upload and Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            confidence = confidence.item()

        result = "No Tumor" if prediction.item() == 0 else "Tumor Detected"
        return jsonify({"prediction": result, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
