from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Define the CNN model structure (must match the trained model)
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)  # Adjust based on your model

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load trained model
model = BrainTumorCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Function to predict image
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return "Tumor" if predicted.item() == 1 else "No Tumor"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    prediction = predict_image(image_bytes)
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
