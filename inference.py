# inference.py
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Define transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    result = "No Tumor" if prediction.item() == 0 else "Tumor Detected"

# Example usage
image_path = "test_image.jpg"  # Replace with actual image path
print(predict(image_path))
