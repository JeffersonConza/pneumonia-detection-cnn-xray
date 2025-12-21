import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CheXDS
from src.config import DEVICE, MODEL_SAVE_PATH, IMG_SIZE, TEST_DIR

def predict_single_image(image_path, model):
    # Same preprocessing as training (Resize -> Tensor -> Normalize)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # ImageNet normalization statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None, None

    # Prepare batch of size 1
    input_tensor = transform(image).unsqueeze(0) 
    input_tensor = input_tensor.to(DEVICE)
    
    # Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    class_names = ['NORMAL', 'PNEUMONIA']
    prediction = class_names[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    return image, prediction, conf_score

if __name__ == "__main__":
    # 1. Initialize Model
    print("Initializing CheX-DS Model...")
    model = CheXDS(num_classes=2)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    # 2. Pick Random Image from Test Set
    # We look into data/chest_xray/test/
    target_class = random.choice(['NORMAL', 'PNEUMONIA'])
    folder_path = os.path.join(TEST_DIR, target_class)
    
    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} not found.")
        sys.exit()

    filename = random.choice(os.listdir(folder_path))
    full_path = os.path.join(folder_path, filename)
    
    print(f"\nAnalyzing: {filename}")
    print(f"Ground Truth: {target_class}")
    
    # 3. Predict
    img, pred, conf = predict_single_image(full_path, model)
    
    # 4. Display Result
    if img:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        
        # Color title based on correctness
        color = 'green' if pred == target_class else 'red'
        plt.title(f"Prediction: {pred}\nConfidence: {conf:.2f}%", color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
        print(f"Result: {pred} ({conf:.2f}%)")
