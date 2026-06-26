import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import SimpleCNN, TransferResNet, CheXDS
from src.config import DEVICE, MODEL_SAVE_PATH, IMG_SIZE, TEST_DIR, BASE_DIR

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
    parser = argparse.ArgumentParser(description="Predict single image class")
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'chexds'], default='chexds',
                        help="Choose model type to run inference: 'cnn', 'resnet', or 'chexds'")
    args = parser.parse_args()

    # 1. Initialize Model
    print(f"Initializing {args.model.upper()} Model...")
    if args.model == 'cnn':
        model = SimpleCNN(num_classes=2)
    elif args.model == 'resnet':
        model = TransferResNet(num_classes=2)
    else:
        model = CheXDS(num_classes=2)

    # Check if a model-specific file exists, otherwise fallback to default MODEL_SAVE_PATH
    specific_path = os.path.join(BASE_DIR, 'models', f'pneumonia_model_{args.model}.pth')
    if os.path.exists(specific_path):
        load_path = specific_path
    elif os.path.exists(MODEL_SAVE_PATH):
        print(f"Warning: Model-specific weights not found at {specific_path}. Falling back to default path {MODEL_SAVE_PATH}.")
        load_path = MODEL_SAVE_PATH
    else:
        print(f"Error: No weight file found for {args.model.upper()}. Checked paths:\n - {specific_path}\n - {MODEL_SAVE_PATH}")
        sys.exit(1)

    print(f"Loading weights from {load_path}...")
    try:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        print(f"Error loading state_dict for {args.model.upper()} from {load_path}: {e}")
        print("This typically happens if the file contains weights for a different architecture.")
        print("Please train the model first using: python main.py --model <model_type>")
        sys.exit(1)

    model.to(DEVICE)
    model.eval()
    
    # 2. Pick Random Image from Test Set
    # We look into data/chest_xray/test/
    target_class = random.choice(['NORMAL', 'PNEUMONIA'])
    folder_path = os.path.join(TEST_DIR, target_class)
    
    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} not found.")
        sys.exit(1)

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
