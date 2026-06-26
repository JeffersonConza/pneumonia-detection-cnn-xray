import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import CheXDS
from src.config import MODEL_SAVE_PATH, DEVICE

# Load the trained CheX-DS model
print(f"Loading model on device: {DEVICE}...")
model = CheXDS(num_classes=2)

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Please run training first.")
    sys.exit(1)

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

def predict_xray(img):
    # standard validation preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # ImageNet normalization statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert numpy input image to PIL and transform
    img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    classes = ['NORMAL', 'PNEUMONIA']
    return {classes[i]: float(probs[i]) for i in range(2)}

# Build GUI layout
demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(label="Upload Frontal Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="Diagnostic Output"),
    title="🫁 Pneumonia Detection AI",
    description="CheX-DS Ensemble (DenseNet121 + Swin Transformer Base) running on Google Colab GPU.",
    theme="glass",
    live=False
)

if __name__ == "__main__":
    # share=True exposes a secure public HF link from Colab
    demo.launch(share=True)
