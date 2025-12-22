import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# --- Project Imports ---
# Add the current directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import CheXDS
from src.config import DEVICE, MODEL_SAVE_PATH, IMG_SIZE

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI ---
st.markdown("""
    <style>
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. Load Model (Cached) ---
@st.cache_resource
def load_model():
    """
    Loads the CheX-DS model once and caches it in memory.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        st.error(f"❌ Model weight file not found at: `{MODEL_SAVE_PATH}`")
        st.info("💡 Please run `python main.py --model chexds` to train the model first.")
        st.stop()

    try:
        # Initialize Architecture
        model = CheXDS(num_classes=2)
        
        # Load Weights
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        
        model.to(DEVICE)
        model.eval()
        
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# --- 2. Preprocessing Function ---
def process_image(image):
    """
    Prepares the uploaded image for the model (Resize -> Tensor -> Normalize).
    """
    # Use RGB to ensure 3 channels (handles grayscale or RGBA inputs)
    image = image.convert('RGB')
    
    # Must match the training transformations exactly
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # ImageNet normalization standard
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension: [1, 3, 256, 256]
    return transform(image).unsqueeze(0).to(DEVICE)

# --- 3. UI Layout ---
st.markdown("<h1 class='main-header'>🫁 Pneumonia Detection AI</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <strong>CheX-DS Implementation</strong> - <em>DenseNet121 + Swin Transformer Ensemble</em><br>
    State-of-the-art ensemble model with <strong>99% sensitivity</strong> for pneumonia detection
</div>
""", unsafe_allow_html=True)

# --- Main Layout: Sidebar + Two Column Content ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # System Information
    with st.expander("🖥️ System Information", expanded=False):
        st.info(f"**Device:** {str(DEVICE).upper()}")
        st.info(f"**Python:** {sys.version.split()[0]}")
        st.info(f"**PyTorch:** {torch.__version__}")
        st.info(f"**CUDA Available:** {torch.cuda.is_available()}")
    
    st.divider()
    
    # Model Configuration
    st.subheader("📊 Model Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=50, 
        step=5,
        help="Minimum confidence for a NORMAL diagnosis to be considered reliable"
    )
    
    show_probabilities = st.checkbox("Show Class Probabilities", value=True)
    show_image_details = st.checkbox("Show Image Details", value=True)
    
    st.divider()
    
    # Medical Disclaimer
    with st.expander("⚠️ Medical Disclaimer"):
        st.warning("""
        **This tool is for educational and research purposes only.**
        
        - Not FDA approved
        - Not a substitute for professional medical diagnosis
        - Always consult qualified healthcare professionals
        - Do not make treatment decisions based solely on this tool
        """)
    
    st.divider()
    
    # Model Performance Info
    with st.expander("📈 Model Performance"):
        st.markdown("""
        **CheX-DS Metrics:**
        - **Sensitivity:** 99%
        - **Accuracy:** 85.26%
        - **Test Loss:** 0.2856
        
        **Architecture:**
        - DenseNet121
        - Swin Transformer Base
        - Learnable Ensemble Weights
        """)

# --- Main Content: Two Column Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPEG/PNG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal chest X-ray for pneumonia detection"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-Ray', width='stretch')
        
        if show_image_details:
            with st.expander("📋 Image Details", expanded=True):
                st.write(f"**📏 Dimensions:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**📊 Color Mode:** {image.mode}")
                st.write(f"**📁 Format:** {image.format}")
                file_size = len(uploaded_file.getvalue()) / 1024
                st.write(f"**💾 File Size:** {file_size:.1f} KB")
        
        st.divider()
        analyze_button = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
    else:
        st.info("""
        👆 **Get Started:**
        1. Upload a chest X-ray image using the file uploader above
        2. Click the "Analyze Image" button
        3. Review the diagnostic results in the right panel
        
        **Supported formats:** JPEG, PNG
        """)
        analyze_button = False

with col_right:
    st.subheader("📋 Diagnostic Results")
    
    if uploaded_file is not None and analyze_button:
        with st.spinner("🔄 Analyzing with CheX-DS Model..."):
            # Load Model
            model = load_model()
            
            try:
                # Process & Predict
                input_tensor = process_image(image)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)

                # Interpret Results
                class_names = ['NORMAL', 'PNEUMONIA']
                prediction = class_names[pred_idx.item()]
                conf_score = confidence.item() * 100
                
                # Get individual class probabilities
                normal_prob = probs[0][0].item() * 100
                pneumonia_prob = probs[0][1].item() * 100

                # --- Display Results ---
                st.success("✅ Analysis Complete")
                
                # Main metrics in 3 columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric("🩺 Diagnosis", prediction)
                
                with metric_col2:
                    st.metric("🎯 Confidence", f"{conf_score:.2f}%")
                
                with metric_col3:
                    reliability = "High" if conf_score >= 80 else "Medium" if conf_score >= 60 else "Low"
                    st.metric("📊 Reliability", reliability)

                st.divider()
                
                # Show probability breakdown if enabled
                if show_probabilities:
                    st.markdown("**📊 Class Probabilities**")
                    
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric("Normal", f"{normal_prob:.2f}%")
                        st.progress(normal_prob / 100)
                    
                    with prob_col2:
                        st.metric("Pneumonia", f"{pneumonia_prob:.2f}%")
                        st.progress(pneumonia_prob / 100)
                    
                    st.divider()
                
                # Color-coded feedback based on prediction
                if prediction == "PNEUMONIA":
                    st.error(
                        f"⚠️ **PNEUMONIA DETECTED**\n\n"
                        f"The model is **{conf_score:.1f}%** confident this scan shows signs of pneumonia.\n\n"
                        f"**Recommendation:** Immediate medical consultation is advised."
                    )
                else:
                    if conf_score < confidence_threshold:
                        st.warning(
                            f"⚠️ **UNCERTAIN RESULT**\n\n"
                            f"Prediction is NORMAL, but confidence ({conf_score:.1f}%) is below your threshold ({confidence_threshold}%).\n\n"
                            f"**Recommendation:** Consider additional imaging or clinical evaluation."
                        )
                    else:
                        st.success(
                            f"✅ **NORMAL SCAN**\n\n"
                            f"The model is **{conf_score:.1f}%** confident this scan appears healthy.\n\n"
                            f"**Note:** This does not rule out other conditions. Consult a healthcare provider for complete evaluation."
                        )
                
                # Additional Information
                with st.expander("ℹ️ Understanding the Results"):
                    st.markdown("""
                    **How to interpret confidence levels:**
                    
                    - **High (80-100%)**: Strong agreement with training patterns
                    - **Medium (60-80%)**: Moderate certainty, review recommended
                    - **Low (<60%)**: Uncertain, additional evaluation needed
                    
                    **Clinical Context:**
                    - **False Negative Rate:** <1% (Very low chance of missing pneumonia)
                    - **Sensitivity:** 99% (Excellent at detecting pneumonia cases)
                    - **Use Case:** Best suited for screening and triage
                    
                    **Important Notes:**
                    - This tool cannot replace professional radiologist interpretation
                    - Clinical symptoms and patient history must be considered
                    - Follow-up with healthcare provider is always recommended
                    """)

            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)
    
    else:
        st.info("👈 Upload an X-ray image and click 'Analyze' to see results here.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
        <strong>Powered by CheX-DS Architecture</strong> | Built with Streamlit & PyTorch<br>
        For research and educational purposes only | Not for clinical use
    </small>
</div>
""", unsafe_allow_html=True)