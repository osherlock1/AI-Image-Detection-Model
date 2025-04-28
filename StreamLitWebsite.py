import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import CLIPModel
from peft import PeftModel

# --- Model Class ---
class CLIPBinaryClassifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_id)
        self.classifier_head = nn.Linear(self.clip.config.vision_config.hidden_size, 1)

    def forward(self, pixel_values, labels=None, **kwargs):
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        image_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output
        logits = self.classifier_head(image_features)

        return {"logits": logits}

# --- Load Model ---
@st.cache_resource
def load_model():
    base_model = CLIPBinaryClassifier("openai/clip-vit-base-patch32")
    model = PeftModel.from_pretrained(base_model, 'final_model/')
    model.eval()
    return model

model = load_model()

# --- Streamlit App ---
st.title("🖼️ AI vs Real Image Detector")
st.markdown(
    "Upload an image, and the model will predict whether it was AI-generated or a real photograph!"
)

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")  # Force 3 channels (RGB)
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()

    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('🔍 Analyzing Image...'):
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # Channels first
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            logits = outputs["logits"]
            prob = torch.sigmoid(logits)

        # Result
        if prob.item() > 0.5:
            st.success("✅ Prediction: **AI-Generated Image**")
        else:
            st.success("✅ Prediction: **Real Photograph**")
else:
    st.info("Please upload an image to classify.")
