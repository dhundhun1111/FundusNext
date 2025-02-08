import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load Model
@st.cache_resource
def load_model():
    model = models.convnext_tiny(weights=None)
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(768, eps=1e-06, elementwise_affine=True),
        nn.Linear(768, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    
    # model.load_state_dict(torch.load("ConvNeXtTiny_best.pth", map_location=torch.device('cpu')))
    # Load state dict from Hugging Face
    state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/dhundhun1111/FundusNext/resolve/main/ConvNeXtTiny_best.pth", map_location=torch.device("cpu"))

    # Load weights into the model
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Welcome to FundusNext")
st.write("Upload the image of your Fundus to predict Glaucoma.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    image = image.convert("RGB")  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension


   # Make Prediction
    with torch.no_grad():
        output = model(image)
        confidence = output.item() * 100  # Convert to percentage

    # Determine Prediction
    prediction = "Glaucoma Detected" if confidence > 50 else "No Glaucoma"

    st.write(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
