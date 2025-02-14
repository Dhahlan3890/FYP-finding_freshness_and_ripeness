import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
import streamlit as st
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ViT model from Hugging Face
model = ViTForImageClassification.from_pretrained('Dhahlan2000/ripeness_detection', num_labels=20)
model.to(device)
model.eval()

# Load ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('Dhahlan2000/ripeness_detection')

# Class labels
predicted_classes = [
    'FreshApple', 'FreshBanana', 'FreshBellpepper', 'FreshCarrot', 'FreshCucumber', 'FreshMango', 'FreshOrange', 
    'FreshPotato', 'FreshStrawberry', 'FreshTomato', 'RottenApple', 'RottenBanana', 'RottenBellpepper', 'RottenCarrot', 
    'RottenCucumber', 'RottenMango', 'RottenOrange', 'RottenPotato', 'RottenStrawberry', 'RottenTomato']

# Function for inference
def classify_fruit(image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return predicted_classes[predicted_class_idx]

# Streamlit UI
st.title("Fruit Ripeness Detection")
st.write("Upload an image or capture from camera to determine whether it's fresh or rotten.")

# Upload from file
uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Uploaded Image"):
        prediction = classify_fruit(image)
        st.write(f"**Prediction:** {prediction}")

# Capture from camera
camera_image = st.camera_input("Capture an image")
if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    if st.button("Classify Captured Image"):
        prediction = classify_fruit(image)
        st.write(f"**Prediction:** {prediction}")
