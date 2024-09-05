import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import io
import os
import RRDBNet_arch as arch

# Load ESRGAN model
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

def upscale_image(image):
    # Convert PIL Image to numpy array (RGB format)
    img = np.array(image.convert('RGB'))
    
    # Normalize and prepare the image for the model
    img = img / 255.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)
    
    # Perform super-resolution
    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Convert the output to image format
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    # Convert numpy array back to PIL Image
    return Image.fromarray(output, 'RGB')

# Streamlit app interface
st.title("Image Upscaler using ESRGAN")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read and display the original image in its original resolution
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Original Image", use_column_width=False, width=input_image.width)

    # Upscale the image
    upscaled_image = upscale_image(input_image)

    # Display the upscaled image
    st.image(upscaled_image, caption="Upscaled Image", use_column_width=False,width=upscaled_image.width)

    # Option to download the upscaled image
    img_byte_arr = io.BytesIO()
    upscaled_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    st.download_button("Download Upscaled Image", img_byte_arr, "upscaled_image.png", "image/png")
