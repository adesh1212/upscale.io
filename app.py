import os
import cv2
import numpy as np
import torch
import glob
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import RRDBNet_arch as arch  # Ensure this module is available
from PIL import Image
import io

app = Flask(__name__)

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ESRGAN model
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Directory for saving images
UPLOAD_FOLDER = 'static/img'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        if file:
            filename = secure_filename(file.filename)
            
            # Save the original image in static/img
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Upscale the image using ESRGAN
            upscaled_img = upscale_image(input_path)
            
            # Save the upscaled image in static/results
            result_filename = f"upscaled_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            upscaled_img.save(result_path)

            # Optionally, return the upscaled image as a file-like object for download or display
            img_io = io.BytesIO()
            upscaled_img.save(img_io, 'PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')

    return render_template('index.html')


def upscale_image(image_path):
    """Upscale the image using the ESRGAN model."""
    # Read the input image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image (convert to range [0, 1])
    img = img * 1.0 / 255
    
    # Convert image to torch tensor (C, H, W)
    img = torch.from_numpy(np.transpose(img[:, :, [0, 1, 2]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    
    # Perform super-resolution using the ESRGAN model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Convert the output tensor back to an image (H, W, C) and back to RGB
    output = np.transpose(output[[0, 1, 2], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    # Convert the numpy array back to a PIL Image
    output_image = Image.fromarray(output)

    return output_image


if __name__ == '__main__':
    app.run(debug=True)
