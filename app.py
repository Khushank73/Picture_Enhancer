from flask import Flask, render_template, request, redirect, url_for, send_from_directory,send_file
import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'

model_path = 'models/RRDB_ESRGAN_x4.pth'  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = osp.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(img_path)

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()

            # Encode the output image to base64 for HTML rendering
            _, buffer = cv2.imencode('.png', output)
            result_img_data = base64.b64encode(buffer).decode('utf-8')

            return render_template('index.html', image=result_img_data)

    return "Error processing image"

@app.route('/download')
def download():
    result_img_path = osp.join(app.config['RESULT_FOLDER'], 'result_image.png')
    return send_file(result_img_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
