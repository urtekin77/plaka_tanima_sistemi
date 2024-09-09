"""
 pip install easyocr
 pip install opencv-python
 pip install numpy
 pip install onnxruntime-gpu
 pip install Flask
 pip install Flask-Cors
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import cv2
import easyocr

app = Flask(__name__)
CORS(app)  # React.js ile Flask arasında CORS sorununu önlemek için

# ONNX modelini yükle
model_path = "C:\\Users\\urtek\\OneDrive\\Desktop\\EDA\\Staj\\Denemeler 2\\yolov10s_20_vehicle\\weights\\model.onnx"
session = ort.InferenceSession(model_path)

reader = easyocr.Reader(['tr'])


# Tahmin işlemi için işlev
def process_image(image):
    h, w = image.shape[:2]
    target_h, target_w = (640, 640)

    # Resmin oranlarını koruyarak hedef boyuta göre yeniden boyutlandırma
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Boş alanları siyahla doldurma
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # Resmi normalize edin ve 4D tensöre çevirme
    input_image = padded_image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))  # HWC'den CHW'ye
    input_image = np.expand_dims(input_image, axis=0)

    # ONNX modelinden tahmin al
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_image})
    # Tespit edilen kutular
    boxes = outputs[0][0]
    detected_boxes = [box for box in boxes]

    if len(detected_boxes) > 0:
        # İlk kutuyu al (plaka kutusu)
        x1, y1, x2, y2 = map(int, detected_boxes[0][:4])
        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
        # Plakanın ideal boyutlarına göre bir hedef noktalar dizisi oluşturun
        width = x2 - x1
        height = y2 - y1
        dest_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # Perspektif dönüşüm matrisini oluştur
        matrix = cv2.getPerspectiveTransform(points, dest_points)

        # Görüntüye perspektif dönüşüm uygulayın
        warped_image = cv2.warpPerspective(padded_image, matrix, (width, height))

        

        scale_factor = 2
        resized_license_plate = cv2.resize(warped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        
        inv_gamma = 1.0 / 1.5
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        brightened_image = cv2.LUT(resized_license_plate, table)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        brightened_image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        results_text = reader.readtext(
            brightened_image_clahe, 
            allowlist="ABCDEFGHIJKLMNOPRSTUQVWXYZ0123456789",
            low_text=0.5,
            detail=1,
            contrast_ths=0.2,
            mag_ratio=2,
            link_threshold=0.65
        )
        detected_text = ' '.join([result[1] for result in results_text])
        print("detected_text = " , detected_text)
        return detected_text
    else:
        return None
    


# Flask API için /predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Resim dosyası eksik'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Resim dosyası yüklenmedi'}), 400

    try:
        # Resim dosyasını OpenCV ile işleyin
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Plaka tespitini gerçekleştirin
        result_text = process_image(image)
        if result_text:
            return jsonify({'result': result_text})
        else:
            return jsonify({'error': 'Plaka tespit edilemedi'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
