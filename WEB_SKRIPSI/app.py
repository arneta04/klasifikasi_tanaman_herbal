from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("model_resnet152v2_herbal.keras")

class_names = [
    'Daun Syngonium', 'Daun jambu biji', 'Daun kentang', 'Daun kersen',
    'Daun kumis kucing', 'Daun mangga', 'Daun rezeki putih', 'Daun ruskus',
    'Daun sirih', 'Daun sirsak', 'Tidak teridentifikasi'
]

kategori_map = {
    "Daun Syngonium": "Herbal",
    "Daun jambu biji": "Herbal",
    "Daun kentang": "Non-Herbal",
    "Daun kersen": "Herbal",
    "Daun kumis kucing": "Herbal",
    "Daun mangga": "Non-Herbal",
    "Daun rezeki putih": "Non-Herbal",
    "Daun ruskus": "Non-Herbal",
    "Daun sirih": "Herbal",
    "Daun sirsak": "Herbal",
    "Tidak teridentifikasi": "Tidak Teridentifikasi"
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_bytes = file.read()

    # Proses prediksi
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))

    nama = class_names[predicted_index]
    tanaman = kategori_map.get(nama, "Tidak Diketahui")

    # Encode gambar sebagai base64 agar bisa ditampilkan langsung
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{encoded_image}"

    return jsonify({
        'nama': nama,
        'tanaman': tanaman,
        'confidence': round(confidence * 100, 2),
        'image_data_url': image_data_url
    })

if __name__ == '__main__':
    app.run(debug=True)
