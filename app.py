from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from dog_breed_names import dog_names
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('dog_model.keras')


def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file)
        image = preprocess_image(image)

        prediction = model.predict(image)
        predicted_class_idx = np.argmax(prediction)
        dog_names_list = dog_names()
        predicted_class_name = dog_names_list[predicted_class_idx]
        print(predicted_class_name)

        response = {
            'prediction': str(predicted_class_name)  
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
