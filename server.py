from flask import Flask, request, jsonify
from flask_cors import CORS 
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)
with open("pickled_model.pkl", "rb") as f:
    pickled_model = f.read()
model = pickle.loads(pickled_model)

@app.route('/api/disease_prediction', methods=['POST'])
def image_size():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img = Image.open(file).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predict = {
        'Bacterial Red disease' : float(prediction[0][0]),
        'Bacterial diseases - Aeromoniasis' : float(prediction[0][1]),
        'Bacterial gill disease' :float(prediction[0][2]),
        'Fungal diseases Saprolegniasis' : float(prediction[0][3]),
        'Healthy Fish' : float(prediction[0][4]),
        'Parasitic diseases' : float(prediction[0][5]),
        'Viral diseases White tail disease' : float(prediction[0][6])
    }
    return jsonify({'predictions': predict})

if __name__ == '__main__':
    app.run(debug=False)
