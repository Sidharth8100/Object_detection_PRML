from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.datasets import cifar10
from model_utils import create_model, class_names
from PIL import Image

app = Flask(__name__)
model = None
x_test = None
y_test = None

@app.route('/')
def index():
    image_files = [f'image_{i}.png' for i in range(10)]
    return render_template('index.html', image_files=image_files)

@app.route('/train', methods=['POST'])
def train_model():
    global model, x_test, y_test

    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = create_model()
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

    # Save first 10 test images
    os.makedirs('static/images', exist_ok=True)
    for i in range(10):
        img = Image.fromarray((x_test[i] * 255).astype(np.uint8))
        img.save(f'static/images/image_{i}.png')

    return jsonify({'message': 'Model trained and images saved.'})

@app.route('/predict', methods=['POST'])
def predict():
    global model, x_test

    if model is None:
        return jsonify({'error': 'Model not trained yet.'}), 400

    index = int(request.json['index'])
    img = np.expand_dims(x_test[index], axis=0)
    pred = model.predict(img)
    pred_class = class_names[np.argmax(pred)]

    return jsonify({'prediction': pred_class})

if __name__ == '__main__':
    app.run(debug=True)
