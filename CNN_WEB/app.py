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
MODEL_PATH = 'static/saved_model/cifar10_model.keras'

@app.route('/')
def index():
    global x_test
    num_images = 20
    
    # Load test data if not already loaded
    if x_test is None:
        _, (x_test, _) = cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
    
    # Generate random indices for the test set
    random_indices = np.random.randint(0, len(x_test), size=num_images)
    
    # Save new random test images
    images_dir = 'static/images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Generate new image files
    image_files = []
    for i, idx in enumerate(random_indices):
        img = Image.fromarray((x_test[idx] * 255).astype(np.uint8))
        image_name = f'image_{i}.png'
        img_path = os.path.join(images_dir, image_name)
        img.save(img_path)
        image_files.append(image_name)
    
    model_exists = os.path.exists(MODEL_PATH)
    return render_template('index.html', image_files=image_files, image_indices=random_indices, model_exists=model_exists)

def load_saved_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return True
    except:
        return False

@app.route('/train', methods=['POST'])
def train_model():
    global model, x_test, y_test

    try:
        # Create necessary directories
        model_dir = os.path.dirname(MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory: {model_dir}")

        # Check if model already exists
        if os.path.exists(MODEL_PATH):
            print(f"Model already exists at {MODEL_PATH}")
            return jsonify({'message': 'Model already exists. Using saved model.'})

        # Load and preprocess CIFAR-10 dataset
        print("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        # Create and train model
        print("Training model...")
        model = create_model()
        history = model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_test, y_test))

        # Save the model
        print(f"Saving model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        print("Model saved successfully!")

        return jsonify({
            'message': 'Model trained and saved successfully.',
            'model_path': MODEL_PATH
        })

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, x_test

    if model is None:
        # Try to load the model if it exists
        if not load_saved_model():
            return jsonify({'error': 'Model not trained yet.'}), 400

    # Load test data if not already loaded
    if x_test is None:
        _, (x_test, _) = cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0

    index = int(request.json['index'])
    img = np.expand_dims(x_test[index], axis=0)
    pred = model.predict(img)
    pred_class = class_names[np.argmax(pred)]

    return jsonify({'prediction': pred_class})

if __name__ == '__main__':
    # Try to load the model at startup
    load_saved_model()
    app.run(debug=True)
