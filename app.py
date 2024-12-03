from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf  # Replace with PyTorch or another framework if needed
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask_cors import CORS
import scipy.stats

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/predict": {"origins": ["https://umang-shikarvar.github.io/Digit_Recognition/"]}})

# Load your trained model
model = load_model('digit_recognition.h5')  # Update with your actual model path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.json
        image_data = np.array(data['image'], dtype=np.float32)

        # Reshape and normalize the image data
        image = image_data.reshape(1, 28, 28, 1)  # Assuming model expects normalized input
        
        # Define the ImageDataGenerator with augmentation parameters
        datagen = ImageDataGenerator(
        rotation_range=30,        # Random rotation between 0 and 15 degrees
        width_shift_range=0.15,    # Random horizontal shift
        height_shift_range=0.15,   # Random vertical shift
        zoom_range=0.2,           # Random zoom
        shear_range=0.3,          # Random shear
        horizontal_flip=False,    # Flip not used for MNIST-like images
        fill_mode='nearest'       # Fill in missing pixels
        )
    
        datagen.fit(image)
        
        # Generate augmented images
        augmented_gen = datagen.flow(image, batch_size=1)

        augmented_images=[]

        for j in range(10):
            augmented_images.append(next(augmented_gen).reshape(28,28,1))

        augmented_images=np.array(augmented_images)
        
        predict =scipy.stats.mode(np.argmax(model.predict((augmented_images),verbose=0),axis=1))[0]

        # Return the prediction as JSON
        return jsonify({'prediction': int(predict)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)