# **Handwritten Digit Recognition Web Application**

## **Overview**  
This project is an interactive web-based application designed to recognize handwritten digits. Users can draw a digit on a virtual canvas, submit it, and receive a prediction based on a trained deep learning model. The project combines a convolutional neural network (CNN) model, a Flask backend, and an intuitive HTML frontend to provide an engaging and functional user experience. 

An ensemble learning approach is implemented in the backend to improve prediction accuracy by leveraging multiple augmented versions of the input.

---

## **Key Features**  
1. **Interactive Canvas:**  
   - A user-friendly drawing interface implemented in HTML and JavaScript.  
   - Allows users to draw digits directly on a canvas and submit them for recognition.  

2. **Preprocessing:**  
   - The drawn digit is resized to 28x28 pixels and converted to grayscale to match the input format of the CNN model.  
   - Colors are inverted to align with the training data (black background, white digit).  

3. **Backend Integration:**  
   - A Flask API (`app.py`) handles the request from the frontend.  
   - The digit image is processed and passed to the trained CNN model (`digit_recognition.h5`) for prediction.  
   - Implements **ensemble learning** using augmented variations of the input to improve robustness and accuracy.  

4. **Machine Learning Model:**  
   - A Convolutional Neural Network (CNN) trained on the MNIST dataset for digit recognition.  
   - Model architecture includes layers for convolution, activation (ReLU), and max-pooling to extract and reduce features.  
   - Data augmentation techniques, such as rotation and zooming, were employed during training to improve robustness against real-world variations.  

5. **Deployment-Ready:**  
   - All dependencies (`requirements.txt`) are listed for easy setup.  
   - Hosted online for global access:  
     [**Digit Recognition Web App**](https://umang-shikarvar.github.io/Digit_Recognition/)

---

## **Technical Components**  

1. **Frontend (HTML & JavaScript):**  
   - Implements a canvas where users can draw digits and clear the drawing if needed.  
   - Dynamically sends image data as a JSON object to the Flask backend via an HTTP POST request.

2. **Backend (Flask):**  
   - Accepts the preprocessed image data and applies **ensemble learning**:
     - Generates 10 augmented versions of the input image (e.g., rotated, scaled, shifted).
     - Predicts each augmented image using the CNN model.  
     - Combines the predictions using majority voting to produce the final result.  
   - Flask-CORS ensures seamless communication between the frontend and backend.  

3. **Machine Learning Model:**  
   - CNN architecture with multiple convolutional layers and max-pooling layers for feature extraction.  
   - Optimized using data augmentation to generalize better on user-drawn inputs.  
   - Trained and saved as `digit_recognition.h5`.

4. **Dependencies:**  
   - `Flask` for backend development.  
   - `Flask-CORS` for handling cross-origin requests.  
   - `TensorFlow` for loading and running the machine learning model.  
   - `NumPy` and `SciPy` for numerical processing.  

---

## **How It Works**  
1. The user draws a digit on the canvas.  
2. The drawing is resized, converted to grayscale, and inverted to match the training data.  
3. The processed image data is sent to the Flask backend.  
4. In the backend:
   - 10 augmented versions of the image are generated using random transformations.  
   - Each augmented image is predicted using the trained CNN model.  
   - The final prediction is made using **majority voting** across all predictions.  
5. The predicted digit is displayed on the frontend for the user.  

---

## **Applications**  
- Digit recognition for handwritten forms or input fields.  
- A starting point for optical character recognition (OCR) systems.  
- Educational tool for understanding machine learning and its applications.  

---

This project showcases the seamless integration of web development and machine learning, with an advanced ensemble learning approach for robust and accurate digit recognition.

**Deployment Link:** [**Digit Recognition Web App**](https://umang-shikarvar.github.io/Digit_Recognition/)
