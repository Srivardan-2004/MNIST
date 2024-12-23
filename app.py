# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('mnist.keras')

# Define the Streamlit app
st.title("MNIST Digit Prediction")

# Upload an image
uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape(1, 28, 28)  # Reshape for model

    # Predict the digit
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    st.image(image, caption=f"Uploaded Image (Predicted: {predicted_digit})", use_column_width=True)

