import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🖐️ MNIST Digit Classifier")
st.write("Draw or upload a digit and the model will try to recognize it.")

# Upload image section
uploaded_file = st.file_uploader("Upload a 28x28 hand-drawn digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")   # Convert to grayscale
    image = ImageOps.invert(image)                   # Invert (MNIST digits are white on black)
    image = image.resize((28, 28))                   # Resize to MNIST size

    img_array = np.array(image) / 255.0              # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)      # Reshape to model input

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(image, caption="Processed Input", width=150)
    st.write(f"### 🔢 Predicted Digit: **{predicted_class}**")
