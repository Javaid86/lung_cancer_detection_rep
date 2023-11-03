# Import necessary libraries
import streamlit as st
# import tensorflow as tf
from keras.preprocessing import image
import numpy as np


# Define a function to preprocess the image
def preprocess_image(image_path, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    return img


# Load a pre-trained model for lung cancer prediction
model = keras.models.load_model('model_trained.h5')

# Streamlit app header
st.title('Lung Cancer Prediction App')

# Upload an X-ray image
uploaded_image = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

# Make predictions when an image is uploaded
if uploaded_image is not None:
    # Preprocess the uploaded image
    img = preprocess_image(uploaded_image)

    # Make predictions
    prediction = model.predict(img)

    # Display the result
    st.image(uploaded_image, caption='Uploaded X-ray Image', use_column_width=True)

    if prediction[0][0] > 0.5:
        st.write("Prediction: Cancerous")
    else:
        st.write("Prediction: Non-Cancerous")

# Provide information and instructions
st.write("This app uses a pre-trained deep learning model to predict lung cancer from X-ray images.")
st.write("Upload an X-ray image to make a prediction.")
