import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("WhatsApp Image 2025-06-24 at 13.26.59_99619a7d.jpg")

model = tf.keras.models.load_model(r"D:\Amit\Code DL Ali\Shoe vs Sandal vs Boot Dataset\model.h5")


class_names = ['Boot', 'Sandal', 'Shoe']

st.markdown(
    "<h1 style='text-align: center; color: white;'>👟 Shoe Type Classifier</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #f0f0f0;'>Upload a picture of a shoe (Boot / Sandal / Shoe) and the model will tell you the type</h4>",
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='The picture you uploaded', use_column_width=True)

    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(
    f"""
    <div style='
        background-color:#222;
        padding:20px;
        border-radius:10px;
        text-align:center;
        color:white;
        font-size:22px;
        margin-top:20px;
    '>
    🧠 Prediction: <b>{predicted_class}</b>
    </div>
    """,
    unsafe_allow_html=True
)

    st.bar_chart({class_names[i]: float(prediction[0][i]) for i in range(3)})

    st.sidebar.image("WhatsApp Image 2025-06-24 at 13.26.59_99619a7d.jpg", width=150, caption="Ahmed Abdelhady")
st.sidebar.markdown("**👤 Built by Ahmed Abdelhady – ML Engineer**")



st.markdown("---")
st.markdown("<center>👤 Built by Ahmed Abdelhady – ML Engineer", unsafe_allow_html=True)

