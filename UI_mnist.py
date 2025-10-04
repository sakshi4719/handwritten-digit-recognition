import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("digit_classification_model.h5")
    return model

model = load_model()

st.markdown("<h1 style='text-align: center;'>Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a digit below (0–9) and the neural net will try to recognize it.</p>", unsafe_allow_html=True)

# Set up side-by-side layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Draw Here")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

        # Threshold to binary image
        _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

        # Find bounding box of digit
        coords = cv2.findNonZero(img_bin)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_cropped = img_bin[y:y+h, x:x+w]
        else:
            img_cropped = img_bin

        # Resize to 20x20
        img_resized = cv2.resize(img_cropped, (20, 20), interpolation=cv2.INTER_AREA)

        # Pad to 28x28
        top = bottom = (28 - 20) // 2
        left = right = (28 - 20) // 2
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

        # Normalize and reshape
        img_normalized = img_padded / 255.0
        img_input = img_normalized.reshape(1, 28, 28)

        st.subheader("Processed Input")
        st.image(img_padded, width=150)

        if st.button("Predict"):
            prediction = model.predict(img_input)
            predicted_class = np.argmax(prediction)
            st.markdown(f"<h3 style='text-align: center;'>Prediction: {predicted_class}</h3>", unsafe_allow_html=True)

# Footer with names
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>Team Members</h4>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
    Sakshi Raut<br>
    Asmi Rode<br>
    Aakanksh Sen<br>
    Jay Shah<br>
    Samridhi Sharan
</div>
""", unsafe_allow_html=True)
