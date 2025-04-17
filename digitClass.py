import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas

# Mapping from class index to character
index_to_char = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model.h5")
    return model

model = load_model()

st.markdown("<h1 style='text-align: center;'>Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a letter or digit and the neural net will try to recognize it.</p>", unsafe_allow_html=True)

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
            predicted_char = index_to_char.get(predicted_class, "?")
            st.markdown(f"<h3 style='text-align: center;'>Prediction: {predicted_char}</h3>", unsafe_allow_html=True)

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
