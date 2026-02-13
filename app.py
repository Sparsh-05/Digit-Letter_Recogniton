import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow_datasets as tfds


st.title("Digit Recognition System")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "/Users/sparshkapoor/Desktop/DRS/Digit_recognition.h5"
    )

model = load_model()

@st.cache_resource
def load_label_map():
    # EMNIST Balanced character mapping
    chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
    ]
    return lambda idx: chars[idx]

label_map = load_label_map()

st.write("Draw a digit or letter and click Predict")

st.sidebar.header("Controls")

stroke_width = st.sidebar.slider("Stroke width", 5, 25, 12)
stroke_color = "#FFFFFF"
bg_color = "#000000"

canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Extract image
        img = canvas.image_data.astype(np.uint8)
        
        # Use alpha channel (white drawings on black background)
        img = Image.fromarray(img).convert("L")
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize to [0,1]
        img = np.array(img).astype('float32')/255.0

        # rotate image 90 degree
        img = np.rot90(img, k=1)      
        img = np.fliplr(img)          
        
        # Reshape for model input (batch_size, height, width, channels)
        img = img.reshape(1, 28, 28, 1)
        
        # Make prediction
        pred = model.predict(img, verbose=0)
        
        # Get predicted class and confidence
        cls = int(np.argmax(pred))
        conf = float(np.max(pred)) * 100
        
        # Convert class index to character
        char = label_map(int(cls))
        
        st.success(f"Prediction: **{char}**")
        st.info(f"Confidence: {conf:.2f}%")
        
        # Show what the model sees
        with st.expander("See preprocessed image (28x28)"):
            st.image(img.reshape(28, 28), width=150, clamp=True)
            
        # Show top 5 predictions
        with st.expander("See top 5 predictions"):
            top5_indices = np.argsort(pred[0])[-5:][::-1]
            for idx in top5_indices:
                char_pred = str(label_map(int(idx)))
                confidence = pred[0][idx] * 100
                st.write(f"{char_pred}: {confidence:.2f}%")
    else:
        st.warning("Please draw something first!")

if st.button("Clear Canvas"):
    st.rerun()