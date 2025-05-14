# Full Project: Handwritten Digit Recognition with Deep Learning and Streamlit

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# TRAINING SECTION
@st.cache_resource
def train_and_save_model():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/mnist.csv")
    
    # Preprocessing
    X = df.drop('label', axis=1).values / 255.0
    y = to_categorical(df['label'].values)
    X = X.reshape(-1, 28, 28, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CNN Model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/cnn_digit_model.h5")
    return model

# Load model
if not os.path.exists("saved_model/cnn_digit_model.h5"):
    model = train_and_save_model()
else:
    model = load_model("saved_model/cnn_digit_model.h5")

# STREAMLIT UI
st.title("Handwritten Digit Recognizer")
st.write("Upload a 28x28 grayscale PNG image of a digit (black on white background).")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption='Uploaded Image', use_column_width=False)

    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image_array)
    st.write(f"Predicted Digit: **{np.argmax(prediction)}**")
