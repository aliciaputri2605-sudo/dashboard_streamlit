
import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Lung Cancer Analytics",
    layout="wide"
)

st.title(
    "Lung Cancer Analytics Dashboard"
)

# --- Overview Section ---
df_scaled_features = pd.read_csv(
    "data/features/scaled_features.csv"
)

total = len(df_scaled_features)
cancer = len(df_scaled_features[df_scaled_features["label"]==0])
normal = len(df_scaled_features[df_scaled_features["label"]==1])

col1,col2,col3 = st.columns(3)

col1.metric(
    "Total Data",
    total
)

col2.metric(
    "Cancer",
    cancer
)

col3.metric(
    "Normal",
    normal
)

# --- Cluster Distribution ---
cluster_df = pd.read_csv(
    "data/features/kmeans_results.csv"
)

st.subheader(
    "Cluster Distribution"
)

st.bar_chart(
    cluster_df["cluster"].value_counts()
)

# --- AI Prediction Section ---
st.title(
    "AI Lung Cancer Prediction"
)

# Load the pre-trained Logistic Regression model
model = joblib.load(
    "models/logistic_regression.pkl"
)

# Load MobileNetV2 for feature extraction
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Define the feature extraction function
def extract_feature(image):
    image = cv2.resize(
        image,
        (224,224)
    )
    image = np.expand_dims(
        image,
        axis=0
    )
    image = preprocess_input(image)
    feature = base_model.predict(
        image,
        verbose=0
    )
    return feature.flatten()

# Define the make_gradcam_heatmap function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# File uploader for CT Scan
uploaded = st.file_uploader(
    "Upload CT Scan",
    type=[
        "jpg",
        "png",
        "jpeg"
    ]
)

if uploaded:
    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )

    image_raw = cv2.imdecode(
        file_bytes,
        1
    )

    st.image(
        image_raw,
        channels="BGR",
        caption="Uploaded Image"
    )

    # Make prediction
    feature = extract_feature(image_raw)
    feature = feature.reshape(1,-1)
    prediction = model.predict(feature)[0]

    if prediction == 0:
        st.success(
            "Normal Lung"
        )
    else:
        st.error(
            "Cancer Lung"
        )

    # Generate Grad-CAM heatmap
    # Resize and preprocess the image for the cnn_model (used for Grad-CAM)
    image_for_cam = cv2.resize(image_raw, (224, 224))
    image_for_cam = np.expand_dims(image_for_cam, axis=0)
    image_for_cam = preprocess_input(image_for_cam)

    # Dynamically find the name of the last convolutional layer for MobileNetV2
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        # Fallback to a common MobileNetV2 last convolutional layer name if not found dynamically
        # This might need adjustment based on the exact MobileNetV2 architecture
        print("Could not find a convolutional layer name dynamically. Using 'Conv_1'.")
        last_conv_layer_name = "Conv_1"

    heatmap = make_gradcam_heatmap(
        image_for_cam,
        base_model,
        last_conv_layer_name
    )

    # Apply the heatmap to the original image for visualization
    heatmap = cv2.resize(heatmap, (image_raw.shape[1], image_raw.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    superimposed_img = superimposed_img * 0.5 + heatmap * 0.5
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    st.image(
        superimposed_img,
        caption="Grad-CAM Heatmap"
    )
