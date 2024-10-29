import streamlit as st
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN

# Initialize MTCNN face detector
detector = MTCNN()

# Load the VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load the feature list from the pickle file
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Helper function to save the uploaded image
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


# Function to rotate image by a given angle
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


# Function to extract features from the image
def extract_features(img_path, model, detector):
    sample_img = cv2.imread(img_path)

    # Resize image if too small
    if sample_img.shape[1] < 300:
        sample_img = cv2.resize(sample_img, (300, int(300 * sample_img.shape[0] / sample_img.shape[1])))

    # Attempt face detection with multiple rotations
    angles = [0, -10, 10]
    results = None

    for angle in angles:
        rotated_img = rotate_image(sample_img, angle)
        results = detector.detect_faces(rotated_img)
        if results:
            x, y, width, height = results[0]['box']
            face = rotated_img[y:y + height, x:x + width]
            break

    if not results:
        return None

    # Process face for feature extraction
    face_img = cv2.resize(face, (224, 224))
    face_array = np.asarray(face_img).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    features = model.predict(preprocessed_img).flatten()
    return features


# Function to recommend similar faces based on extracted features
def recommend(feature_list, feature):
    similarity = [cosine_similarity(feature.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.sidebar.title("Find Your Bollywood Lookalike")
st.markdown(
    """
    <style>
    .title {
        font-size:40px;
        text-align:center;
        font-weight:bold;
        color: #FF5733;
    }
    .header-text {
        font-size:25px;
        color:#333333;
        text-align:center;
        font-weight:normal;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        padding: 20px;
    }
    .result-text {
        font-size: 20px;
        font-weight:bold;
        text-align:center;
        color:#2E86C1;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">Bollywood Face Similarity Detector</div>', unsafe_allow_html=True)
st.markdown('<p class="header-text">Upload a photo to find out which Bollywood celebrity you resemble the most!</p>',
            unsafe_allow_html=True)

# Image upload section
uploaded_image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        # Display the uploaded image
        display_image = Image.open(uploaded_image)

        # Extract features from the uploaded image
        feature = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)

        if feature is not None:
            index_pos = recommend(feature_list, feature)
            predicted_actor = " ".join(filenames[index_pos].split("\\")[1].split('_'))

            # Display results with enhanced layout
            st.markdown(
                '<div class="result-text">You resemble: <span style="color:#FF5733;">' + predicted_actor + '</span></div>',
                unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(display_image, caption="Your Uploaded Image", use_column_width=True)
            with col2:
                st.image(filenames[index_pos], caption="Matched Bollywood Celebrity", use_column_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No face detected in the uploaded image. Please upload a clear and frontal image.")
