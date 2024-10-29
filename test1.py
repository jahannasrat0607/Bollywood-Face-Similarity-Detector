from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from mtcnn import MTCNN

# Load the feature list from the pickle file
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Initialize the face detector
detector = MTCNN()

# Load the image and detect faces
# img_path = r'C:\Users\Nasrat Jahan\PycharmProjects\Bollywood-Face-Similarity-Detector\sample\amitabh.jpeg'
# img_path = r'C:\Users\Nasrat Jahan\PycharmProjects\Bollywood-Face-Similarity-Detector\sample\hritik.jpg'
# img_path = r'C:\Users\Nasrat Jahan\PycharmProjects\Bollywood-Face-Similarity-Detector\sample\IMG_20240817_142146.jpg'
img_path = r'C:\Users\Nasrat Jahan\PycharmProjects\Bollywood-Face-Similarity-Detector\uploads\paa.jpeg'
sample_img = cv2.imread(img_path)
results = detector.detect_faces(sample_img)

# Assuming at least one face is detected
if results:
    x, y, width, height = results[0]['box']
    face = sample_img[y:y + height, x:x + width]

    # Extract features
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    # Find the cosine distance of the current image with all the features
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    print(sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1]))
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    # Load and display the recommended image
    temp_img = cv2.imread(filenames[index_pos])

    # Resize the window to fit the image
    temp_img = cv2.resize(temp_img, (224, 224))  # Adjust dimensions as needed

    cv2.imshow('Recommended Image', temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected.")
