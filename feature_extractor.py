# pip install tensorflow==2.11.0
# pip install keras
# pip install mtcnn==0.1.0
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from tqdm import tqdm
from keras_vggface.vggface import VGGFace
##############################
import os
import pickle

filenames = []

fol_path = 'data'
# List the contents of fol_path to verify the folder structure
print("Contents of root folder:", os.listdir(fol_path))

# Loop through each actor folder directly
for folder in os.listdir(fol_path):
    folder_path = os.path.join(fol_path, folder)
    # print(f"Checking folder: {folder_path}")

    # Check if folder_path is indeed a directory
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # Ensure the path is a file (an image in this case)
            if os.path.isfile(file_path):
                # print(f"Adding file to list: {file_path}")
                filenames.append(file_path)

print("Total files found:", len(filenames))

# Save the filenames list for later use
pickle.dump(filenames, open('filenames.pkl', 'wb'))
##############################
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
def feature_extractor(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = keras.applications.resnet50.preprocess_input(expanded_img)

    # Get the features
    result = model.predict(preprocessed_img).flatten()
    return result

#
# # Extract features for all images
features = []
for file in tqdm(filenames):
    features.append(feature_extractor(file, model))
#
# Save features to a pickle file
pickle.dump(features, open('embeddings.pkl', 'wb'))
