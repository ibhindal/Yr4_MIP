#this script is for the training of the model we will be using for the project tensorflow and the BraCs dataset

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import nibabel as nib
from tensorflow.keras.callbacks import TensorBoard
import zipfile
import requests
import tarfile

# Download and extract the dataset
def download_and_extract_data(url, data_path):
    # Download the dataset
    response = requests.get(url, stream=True)
    zip_file_path = os.path.join(data_path, "archive.zip")
    
    with open(zip_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Extract the outer ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)

    # Extract the inner TAR file
    tar_file_path = os.path.join(data_path, "BraTS2021_Training_Data.tar")
    with tarfile.open(tar_file_path, 'r') as tar_ref:
        tar_ref.extractall(data_path)

# Set the URL of the dataset
dataset_url = "https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/download?datasetVersionNumber=1"

# Set the path where the dataset will be extracted
current_directory = os.getcwd()
data_path = os.path.join(current_directory, "data")

# Create the data folder if it doesn't exist
os.makedirs(data_path, exist_ok=True)

# Download and extract the dataset
download_and_extract_data(dataset_url, data_path)

# Select the imaging modalities to use (3 out of 4)
modalities = ["t1", "t1ce", "flair"]

import random

# Load and preprocess the data
def load_data(data_path, modalities, fraction=1.0):
    data_list = []
    mask_list = []

    # Traverse all patients
    patient_folders = os.listdir(data_path)
    random.shuffle(patient_folders)
    num_patients = int(len(patient_folders) * fraction)
    selected_patients = patient_folders[:num_patients]

    for patient_folder in selected_patients:
        patient_path = os.path.join(data_path, patient_folder)

        # Load and stack the selected modalities
        image_data = []
        for modality in modalities:
            modality_file = os.path.join(patient_path, f"{modality}.nii.gz")
            modality_data = nib.load(modality_file).get_fdata()
            image_data.append(modality_data)

        # Load the ground truth segmentation mask
        mask_file = os.path.join(patient_path, "seg.nii.gz")
        mask_data = nib.load(mask_file).get_fdata()

        data_list.append(np.stack(image_data, axis=-1))
        mask_list.append(mask_data)

    return np.array(data_list), np.array(mask_list)

# Load 10% of the data
fraction = 0.1
X, y = load_data(data_path, modalities, fraction)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the transfer learning model
def create_segmentation_model(input_shape, n_classes, base_model_name):
    base_model = getattr(tf.keras.applications, base_model_name)(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=output)

# Create the model
input_shape = (*X_train.shape[1:],)
n_classes = 1
base_model_name = "ResNet50V2"
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_segmentation_model(input_shape, n_classes, base_model_name)

    # Compile the model
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Set up the TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=f'./logs/{base_model_name}',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1
)

# Train the model
epochs = 100
batch_size_per_gpu = 4
num_gpus = 2
total_batch_size = batch_size_per_gpu * num_gpus

history = model.fit(
    X_train, y_train,
    batch_size=total_batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    shuffle=True,
    callbacks=[tensorboard_callback]
)

# Evaluate the model
scores = model.evaluate(X_val, y_val, batch_size=total_batch_size, verbose=1)
print("Validation loss:", scores[0])
print("Validation accuracy:", scores[1])

# Save the model
model.save("brain_tumor_segmentation_model.h5")
