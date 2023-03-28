#iter_10p_train

#this works on 10percent_train.py and iterates all possible combinations of up to 3 modalities.

## use in terminal: tensorboard --logdir=./logs


import os
import zipfile
import requests
import tarfile
import random
import numpy as np
import nibabel as nib
from itertools import combinations
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

# Download and extract the dataset
def download_and_extract_data(url, data_path):
    response = requests.get(url, stream=True)
    zip_file_path = os.path.join(data_path, "archive.zip")
    
    with open(zip_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)

    tar_file_path = os.path.join(data_path, "BraTS2021_Training_Data.tar")
    with tarfile.open(tar_file_path, 'r') as tar_ref:
        tar_ref.extractall(data_path)

def load_data(data_path, modalities, fraction=1.0):
    data_list = []
    mask_list = []

    patient_folders = os.listdir(data_path)
    random.shuffle(patient_folders)
    num_patients = int(len(patient_folders) * fraction)
    selected_patients = patient_folders[:num_patients]

    for patient_folder in selected_patients:
        patient_path = os.path.join(data_path, patient_folder)

        image_data = []
        for modality in modalities:
            modality_file = os.path.join(patient_path, f"{modality}.nii.gz")
            modality_data = nib.load(modality_file).get_fdata()
            image_data.append(modality_data)

        mask_file = os.path.join(patient_path, "seg.nii.gz")
        mask_data = nib.load(mask_file).get_fdata()

        data_list.append(np.stack(image_data, axis=-1))
        mask_list.append(mask_data)

    return np.array(data_list), np.array(mask_list)

def create_segmentation_model(input_shape, base_model_name):
    base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights=None)
    
    x = base_model.output
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=x)

def generate_modality_combinations(modalities):
    combinations_list = []
    for i in range(1, len(modalities) + 1):
        for subset in combinations(modalities, i):
            combinations_list.append(list(subset))
    return combinations_list

dataset_url = "https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/download?datasetVersionNumber=1"
current_directory = os.getcwd()
data_path = os.path.join(current_directory, "data")
os.makedirs(data_path, exist_ok=True)
download_and_extract_data(dataset_url, data_path)

all_modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
modality_combinations = generate_modality_combinations(all_modalities)

base_model_name = "ResNet50V2"
strategy = tf.distribute.MirroredStrategy()
batch_size_per_gpu = 4
num_gpus = 2
total_batch_size = batch_size_per_gpu * num_gpus

for modalities in modality_combinations:
    print(f"Training with modalities: {modalities}")

    fraction = 0.1
    X, y = load_data(data_path, modalities, fraction)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and compile the model
    model = create_segmentation_model(input_shape=X_train.shape[1:], base_model_name=base_model_name)

    # Set up the TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir=f'./logs/{base_model_name}_{"_".join(modalities)}',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=1
    )

    # Train the model
    epochs = 100
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
    model.save(f"brain_tumor_segmentation_model_{'_'.join(modalities)}.h5")
