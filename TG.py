import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Function to list files in a directory
def list_files(directory_path, file_extension):
    try:
        return [f for f in os.listdir(directory_path) if f.endswith(file_extension)]
    except FileNotFoundError:
        return []

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values
    return image_array

# Function to generate data and labels
def generate_data_and_labels(image_files, images_path, labels_path):
    data = []
    labels = []
    missing_labels = []
    label_contents = []
    for image_file in image_files:
        image_path = os.path.join(images_path, image_file)
        image_data = load_and_preprocess_image(image_path)
        
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_path, label_file)
        
        if not os.path.exists(label_path):
            missing_labels.append(label_file)
            continue
        
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            label_contents.append((image_file, label_file, label_content))
            
            # Convert to binary label
            if any(cls_id in label_content.split()[0] for cls_id in ["1", "2", "3"]):
                labels.append(1)
            else:
                labels.append(0)
        
        data.append(image_data)
    return np.array(data), np.array(labels), missing_labels, label_contents

# Define the CNN model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # For binary classification, 2 output neurons
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Streamlit app
def main():
    st.title("Turbine Guardian: Predictive Maintenance for Turbines")

    st.subheader("About Us")
    st.write("""
        Turbine Guardian leverages the power of AI and Machine Learning to provide predictive maintenance for turbines. By analyzing images of turbine components, the application can identify defects before they lead to serious issues, thereby reducing downtime and maintenance costs.
    """)

    st.subheader("Upload Data")
    uploaded_zip = st.file_uploader("Upload a ZIP file containing images and labels", type=['zip'])

    if uploaded_zip:
        with st.spinner('Extracting files...'):
            # Save and extract the ZIP file
            zip_path = 'uploaded_data.zip'
            with open(zip_path, 'wb') as f:
                f.write(uploaded_zip.getvalue())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('extracted_data')
            
            # Debug: Check the extracted directory structure
            for root, dirs, files in os.walk('extracted_data'):
                st.write(f"Found directory: {root}")
                st.write(f"Subdirectories: {dirs}")
                st.write(f"Files: {files}")

            # Define paths
            train_images_path = os.path.join('extracted_data', 'Aero-engine_defect-detect_new', 'images', 'train')
            train_labels_path = os.path.join('extracted_data', 'Aero-engine_defect-detect_new', 'labels', 'train')
            val_images_path = os.path.join('extracted_data', 'Aero-engine_defect-detect_new', 'images', 'val')
            val_labels_path = os.path.join('extracted_data', 'Aero-engine_defect-detect_new', 'labels', 'val')

            # Ensure the directories exist
            if not os.path.exists(train_images_path):
                st.error(f"Directory not found: {train_images_path}")
                return
            if not os.path.exists(train_labels_path):
                st.error(f"Directory not found: {train_labels_path}")
                return
            if not os.path.exists(val_images_path):
                st.error(f"Directory not found: {val_images_path}")
                return
            if not os.path.exists(val_labels_path):
                st.error(f"Directory not found: {val_labels_path}")
                return

            train_image_files = list_files(train_images_path, '.jpg')
            train_label_files = list_files(train_labels_path, '.txt')
            val_image_files = list_files(val_images_path, '.jpg')
            val_label_files = list_files(val_labels_path, '.txt')

            train_data, train_labels, train_missing_labels, train_label_contents = generate_data_and_labels(train_image_files, train_images_path, train_labels_path)
            val_data, val_labels, val_missing_labels, val_label_contents = generate_data_and_labels(val_image_files, val_images_path, val_labels_path)

            # Convert labels to categorical format
            train_labels_categorical = to_categorical(train_labels, num_classes=2)
            val_labels_categorical = to_categorical(val_labels, num_classes=2)

            if train_missing_labels:
                st.warning(f"Missing training labels for the following files: {train_missing_labels}")
            if val_missing_labels:
                st.warning(f"Missing validation labels for the following files: {val_missing_labels}")

            st.subheader("Data Summary")
            st.write(f"Training class distribution:\n{pd.Series(train_labels).value_counts()}")
            st.write(f"Validation class distribution:\n{pd.Series(val_labels).value_counts()}")
            st.write(f"Training data shape: {train_data.shape}")
            st.write(f"Training labels shape: {train_labels.shape}")
            st.write(f"Validation data shape: {val_data.shape}")
            st.write(f"Validation labels shape: {val_labels.shape}")

            st.subheader("Sample Images")
            st.write("Training Data Samples")
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for i, ax in enumerate(axes):
                index = np.random.choice(len(train_data))
                ax.imshow(train_data[index])
                ax.set_title(f"Label: {train_labels[index]}")
                ax.axis('off')
            st.pyplot(fig)

            st.write("Validation Data Samples")
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for i, ax in enumerate(axes):
                index = np.random.choice(len(val_data))
                ax.imshow(val_data[index])
                ax.set_title(f"Label: {val_labels[index]}")
                ax.axis('off')
            st.pyplot(fig)

            # Model selection
            st.subheader("Choose Model")
            model_type = st.selectbox("Select Model", ["CNN", "SVM"])

            if model_type == "CNN":
                model = create_cnn_model((224, 224, 3))
                # Train the CNN model
                history = model.fit(train_data, train_labels_categorical, epochs=10, batch_size=32, validation_data=(val_data, val_labels_categorical))

                # Evaluate the model on the validation set
                val_loss, val_accuracy = model.evaluate(val_data, val_labels_categorical)
                st.write(f'Validation accuracy: {val_accuracy:.4f}')

                # Plot model performance
                st.subheader("Model Performance")
                st.write("Training and validation accuracy")
                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'], label='Train Accuracy')
                ax.plot(history.history['val_accuracy'], label='Val Accuracy')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)

                st.write("Training and validation loss")
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Train Loss')
                ax.plot(history.history['val_loss'], label='Val Loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)
            else:
                # Flatten the data for SVM
                train_data_flat = train_data.reshape(len(train_data), -1)
                val_data_flat = val_data.reshape(len(val_data), -1)
                
                # Standardize the data
                scaler = StandardScaler()
                train_data_flat = scaler.fit_transform(train_data_flat)
                val_data_flat = scaler.transform(val_data_flat)

                # Train SVM model
                svm_model = SVC(kernel='linear')
                svm_model.fit(train_data_flat, train_labels)

                # Predict and evaluate
                val_predictions = svm_model.predict(val_data_flat)
                report = classification_report(val_labels, val_predictions, target_names=["Class 0", "Class 1"], output_dict=True)
                st.write(pd.DataFrame(report).transpose())

if __name__ == "__main__":
    main()
