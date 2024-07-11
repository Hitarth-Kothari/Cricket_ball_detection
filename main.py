import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Function to load annotations from CSV file
def load_annotations(csv_file):
    return pd.read_csv(csv_file)

# Function to draw bounding boxes on an image
def draw_boxes(image, boxes):
    for _, box in boxes.iterrows():
        cv2.rectangle(image, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), (255, 0, 0), 2)
    return image

# Function to process an image and detect the ball
def detect_ball_in_image(image_path, model, lower_thresh1, upper_thresh1, lower_thresh2, upper_thresh2):
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image is loaded properly
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for the ball color range
    mask1 = cv2.inRange(hsv, lower_thresh1, upper_thresh1)
    mask2 = cv2.inRange(hsv, lower_thresh2, upper_thresh2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Visualize the mask
    plt.figure()
    plt.title("Color Mask")
    plt.imshow(mask, cmap='gray')
    plt.show()
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualize contours
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    plt.figure()
    plt.title("Contours")
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Extract patches from the image using the contours
    patches = []
    df = pd.DataFrame(columns=['x', 'y', 'w', 'h'])
    num = 20
    cnt = 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        numer = min([w, h])
        denom = max([w, h])
        ratio = numer / denom

        if x >= num and y >= num:
            xmin, ymin = x - num, y - num
            xmax, ymax = x + w + num, y + h + num
        else:
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h

        if ratio >= 0.5 and ((w <= 50) and (h <= 50)):  # Adjusted size criteria
            patch = img[ymin:ymax, xmin:xmax]
            patches.append(cv2.resize(patch, (25, 25), interpolation=cv2.INTER_AREA))
            df.loc[cnt] = [x, y, w, h]
            cnt += 1

    # Print the number of patches extracted
    print(f"Number of patches extracted: {len(patches)}")

    # Visualize the extracted patches
    for idx, patch in enumerate(patches):
        plt.figure()
        plt.title(f"Patch {idx}")
        plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        plt.show()

    # Convert patches to features for classification
    features = np.array([cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches])
    features = features.reshape(features.shape[0], 25, 25, 1) / 255.0
    
    # Predict which patches contain the ball
    if len(features) > 0:
        y_pred = model.predict(features)
        predicted_labels = np.argmax(y_pred, axis=1)
        prob = np.max(y_pred, axis=1)

        print(f"Predicted labels: {predicted_labels}")
        print(f"Prediction probabilities: {prob}")

        if 1 in predicted_labels:
            ind = np.where(predicted_labels == 1)[0]
            confidence = prob[ind]
            if len(confidence) > 0:
                maximum = max(confidence)
                ball_idx = ind[list(confidence).index(maximum)]
                x, y, w, h = df.loc[ball_idx]
                
                # Draw bounding box around the detected ball
                xmin, ymin = x - num, y - num
                xmax, ymax = x + w + num, y + h + num
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                
                # Show the image with the detected ball
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            else:
                print("Ball not detected with high confidence.")
        else:
            print("Ball not detected in the image.")
    else:
        print("No patches found in the image.")

# Build a CNN model for identifying the patch containing the ball
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare the data for the CNN
def prepare_data_for_cnn(image_dir, annotations_df):
    images = []
    labels = []
    
    # Read annotations and prepare dataset
    for _, row in annotations_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        img = cv2.imread(image_path, 0)
        
        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        patch = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        patch = cv2.resize(patch, (25, 25), interpolation=cv2.INTER_AREA)
        images.append(patch)
        labels.append(1)  # 1 means ball is present

    images = np.array(images).reshape(-1, 25, 25, 1) / 255.0
    labels = np.array(labels)
    
    return images, labels

# Train the CNN model
def train_cnn_model(model, images, labels):
    labels = to_categorical(labels, 2)
    x_tr, x_val, y_tr, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=0)
    model.fit(x_tr, y_tr, epochs=10, validation_data=(x_val, y_val))
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    print(classification_report(y_val_classes, y_pred_classes))

# Directory paths
image_dir = 'Data/train'  # Replace with the path to your images directory
annotation_csv = 'Data/annotations.csv'  # Replace with the path to your annotations CSV file

# Load annotations
annotations_df = load_annotations(annotation_csv)

# Prepare data for CNN
images, labels = prepare_data_for_cnn(image_dir, annotations_df)

# Build and train the CNN model
cnn_model = build_cnn_model()
train_cnn_model(cnn_model, images, labels)

# Detect the ball in an example image
example_image_path = 'Data/image.png'  # Replace with the path to an example image

# Define the color range of the ball
lower_thresh1 = np.array([0, 0, 200])
upper_thresh1 = np.array([180, 50, 255])
lower_thresh2 = np.array([0, 0, 200])
upper_thresh2 = np.array([180, 50, 255])

detect_ball_in_image(example_image_path, cnn_model, lower_thresh1, upper_thresh1, lower_thresh2, upper_thresh2)
