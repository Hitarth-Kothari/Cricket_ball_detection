import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical

# Global debug flag for visualization
DEBUG = True

# Global list of colors to detect
COLORS = ['red', 'white']

# Global flag to enable or disable middle third cropping
CROP_MIDDLE_THIRD = True

def load_annotations(csv_file):
    """
    Load annotations from a CSV file.

    Parameters:
    csv_file (str): Path to the CSV file containing annotations.

    Returns:
    pd.DataFrame: DataFrame containing the annotations.
    """
    return pd.read_csv(csv_file)

def draw_boxes(image, boxes):
    """
    Draw bounding boxes on an image.

    Parameters:
    image (np.array): Image on which to draw the boxes.
    boxes (pd.DataFrame): DataFrame containing the bounding box coordinates.

    Returns:
    np.array: Image with drawn bounding boxes.
    """
    for _, box in boxes.iterrows():
        cv2.rectangle(image, (int(box['xmin']), int(box['ymin'])), 
                      (int(box['xmax']), int(box['ymax'])), (255, 0, 0), 2)
    return image

def read_and_preprocess_image(image_path):
    """
    Read and preprocess the image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple: Tuple containing the original image and its HSV conversion.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv

def create_color_mask(hsv, thresholds):
    """
    Create a color mask for the specified HSV thresholds.

    Parameters:
    hsv (np.array): HSV image.
    thresholds (list): List containing lower and upper HSV threshold values.

    Returns:
    np.array: Mask created from the color thresholds.
    """
    lower_thresh1, upper_thresh1, lower_thresh2, upper_thresh2 = thresholds
    mask1 = cv2.inRange(hsv, lower_thresh1, upper_thresh1)
    mask2 = cv2.inRange(hsv, lower_thresh2, upper_thresh2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_and_draw_contours(img, mask, train=False):
    """
    Find contours from the mask and draw them on the image if debugging is enabled.

    Parameters:
    img (np.array): Original image.
    mask (np.array): Mask created from color thresholds.

    Returns:
    tuple: Contours found in the mask and a copy of the image with contours drawn.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()
    if DEBUG and not train:
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    return contours, img_contours

def extract_patches(img, contours, train=False, distance_threshold=10):
    """
    Extract patches from the image using contours, centering on the contour and extracting a fixed size patch.
    Include checks for aspect ratio, area, circularity, and ensure no contours are too close to each other to maintain isolation.

    Parameters:
    img (np.array): Original image.
    contours (list): List of contours found in the image.
    train (bool): If True, skip displaying debug information.
    distance_threshold (int): Minimum distance between contour centers to consider them isolated.

    Returns:
    tuple: List of extracted patches and a DataFrame with their bounding boxes.
    """
    patches = []
    dfs = []
    used_contours = set()
    patch_size = 50  # Fixed patch size

    for index, contour in enumerate(contours):
        if index in used_contours:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        # Check for aspect ratio, area, and circularity to filter likely targets
        if 0.7 <= aspect_ratio <= 1.3 and 10 <= area <= 2000 and circularity > 0.5:
            cx, cy = x + w // 2, y + h // 2  # Center of the contour
            patch_x, patch_y = max(0, cx - patch_size // 2), max(0, cy - patch_size // 2)

            # Ensure the patch does not go outside the image boundaries
            patch_x = min(patch_x, img.shape[1] - patch_size)
            patch_y = min(patch_y, img.shape[0] - patch_size)

            # Check for isolation
            isolated = True
            for other_index, other_contour in enumerate(contours):
                if other_index == index or other_index in used_contours:
                    continue
                ox, oy, ow, oh = cv2.boundingRect(other_contour)
                ocx, ocy = ox + ow // 2, oy + oh // 2
                distance = np.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                if distance < distance_threshold:
                    isolated = False
                    break

            if isolated:
                # Extract the patch
                patch = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
                patches.append(patch)
                used_contours.add(index)

                # Store data for potential use or analysis
                dfs.append({
                    'contour_index': index, 'x': patch_x, 'y': patch_y, 'w': patch_size, 'h': patch_size,
                    'center_x': cx, 'center_y': cy, 'used_for_patch': True
                })

                # Debugging each patch's details
                if DEBUG and not train:
                    plt.figure()
                    plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                    plt.title(f"Patch {index}: CenterX={cx}, CenterY={cy}")
                    plt.show()

    df = pd.DataFrame(dfs)
    return patches, df

def check_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.

    Parameters:
    bbox1 (tuple): Bounding box (x, y, w, h).
    bbox2 (tuple): Bounding box (x, y, w, h).

    Returns:
    bool: True if there is an overlap, False otherwise.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
        return True
    return False

def extract_positive_samples(image_dir, annotations_df):
    """
    Extract positive samples (patches containing the ball) from images.

    Parameters:
    image_dir (str): Directory containing image files.
    annotations_df (pd.DataFrame): DataFrame with annotations for image files.

    Returns:
    tuple: A tuple containing lists of positive image patches and their labels.
    """
    positive_images = []
    positive_labels = []
    
    for _, row in annotations_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        img = load_image(image_path)
        
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        positive_patch = extract_positive_patch(img, bbox)
        positive_images.append(positive_patch)
        positive_labels.append(1)  # 1 means ball is present

    return positive_images, positive_labels

def extract_positive_patch(img, bbox):
    """
    Extract a centered positive image patch containing the ball.

    Parameters:
    img (np.array): Image from which to extract the patch.
    bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
    np.array: Centered positive image patch.
    """
    x_min, y_min, x_max, y_max = bbox
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    patch_size = 50
    patch_x, patch_y = max(0, cx - patch_size // 2), max(0, cy - patch_size // 2)

    # Ensure the patch does not go outside the image boundaries
    patch_x = min(patch_x, img.shape[1] - patch_size)
    patch_y = min(patch_y, img.shape[0] - patch_size)

    # Extract the patch
    patch = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    return patch

def extract_negative_samples(image, positive_bboxes):
    """
    Extract negative samples from an image using all available color filters,
    excluding regions that overlap with positive samples.

    Parameters:
    image (np.array): Original image to process.
    positive_bboxes (list): List of bounding boxes for positive samples.

    Returns:
    list: List of negative samples.
    """
    negative_patches = []
    skipped_images = 0  # Counter for skipped images due to errors

    for color in COLORS:
        try:
            thresholds = get_thresholds_for_color(color)
            mask = create_color_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), thresholds)
            contours, _ = find_and_draw_contours(image, mask, True)
            
            # Extract patches from all contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not any(check_overlap((x, y, w, h), bbox) for bbox in positive_bboxes):
                    patch = image[y:y + h, x:x + w]
                    patch_resized = cv2.resize(patch, (50, 50), interpolation=cv2.INTER_AREA)
                    negative_patches.append(patch_resized)
                    if DEBUG:
                        print(f"Negative patch extracted from contour: {x, y, w, h}")

        except Exception as e:
            skipped_images += 1
            print(f"Error processing color {color}: {e}")

    if DEBUG:
        print(f"Total negative patches extracted: {len(negative_patches)}")
        print(f"Total skipped images due to errors: {skipped_images}")

    return negative_patches

def prepare_data_for_cnn(image_dir, annotations_df):
    """
    Prepare data for training the CNN model by extracting image patches and labels.

    Parameters:
    image_dir (str): Directory containing image files.
    annotations_df (pd.DataFrame): DataFrame with annotations for image files.

    Returns:
    tuple: A tuple containing the images array and labels array.
    """
    images, labels = [], []

    # Extract positive samples
    positive_images, positive_labels = extract_positive_samples(image_dir, annotations_df)
    images.extend(positive_images)
    labels.extend(positive_labels)

    # Calculate the number of positive samples
    num_positive_samples = len(positive_labels)

    # Initialize counters for debugging
    total_negative_extracted = 0
    total_negative_used = 0

    # Generate random negative samples (not containing the ball)
    negative_images = []
    for _, row in annotations_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        img = load_image(image_path)

        # Extract positive bounding box
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

        negative_patches = extract_negative_samples(img, [bbox])

        # Update total extracted negatives
        total_negative_extracted += len(negative_patches)

        for negative_patch in negative_patches:
            if negative_patch.shape == (50, 50, 3):  # Check if the negative patch is the correct size and channels
                negative_images.append(negative_patch)

    # Shuffle negative images and select a subset equal to the number of positive samples
    random.shuffle(negative_images)
    selected_negative_images = negative_images[:num_positive_samples]

    # Add selected negative samples to the images and labels
    images.extend(selected_negative_images)
    labels.extend([0] * len(selected_negative_images))

    # Update total used negatives
    total_negative_used = len(selected_negative_images)

    if DEBUG:
        print(f"Total negative patches extracted: {total_negative_extracted}")
        print(f"Total negative patches used: {total_negative_used}")
        print(f"Total positive patches used: {num_positive_samples}")

    images = np.array(images).reshape(-1, 50, 50, 3) / 255.0
    labels = np.array(labels)

    return images, labels

def train_cnn_model(model, images, labels):
    """
    Train the CNN model using the prepared image patches and labels.

    Parameters:
    model (keras.Model): CNN model to be trained.
    images (np.array): Array of image patches.
    labels (np.array): Array of labels for the image patches.
    """
    labels = to_categorical(labels, 2)
    x_tr, x_val, y_tr, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=0)
    model.fit(x_tr, y_tr, epochs=10, validation_data=(x_val, y_val))
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    print(classification_report(y_val_classes, y_pred_classes))

def build_cnn_model():
    """
    Build a more complex Convolutional Neural Network model for image classification.

    Returns:
    keras.Model: Compiled CNN model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=(50, 50, 3)))
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Third convolutional block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(2, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def load_image(image_path):
    """
    Load an image in grayscale.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    np.array: Loaded image in grayscale.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to load image at {image_path}")
    return img

def detect_ball_in_image(image_path, model, thresholds):
    """
    Detect a ball in an image using color thresholding and a trained CNN model.

    Parameters:
    image_path (str): Path to the image file.
    model (keras.Model): Trained CNN model for ball detection.
    thresholds (list): List of HSV thresholds for ball color detection.
    """
    img, hsv = read_and_preprocess_image(image_path)
    if img is None or hsv is None:
        return

    if CROP_MIDDLE_THIRD:
        # Crop the image to the middle third
        img = crop_middle_third(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = create_color_mask(hsv, thresholds)

    if DEBUG:
        plt.figure()
        plt.title("Color Mask")
        plt.imshow(mask, cmap='gray')
        plt.show()

    contours, img_contours = find_and_draw_contours(img, mask)

    if DEBUG:
        plt.figure()
        plt.title("Contours")
        plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
        plt.show()

    patches, df = extract_patches(img, contours)

    classify_patches(patches, model, df, img)

def classify_patches(patches, model, df, img):
    """
    Classify extracted patches to detect the ball.

    Parameters:
    patches (list): List of image patches.
    model (keras.Model): Trained CNN model for ball detection.
    df (pd.DataFrame): DataFrame containing bounding boxes of patches.
    img (np.array): Original image for drawing detected balls.
    """
    if len(patches) == 0:
        print("No patches found in the image.")
        return

    # Ensure patches are in 3-channel RGB format
    features = np.array(patches).reshape(-1, 50, 50, 3) / 255.0

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
            if not df.empty and 'x' in df.columns:
                x, y, w, h = df.loc[ball_idx, ['x', 'y', 'w', 'h']]
                num = 20
                xmin, ymin = x - num, y - num
                xmax, ymax = x + w + num, y + h + num
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            else:
                print("DataFrame is empty or missing required columns.")
        else:
            print("Ball not detected with high confidence.")
    else:
        print("Ball not detected in the image.")

def crop_middle_third(image):
    """
    Crop the middle third of the image horizontally.

    Parameters:
    image (np.array): Original image.

    Returns:
    np.array: Cropped image containing the middle third.
    """
    height, width = image.shape[:2]
    third_width = width // 3
    middle_third = image[:, third_width:2*third_width]
    
    if DEBUG:
        plt.figure()
        plt.title("Middle Third")
        plt.imshow(cv2.cvtColor(middle_third, cv2.COLOR_BGR2RGB))
        plt.show()
    
    return middle_third

def get_thresholds_for_color(color):
    """
    Get HSV thresholds for detecting a ball of a specified color.

    Parameters:
    color (str): Color of the ball ('red' or 'white').

    Returns:
    list: List of HSV thresholds for color detection.
    """
    if color.lower() == 'red':
        return [
            np.array([0, 70, 50]), np.array([10, 255, 255]),   # Lower red range
            np.array([170, 70, 50]), np.array([180, 255, 255]) # Upper red range
        ]
    elif color.lower() == 'white':
        return [
            np.array([0, 0, 200]), np.array([180, 50, 255]),
            np.array([0, 0, 200]), np.array([180, 50, 255])
        ]
    else:
        raise ValueError("Color not recognized. Please use 'red' or 'white'.")

def main():
    """
    Main function to orchestrate the loading, training, and detection process.
    """
    global DEBUG
    
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
    example_image_path = 'Data/image1.png'  # Replace with the path to an example image
    
    # Get thresholds for detecting a white ball
    thresholds = get_thresholds_for_color('white')
    
    # Detect the ball in the image
    detect_ball_in_image(example_image_path, cnn_model, thresholds)

if __name__ == "__main__":
    main()
