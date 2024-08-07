import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.utils import to_categorical

# Global debug flag for visualization
DEBUG = True

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

def find_and_draw_contours(img, mask):
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
    if DEBUG:
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    return contours, img_contours

def extract_patches(img, contours):
    """
    Extract patches from the image using contours, and show patches if debugging is enabled.

    Parameters:
    img (np.array): Original image.
    contours (list): List of contours found in the image.

    Returns:
    tuple: List of extracted patches and a DataFrame with their bounding boxes.
    """
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

        if ratio >= 0.5 and ((w <= 50) and (h <= 50)):
            patch = img[ymin:ymax, xmin:xmax]
            patches.append(cv2.resize(patch, (25, 25), interpolation=cv2.INTER_AREA))
            df.loc[cnt] = [x, y, w, h]
            cnt += 1
            if DEBUG:
                plt.figure()
                plt.title(f"Patch {cnt}")
                plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                plt.show()

    print(f"Number of patches extracted: {len(patches)}")
    return patches, df

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

    features = np.array([cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches])
    features = features.reshape(features.shape[0], 25, 25, 1) / 255.0

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
            num = 20
            xmin, ymin = x - num, y - num
            xmax, ymax = x + w + num, y + h + num
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("Ball not detected with high confidence.")
    else:
        print("Ball not detected in the image.")

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

def build_cnn_model():
    """
    Build a Convolutional Neural Network model for image classification.

    Returns:
    keras.Model: Compiled CNN model.
    """
    model = Sequential()
    model.add(Input(shape=(25, 25, 1)))  # Explicitly define input shape using Input layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_data_for_cnn(image_dir, annotations_df):
    """
    Prepare data for training the CNN model by extracting image patches and labels.

    Parameters:
    image_dir (str): Directory containing image files.
    annotations_df (pd.DataFrame): DataFrame with annotations for image files.

    Returns:
    tuple: A tuple containing the images array and labels array.
    """
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

def get_thresholds_for_color(color):
    """
    Get HSV thresholds for detecting a ball of a specified color.

    Parameters:
    color (str): Color of the ball ('red' or 'white').

    Returns:
    list: List of HSV thresholds for color detection.
    """
    if color.lower() == 'red':
        return [np.array([0, 0, 200]), np.array([180, 50, 255]),
                np.array([0, 0, 200]), np.array([180, 50, 255])]
    elif color.lower() == 'white':
        return [np.array([0, 0, 230]), np.array([180, 20, 255]),
                np.array([0, 0, 230]), np.array([180, 20, 255])]
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
    example_image_path = 'Data/test.jpg'  # Replace with the path to an example image
    
    # Get thresholds for detecting a red ball
    thresholds = get_thresholds_for_color('white')
    
    # Detect the ball in the image
    detect_ball_in_image(example_image_path, cnn_model, thresholds)

if __name__ == "__main__":
    main()
