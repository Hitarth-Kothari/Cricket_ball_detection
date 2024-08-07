import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Function to load an image and convert it to HSV
def read_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv

# Function to create a color mask based on thresholds
def create_color_mask(hsv, thresholds):
    lower_thresh1, upper_thresh1, lower_thresh2, upper_thresh2 = thresholds
    mask1 = cv2.inRange(hsv, lower_thresh1, upper_thresh1)
    mask2 = cv2.inRange(hsv, lower_thresh2, upper_thresh2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Function to find and draw contours
def find_and_draw_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    return contours, img_contours

# Function to extract and label patches
def extract_and_label_patches(img, contours, image_name, csv_file):
    img_height, img_width = img.shape[:2]
    annotations = []

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Create a copy of the image for displaying the current bounding box
        img_with_bbox = img.copy()
        # Draw all boxes in green
        for cnt in contours:
            xx, yy, ww, hh = cv2.boundingRect(cnt)
            cv2.rectangle(img_with_bbox, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        # Draw the current box in red
        cv2.rectangle(img_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Annotate the current box
        cv2.putText(img_with_bbox, 'Current', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the image with bounding box
        plt.imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
        plt.title(f'Bounding Box {index + 1}')
        plt.show()

        # Ask user for input
        label = input(f"Label for Bounding Box {index + 1} (0 or 1): ")

        if label.strip() == '1':
            annotations.append({
                'filename': image_name,
                'width': img_width,
                'height': img_height,
                'class': 'ball',  # Label as 'ball' for positive patches
                'xmin': x,
                'ymin': y,
                'xmax': x + w,
                'ymax': y + h
            })
            break  # Stop processing further bounding boxes

    # Process existing CSV entries and update with new annotations
    update_annotations_csv(image_name, annotations, csv_file)

def update_annotations_csv(image_name, annotations, csv_file):
    # Read the existing CSV file
    if os.path.isfile(csv_file):
        existing_df = pd.read_csv(csv_file)
        # Remove existing entries for this image
        existing_df = existing_df[existing_df['filename'] != image_name]
    else:
        existing_df = pd.DataFrame()

    # Create a new DataFrame for the current image's annotations
    new_df = pd.DataFrame(annotations)

    # Concatenate the new annotations with the existing ones
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    updated_df.to_csv(csv_file, index=False)



# Main function to process an image
def process_image(image_path, csv_file='annotations.csv', color='red'):
    img, hsv = read_and_preprocess_image(image_path)
    if img is None or hsv is None:
        return

    thresholds = get_thresholds_for_color(color)
    mask = create_color_mask(hsv, thresholds)

    # Display mask and contours for debugging using matplotlib
    plt.imshow(mask, cmap='gray')
    plt.title("Color Mask")
    plt.show()

    contours, img_contours = find_and_draw_contours(img, mask)

    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title("Contours")
    plt.show()

    extract_and_label_patches(img, contours, os.path.basename(image_path), csv_file)

# Function to get HSV thresholds for color detection
def get_thresholds_for_color(color):
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

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    color = input("Enter the color of the ball (red or white): ")
    
    process_image(image_path, csv_file='gathered.csv', color=color)
