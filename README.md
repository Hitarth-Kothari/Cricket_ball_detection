# Cricket Ball Detection

This project aims to detect cricket balls in images using a Convolutional Neural Network (CNN). The CNN is trained to identify patches in an image that contain the cricket ball. 

## Table of Contents
- [Cricket Ball Detection](#cricket-ball-detection)
- [Table of Contents](#table-of-contents)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Detecting the Ball](#detecting-the-ball)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Structure

The project directory contains the following files and folders:

Cricket_ball_detection/
│
├── Data/
│ ├── train/
│ │ ├── *.jpg
│ ├── annotations.csv
│ ├── image.png
│
├── env/
│ ├── Scripts/
│ │ ├── activate
│
├── main.py
├── requirements.txt
├── README.md


- `Data/train/`: Contains training images.
- `Data/annotations.csv`: Contains bounding box annotations for the training images.
- `Data/image.png`: Example image for ball detection.
- `env/`: Virtual environment directory.
- `main.py`: Main script for training and detecting the ball.
- `requirements.txt`: Lists the required packages for the project.
- `README.md`: This README file.

## Installation

1. Clone the repository:
   ```sh
    git clone https://github.com/your_username/Cricket_ball_detection.git
    cd Cricket_ball_detection
2. Create and activate a virtual environment:
    ```sh
    python -m venv env
    env\Scripts\activate  # On Windows
    source env/bin/activate  # On Unix-based systems
3. Install the required packages:
    ```sh
    pip install -r requirements.txt

## Usage

### Training the Model

1. Place your training images in the Data/train/ directory.

2. Ensure the annotations.csv file in Data/ contains the bounding box annotations for the training images.

3. Run the main.py script to train the model:
    ```sh
    python main.py
### Detecting the Ball

1. After training, place the test image in the Data/ directory and update the example_image_path variable in main.py to point to this image.

2. Depending on the color of the vall update the color range threshhold variable

2. Run the main.py script to detect the ball
    ```sh
    python main.py

## Results

The script will display the detected ball with a bounding box around it in the test image.

## Acknowledgements

1. This project uses OpenCV for image processing and contour detection.
2. The CNN model is built and trained using Keras.
3. Thanks to Labellerr for code ideas (https://www.labellerr.com/blog/cricket-ball-detection/).
4. Thanks to Roboflow for the dataset (https://universe.roboflow.com/cricket-2rxrt/cricket-ball-detection/dataset/1).
