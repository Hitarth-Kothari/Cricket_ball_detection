import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the provided ball image
ball_image_path = 'data/color.jpg'
ball_img = cv2.imread(ball_image_path)

# Convert to HSV color space
hsv_ball = cv2.cvtColor(ball_img, cv2.COLOR_BGR2HSV)

# Get the dominant color in the HSV space (assuming the ball is the most prominent color)
hsv_values = hsv_ball.reshape((-1, 3))
mean_hsv = np.mean(hsv_values, axis=0)

print(f"Mean HSV value: {mean_hsv}")

# Define a range around the mean HSV value
lower_color = mean_hsv - np.array([10, 50, 50])
upper_color = mean_hsv + np.array([10, 50, 50])

print(f"Lower HSV range: {lower_color}")
print(f"Upper HSV range: {upper_color}")

# Visualize the ball image
plt.imshow(cv2.cvtColor(ball_img, cv2.COLOR_BGR2RGB))
plt.title('Ball Image')
plt.show()
