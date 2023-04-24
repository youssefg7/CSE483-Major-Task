import cv2
import numpy as np
import os

# Load test images from SVHN dataset
test_images = []
folder_path = 'F:/Semester 8/Computer Vision/Project/test'

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(folder_path, filename))
        test_images.append(image)

# Preprocess images
for i in range(len(test_images)):
    # Convert to grayscale
    gray = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours to extract digits
    for contour in contours:
        # Get bounding box of contour
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out contours that are too small or too large to be digits
        if h >= 10 and h <= 100 and w >= 5 and w <= 50:
            # Extract digit and resize to 28x28
            digit = thresh[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28, 28))
            # Save digit as new image file
            cv2.imwrite('digit' + str(i) + '.png', digit)

            # Draw bounding box around digit on original image
            cv2.rectangle(test_images[i], (x, y),
                          (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Localized Digits', test_images[i])
    cv2.waitKey(0)

cv2.destroyAllWindows()



#######################################################################################################
########## Modified one: worked on non croped images "not best solution still errors occured"########## 
#######################################################################################################

import cv2
import numpy as np
import os

# Define function to localize digits in an image


def localize_digits(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to enhance contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 10, 5)
    # Apply median filtering to remove noise
    thresh = cv2.medianBlur(thresh, 3)
    # Apply Sobel operator to detect edges
    grad_x = cv2.Sobel(thresh, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(thresh, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    # Find contours in the image
    contours, hierarchy = cv2.findContours(
        edge.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize list to store bounding boxes
    bounding_boxes = []
    # Loop over contours and extract bounding boxes for digits
    for contour in contours:
        # Get bounding box coordinates for contour
        x, y, w, h = cv2.boundingRect(contour)
        # Ignore small contours
        if w < 10 or h < 10:
            continue
        # Ignore contours that are too elongated
        aspect_ratio = w / float(h)
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue
        # Calculate area of bounding box
        area = w * h
        # Ignore bounding boxes that are too small or too large
        if area < 100 or area > 10000:
            continue
        # Add bounding box to list
        bounding_boxes.append((x, y, w, h))
    # Return list of bounding boxes
    return bounding_boxes


# Define input and output directories
input_dir = 'F:/Semester 8/Computer Vision/Project/filter'
output_dir = "F:/Semester 8/Computer Vision/Project/Output"

# Loop over images in input directory, resize to 64x64, and localize digits
for filename in os.listdir(input_dir):
    # Load image
    image = cv2.imread(os.path.join(input_dir, filename))
    # Resize image to 64x64
    image = cv2.resize(image, (64, 64))
    # Localize digits in image
    digit_boxes = localize_digits(image)
    # Draw bounding boxes on image
    for box in digit_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Show image with bounding boxes
    cv2.imshow('Localized Digits', image)
    cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
