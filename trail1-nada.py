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
