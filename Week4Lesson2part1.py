import cv2
import numpy as np

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread("Ronaldo.jpg")
if img is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# -----------------------------
# Harris Corner Detection
# -----------------------------
block_size = 2      # Neighborhood size
ksize = 3           # Aperture parameter for Sobel operator
k = 0.04            # Harris detector free parameter

harris_response = cv2.cornerHarris(gray, block_size, ksize, k)

# Dilate the result to make the corners more visible
harris_response = cv2.dilate(harris_response, None)

# Threshold for marking strong corners
threshold = 0.01 * harris_response.max()

# Make a copy to draw points
output = img.copy()

# Mark corners in red
output[harris_response > threshold] = [0, 0, 255]

# -----------------------------
# Display Results
# -----------------------------
cv2.imshow("Original Image", img)
cv2.imshow("Harris Corners", output)
cv2.waitKey(0)
cv2.destroyAllWindows()