import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

input_dir='/Indian_pines (1).mat'
output_dir='/content/saves'
data=loadmat(input_dir)

print(data.keys())
key_val=data['indian_pines']

image=key_val[0,0]
print(image.shape)

image_normalized = cv.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
blurred = cv.GaussianBlur(image, (5, 5), 0)
_, thresh_simple = cv.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image_normalized, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Thresholded image
plt.subplot(1, 2, 2)
plt.imshow(thresh_simple, cmap='gray')
plt.title('Simple Thresholding')
plt.axis('off')

plt.show()
