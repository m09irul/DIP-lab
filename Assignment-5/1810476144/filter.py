import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread('s.jpg', 0)

laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
sobel_kernel = np.array([[-1,0, 1],[-2,0,2],[-1,0, 1]])

processed_img1 = cv2.filter2D(gray, -1, laplacian_kernel)
processed_img2 = cv2.filter2D(gray, -1, sobel_kernel)

plt.figure(figsize = (10, 10))

plt.subplot(1,3,1)
plt.title("Grayscale")
plt.imshow(gray, cmap = 'gray')

plt.subplot(1,3,2)
plt.title("laplacian filter")
plt.imshow(processed_img1, cmap = 'gray')

plt.subplot(1,3,3)
plt.title("Sobel filter")
plt.imshow(processed_img2, cmap = 'gray')

plt.savefig('Filter')
plt.show()
