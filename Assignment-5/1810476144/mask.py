import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread('s.jpg', 0)

mask = np.zeros(gray.shape, dtype=np.uint8)

mask = cv2.circle(mask, (260, 300), 255, (255,255,255), -1) 

result = cv2.bitwise_and(gray, mask)

plt.figure(figsize = (10, 10))

plt.subplot(1,3,1)
plt.title("Grayscale")
plt.imshow(gray, cmap = 'gray')

plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask, cmap = 'gray')

plt.subplot(1,3,3)
plt.title("Masked result")
plt.imshow(result, cmap = 'gray')

plt.savefig('Masking')
#plt.show()
