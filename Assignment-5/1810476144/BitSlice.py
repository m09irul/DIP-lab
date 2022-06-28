import matplotlib.pyplot as plt
import cv2
import numpy as np

# Reading the image in grayscale
img_path = "./s.jpg"
img = cv2.imread(img_path, 0)

plt.figure(figsize = (10, 10))

plt.subplot(3, 3, 1)
plt.imshow(img, cmap = 'gray')
plt.title("Grayscale")

# Change each pixel to it's corresponding binary value
sliced_image = np.zeros([img.shape[0], img.shape[1]], dtype = np.uint8)

for k in range(8):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bin = np.binary_repr(img[i][j], width = 8)
            sliced_image[i][j] = int(bin[7-k])
    
    # Plotting Images
    plt.subplot(3, 3, k+2)
    plt.imshow(sliced_image, cmap = 'gray')
    plt.title("Bit {} image".format(k))

plt.savefig('Bit Slicing')
plt.show()
