from turtle import color
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def add_noise(img, min_pixel_range, max_pixel_range, noise_color):
    row , col = img.shape
    
    number_of_pixels = random.randint(min_pixel_range, max_pixel_range)
    
    for i in range(number_of_pixels):
        y=random.randint(0, col - 1)
        
        x=random.randint(0, row - 1)
        
        img[x][y] = noise_color
    
    return img

gray = cv2.imread("./s.jpg", 0)

tmp = np.copy(gray)

white_noised_image = add_noise(tmp, 5, 100000, 255)

both_noised_image = add_noise(white_noised_image, 5, 100000, 0)

average_kernel = np.ones((7,7))/49
gaussian_kernel = np.array(([1,2,1], [2,4,2], [1,2,1]))/16 

average_kernel_image_gray = cv2.filter2D(gray, -1, average_kernel)
average_kernel_imag_noised = cv2.filter2D(both_noised_image, -1, average_kernel)
gaussian_kernel_image = cv2.filter2D(both_noised_image, -1, gaussian_kernel)
median_kernel_image = median = cv2.medianBlur(both_noised_image,3)

plt.figure(figsize = (15, 15))

plt.subplot(2,3,1)
plt.title("Grayscale")
plt.imshow(gray, cmap = "gray")

plt.subplot(2,3,2)
plt.title("Average filtered gray")
plt.imshow(average_kernel_image_gray, cmap = "gray")

plt.subplot(2,3,3)
plt.title("Salt and paper noised")
plt.imshow(both_noised_image, cmap = "gray")

plt.subplot(2,3,4)
plt.title("Average filtered noised image")
plt.imshow(average_kernel_imag_noised, cmap = "gray")

plt.subplot(2,3,5)
plt.title("Gausian filtered noised image")
plt.imshow(gaussian_kernel_image, cmap = "gray")

plt.subplot(2,3,6)
plt.title("Median filtered noised image")
plt.imshow(median_kernel_image, cmap = "gray")

plt.savefig("Salt and paper")
plt.show()