import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_image_and_histogram(img_set):
    subplot_row = 2
    subplot_column = 4
    j = 1

    plt.figure(figsize = (15, 15))
    
    for i in img_set:
        plt.subplot(subplot_row, subplot_column, j)
        plt.imshow(img_set[i][0], cmap = 'gray')

        j += 1

        plt.subplot(subplot_row, subplot_column, j)
        plt.title(i)
        plt.plot(img_set[i][1])

        j += 1

    plt.savefig("Contrast stretching")
    plt.show()

gray = cv2.imread("./s.jpg", 0)

left_intensity = np.copy(gray)
right_intensity = np.copy(gray)
narrow_band_intensity = np.copy(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        #move left
        if(left_intensity[i][j]  - 80 < 0):
            left_intensity[i][j] = 0
        else:
            left_intensity[i][j] = left_intensity[i][j] - 80

        #move right
        if(right_intensity[i][j]  + 80 > 255):
            right_intensity[i][j] = 255
        else:
            right_intensity[i][j] = right_intensity[i][j] + 80

        #narrow band
        if(narrow_band_intensity[i][j] < 100):
            narrow_band_intensity[i][j] = 100
        elif(narrow_band_intensity[i][j] > 150):
            narrow_band_intensity[i][j] = 150

hist1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([right_intensity], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([left_intensity], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([narrow_band_intensity], [0], None, [256], [0, 256])

img_set =   {   "Original image":[gray, hist1],
                "Moved right":[right_intensity, hist2],
                "Moved left":[left_intensity, hist3], 
                "Narrow band":[narrow_band_intensity, hist4] 
            }

plot_image_and_histogram(img_set)