import matplotlib.pyplot as plt
import cv2
from numpy import dtype, uint, uint8
import numpy as np

def plot_image(img_set):
    subplot_row = 4
    subplot_column = 4
    j = 1

    plt.figure(figsize = (15, 15))
    
    for i in img_set:
        plt.subplot(subplot_row, subplot_column, j)

        if(img_set[i][1] != ""):
            plt.imshow(img_set[i][0], cmap = img_set[i][1])
        else:
            plt.imshow(img_set[i][0])

        plt.title(i)
        j += 1

    plt.savefig("Morphological analysis")
    plt.show()

def my_erosion(img, kernel):
    m, n = kernel.shape
    row, col = img.shape
    res = np.zeros((row, col))

    for i in range(row-m):
        for j in range(col-n):
            tmp = img[i:i+m, j:j+n] * kernel

            if(tmp == kernel).all():
                res[i, j] = 1
            else:
                res[i, j] = 0

    return res

def my_dilation(img, kernel):
    m, n = kernel.shape
    row, col = img.shape
    tmp = np.zeros((row, col))

    for i in range(row-m):
        for j in range(col-n):
            tmp[i, j] = np.max(img[i:i+m, j:j+n] * kernel)
    return tmp

def my_opening(img, kernel):
    erode = my_erosion(img, kernel)
    res = my_dilation(erode, kernel)
    return res

def my_closing(img, kernel):
    dilate = my_dilation(img, kernel)
    res = my_erosion(dilate, kernel)
    return res

def main():
    img = plt.imread('./s.png')  
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, bin_img = cv2.threshold(gray_img, 0, 1, cv2.THRESH_BINARY)

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(35,35))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(35,35))
    
    dilation_1 = my_dilation(bin_img,kernel_1)
    dilation_2 = my_dilation(bin_img,kernel_2)
    dilation_3 = my_dilation(bin_img,kernel_3)

    erosion_1 = my_erosion(bin_img, kernel_1)
    erosion_2 = my_erosion(bin_img, kernel_2)
    erosion_3 = my_erosion(bin_img, kernel_3)

    opening_1 = my_opening(bin_img, kernel_1)
    opening_2 = my_opening(bin_img, kernel_2)
    opening_3 = my_opening(bin_img, kernel_3)

    closing_1 = my_closing(bin_img, kernel_1)
    closing_2 = my_closing(bin_img, kernel_2)
    closing_3 = my_closing(bin_img, kernel_3)


    img_set =   {
                "Original image":[img, ""],

                "Binary image":[bin_img, "gray"],

                "Dilation 1":[dilation_1, "gray"],
                "Dilation 2":[dilation_2, "gray"],
                "Dilation 3":[dilation_3, "gray"],

                "Erosion 1":[erosion_1, "gray"],
                "Erosion 2":[erosion_2, "gray"],
                "Erosion 3":[erosion_3, "gray"],

                "Opening 1":[opening_1, "gray"],
                "Opening 2":[opening_2, "gray"],
                "Opening 3":[opening_3, "gray"],

                "Closing 1":[closing_1, "gray"],
                "Closing 2":[closing_2, "gray"],
                "Closing 3":[closing_3, "gray"],
                }

    plot_image(img_set)  

if __name__ == '__main__':
    main()
