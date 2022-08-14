import matplotlib.pyplot as plt
import cv2
from numpy import dtype, uint, uint8
import numpy as np

def plot_image(img_set):
    subplot_row = 2
    subplot_column = 3
    j = 1

    plt.figure(figsize = (15, 15))
    
    for i in img_set:
        plt.subplot(subplot_row, subplot_column, j)

        if(img_set[i][1] == "hist"):
            plt.plot(img_set[i][0])
        elif(img_set[i][1] == "gray"):
            plt.imshow(img_set[i][0], cmap = img_set[i][1])
        else:
            plt.imshow(img_set[i][0])

        plt.title(i)
        j += 1

    plt.savefig("Histogram equalization")
    plt.show()



def main():
    img = plt.imread('./s.JPG')  
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1=np.uint8(cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX))

    equ = cv2.equalizeHist(img1)

    hist_before = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_after = cv2.calcHist([equ],[0],None,[256],[0,256])

    img_set =   {
                "Original image":[img, ""],

                "Gray image":[gray_img, "gray"],

                "Equalized image":[equ, "gray"],

                "Histogram before":[hist_before, "hist"],

                "Histogram after":[hist_after, "hist"],
                }

    plot_image(img_set)  

if __name__ == '__main__':
    main()
