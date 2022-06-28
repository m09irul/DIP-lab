import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (15, 15))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)

        plt.subplot( 3, 2, i + 1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('Output')
    plt.show()

def main():
    img_path = './a.jpg'
    rgb = plt.imread(img_path)
    print(rgb.shape)

    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)

    processed_img1 = np.copy(grayscale)
    processed_img2 = np.copy(grayscale)
    processed_img3 = np.copy(grayscale)
    processed_img4 = np.copy(grayscale)

    t1 = 50
    t2 = 160

    c = 10
    p = 40

    epsilon = 0.000001

    #    1st process
    row, column = grayscale.shape
    for x in range(row):
        for y in range(column):

            r = processed_img1[x, y]

            if r >= t1 and r <= t2:
                processed_img1[x, y] = 100
            else:
                processed_img1[x, y] = 10

    #    2nd process
    row, column = grayscale.shape
    for x in range(row):
        for y in range(column):

            r = processed_img2[x, y]

            if r >= t1 and r <= t2:
                processed_img2[x, y] = 100

    #    3rd process
    row, column = grayscale.shape
    for x in range(row):
        for y in range(column):

            r = processed_img3[x, y]
            s = c * np.log(1 + r)
                
            processed_img3[x, y] = s

    #    4th process
    row, column = grayscale.shape
    for x in range(row):
        for y in range(column):

            r = processed_img4[x, y]
            s = c * pow(( r + epsilon ), p)

            processed_img4[x, y] = s

    img_set = [rgb, grayscale, processed_img1, processed_img2, processed_img3, processed_img4]
    title_set = ['RGB', 'Grayscale', 'processed_img1', 'processed_img2', 'processed_img3', 'processed_img4']
    plot_img(img_set, title_set)

if __name__ == '__main__':
	main()