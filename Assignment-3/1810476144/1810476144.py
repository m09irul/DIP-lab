import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	path = './clock.jpg'
	
	rgb = plt.imread(path)
	print(rgb.shape)

	gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	print(gray.shape)

	kernel_1 = np.ones((5, 5), np.float32)/30
	kernel_2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	kernel_3 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
	kernel_4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/5
	kernel_5 = np.ones((3, 3), dtype = np.float32) / 9
	kernel_6 = np.array([[0, 5, 0], [1, -2, 1], [-1, 1, -1]])
	
	print('Kernel 1: ' + format(kernel_1))
	print('Kernel 2: ' + format(kernel_2))
	print('Kernel 3: ' + format(kernel_3))
	print('Kernel 4: ' + format(kernel_4))
	print('Kernel 5: ' + format(kernel_5))
	print('Kernel 6: ' + format(kernel_6))
	
	processed_img1 = cv2.filter2D(gray, -1, kernel_1)
	processed_img2 = cv2.filter2D(gray, -1, kernel_2)
	processed_img3 = cv2.filter2D(gray, -1, kernel_3)
	processed_img4 = cv2.filter2D(gray, -1, kernel_4)
	processed_img5 = cv2.filter2D(gray, -1, kernel_5)
	processed_img6 = cv2.filter2D(gray, -1, kernel_6)
		
    #plot
	
	img_set = [rgb, gray, processed_img1, processed_img2, processed_img3, processed_img4, processed_img5, processed_img6]
	title_set = ['RGB', 'Grayscale', 'Kernel1', 'Kernel2', 'Kernel3', 'Kernel4', 'Kernel5', 'Kernel6']
	PlotImg(img_set, title_set)


def PlotImg(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize=(10, 10))
    for i in range(n):
        img = img_set[i]
        channel = len(img.shape)

        plt.subplot(2, 4, i+1)

        if(channel == 3):
            plt.imshow(img)
        else:
            plt.imshow(img, cmap = 'gray')
		
        plt.title(title_set[i])
    plt.show()

if __name__ == '__main__':
    main()