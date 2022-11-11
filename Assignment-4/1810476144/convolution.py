import numpy as np
#import sys
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt 
import cv2

a = plt.imread('./clk.jpg')
g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) 

g_width = g.shape[0]
g_height = g.shape[1]

''' set kernel and padding image'''

#kernel1 = np.ones((5, 5), dtype = np.float32)/20
kernel1 = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
#kernel1 = np.array([[-1, -1, -1, -1, -1],[-1, -1, 8, -1, -1],[-1, -1, 8, -1, -1],[-1, -1, 8, -1, -1],[-1, -1, -1, -1, -1]])

pad_quantity = int((len(kernel1) - 1) / 2)

padded_g = pad_arr = np.pad(g, pad_quantity)

#print(padded_g)
#padded_g = cv2.copyMakeBorder(src=g, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT) 

''' Built in func '''

processed_img1 = cv2.filter2D(padded_g, -1, kernel1, cv2.BORDER_CONSTANT)
processed_img1 = processed_img1[pad_quantity:g_width+pad_quantity, pad_quantity:g_height+pad_quantity]

print(padded_g[0:5,padded_g.shape[1]-5:]) #cropped padding
print(processed_img1) #cropped padding

''' User defined func '''

m, n = kernel1.shape

y, x = padded_g.shape

processed_img2 = np.zeros((g_width, g_height))

for i in range(g_width):
    for j in range(g_height):
        processed_img2[i][j] = int(np.sum(padded_g[i:i+m, j:j+n]*kernel1))

        if(processed_img2[i][j] > 255):
            processed_img2[i][j] = 255
        if(processed_img2[i][j] < 0):
            processed_img2[i][j] = 0

print(processed_img2)

''' plot '''
plt.figure(figsize=(15,15))

plt.subplot(1, 2, 1)
plt.title('Built in function')
plt.imshow(processed_img1, cmap = 'gray')

plt.subplot(1, 2, 2)
plt.title('User defined function')
plt.imshow(processed_img2, cmap = 'gray')

#plt.show()
plt.savefig("conv out")

