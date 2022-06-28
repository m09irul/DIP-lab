from typing import Counter
import matplotlib.pyplot as plt 
import numpy as np
import cv2

a = plt.imread('./clk.jpg')
g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) 

''' Built in function '''

hist_b, bins = np.histogram(g, 256)

''' User defined function '''

hist_u = []
for i in range(256):
    hist_u += [np.count_nonzero(g == i)]

# printing histogram
print('Built in result: ')
#print (hist_b)
print()

print('Built in result: ')
#print (hist_u)
print()

''' plot '''
plt.figure(figsize=(10,10))

plt.subplot(1, 2, 1)
plt.title('Built in function')
plt.plot(hist_b)

plt.subplot(1, 2, 2)
plt.title('User defined function')
plt.plot(hist_u)

plt.show()


