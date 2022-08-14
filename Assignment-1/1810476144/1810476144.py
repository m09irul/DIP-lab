import matplotlib.pyplot as plt
import cv2

def main():
  img_path = "./portulica.jpg"
  img = plt.imread(img_path)

  # Grayscale image
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Creating Histogram
  gray_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])


  plt.figure(figsize=(15,15))
  
  # Plotting Histogram
  plt.subplot(2, 3, 1)
  plt.title("Grayscale")
  plt.plot(gray_hist)

 
  #extract red channel
  red_channel = img[:,:,0]

  # Histogram
  red_hist = cv2.calcHist([red_channel], [0], None, [256], [0, 256])

  # Plotting Histogram
  plt.subplot(2, 3, 2)
  plt.title("Red Channel")
  plt.plot(red_hist)


  #extract green channel
  green_channel = img[:,:,1]

  # Creating Histogram
  green_hist = cv2.calcHist([green_channel], [0], None, [256], [0, 256])

  # Plotting Histogram
  plt.subplot(2, 3, 3)
  plt.title("Green Channel")
  plt.plot(green_hist)


  #extract blue channel
  blue_channel = img[:,:,2]

  # Creating Histogram
  blue_hist = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

  # Plotting Histogram
  plt.subplot(2, 3, 4)
  plt.title("Blue Channel")
  plt.plot(blue_hist)

  # Binary Image
  b_tuple = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
  binary = b_tuple[1]

  # Creating Histogram
  binary_hist = cv2.calcHist([binary], [0], None, [256], [0, 256])

  # Plotting Histogram
  plt.subplot(2, 3, 5)
  plt.title("Binary Image")
  plt.plot(binary_hist)
  
  # set the spacing between subplots
  #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

  plt.savefig("All Histograms")

  plt.show()
 # plt.clf()

  plt.figure(figsize=(15,15))

  
# section for images


  plt.subplot(2, 3, 1)
  plt.title("RGB")
  plt.imshow(img)

  plt.subplot(2, 3, 2)
  plt.title("Grayscale")
  plt.imshow(gray_img, cmap = 'gray')

  plt.subplot(2, 3, 3)
  plt.title("Red Channel")
  plt.imshow(red_channel, cmap = 'gray')

  plt.subplot(2, 3, 4)
  plt.title("Green Channel")
  plt.imshow(green_channel, cmap = 'gray')

  plt.subplot(2, 3, 5)
  plt.title("Blue Channel")
  plt.imshow(blue_channel, cmap = 'gray')

  plt.subplot(2, 3, 6)
  plt.title("Binary image")
  plt.imshow(binary, cmap = 'gray')

  
  plt.savefig("All Images")
  plt.show()

if __name__ == '__main__':
	main()