import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = 's.jpg'
    rgb = plt.imread(img_path)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

	# Perform Fast Fourier Transformation for 2D signal, i.e., image
    ftimg = np.fft.fft2(gray)
    centered_ftimg = np.fft.fftshift(ftimg)
    magnitude_spectrum = 100 * np.log(np.abs(ftimg))
    centered_magnitude_spectrum = 100 * np.log(np.abs(centered_ftimg))

	# Build filters.
    ncols, nrows = gray.shape

    global cx, cy 
    cx, cy = (int)(nrows/2),(int) (ncols/2)
    
    filter1 = build_gaussian_filter(ncols, nrows)

    prefilter = np.ones((ncols, nrows), np.uint8)
    filter2 = cv2.rectangle(prefilter,(cx, cy),(cx + 100, cy - 90),(255,255,255),-1)

    prefilter = np.ones((ncols, nrows), np.uint8)
    filter3 = cv2.circle(prefilter,(cx + 90,cy + 90),50,(255,255,255),-1)

    prefilter = np.ones((ncols, nrows), np.uint8)
    filter4 = cv2.rectangle(prefilter,(cx - 50, cy - 150),(cx + 10, cy - 10),(255,255,255),-1)
    filter4 = cv2.circle(filter4,(cx + 150,cy + 150),30,(255,255,255),-1)

	# Apply filters
    ftimg_gf = centered_ftimg * filter1
    filtered_img = np.abs(np.fft.ifft2(ftimg_gf))

    ftimg_gf = centered_ftimg * filter2
    filtered_img2 = np.abs(np.fft.ifft2(ftimg_gf))

    ftimg_gf = centered_ftimg * filter3
    filtered_img3 = np.abs(np.fft.ifft2(ftimg_gf))

    ftimg_gf = centered_ftimg * filter4
    filtered_img4 = np.abs(np.fft.ifft2(ftimg_gf))

	# Save images all together by matplotlib. 
    img_set = [rgb, gray, magnitude_spectrum, centered_magnitude_spectrum, filter1, filter2, filter3, filter4, filtered_img, filtered_img2, filtered_img3, filtered_img4]
    
    title_set = ['RGB', 'Gray', 'FFT2', 'Centered FFT2', 'Gaussian Filter', 'Filter2', 'Filter3', 'Filter4', 'Filtered Img1', 'Filtered Img2', 'Filtered Img3', 'Filtered Img4']
    
    matplotlib_plot_img(img_set, title_set)

def build_gaussian_filter(ncols, nrows):
    sigmax, sigmay = 10, 10
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gaussian_filter = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    
    return gaussian_filter

def matplotlib_plot_img(img_set, title_set):
	plt.figure(figsize = (10, 10))
	n = len(img_set)
	for i in range(n):
		plt.subplot(3, 4, i + 1)
		plt.title(title_set[i])
		img = img_set[i]
		ch = len(img.shape)
		if (ch == 2):
			plt.imshow(img, cmap = 'gray')
		else:
			plt.imshow(img)			

	plt.savefig("Fourier transform")	
	plt.show()

if __name__ == '__main__':
	main()