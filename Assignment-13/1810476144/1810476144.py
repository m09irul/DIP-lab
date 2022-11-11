
from PIL import Image

# To convert the image From JPG to PNG 
img = Image.open("s.jpg")
img.save("s_png.png")

# To convert the Image From PNG to JPG
img = Image.open("s_png.png")
img.save("s_jpg.jpg")
