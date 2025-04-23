import cv2
import os

# Rescale down to 500x500 since smallest image is 523
WIDTH, HEIGHT = 500, 500

# Path to original image directory 
original_image_dir = 'C:/Users/almal/Desktop/termin8/TNM098/lab 3/TNM098_Lab3/Lab3.1'

# create a new directory for rescaled images
if not os.path.exists('TNM098_Lab3/rescaled_images'):
    os.makedirs('TNM098_Lab3/rescaled_images')


# Iterate all images in the original image directory 
for filename in os.listdir(original_image_dir):
    # Read image
    image = cv2.imread(os.path.join(original_image_dir, filename))

    # Rescale image
    rescaled = cv2.resize(image, (WIDTH, HEIGHT))

    # Save rescaled image
    cv2.imwrite(os.path.join('TNM098_Lab3/rescaled_images', filename), rescaled)
