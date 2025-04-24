import cv2
import os 

from makeFeatureVector import makeVector 
from colorConverter import convertToHSV, convertToLAB, convertToYCrCb

# TODO: compute the cosine similarity distance between all vectors 

# TODO: save results in a 12x12 matrix of cosine similarity distances

# TODO: use the matrix to rank the 11 images in similarity to one chosen image

# TODO: visualize the distance results from the ranking eg using a heatmap on pixel-by-pixel basis


# Call images from original image directory 
original_image_dir = 'C:/Users/almal/Desktop/termin8/TNM098/lab 3/TNM098_Lab3/Lab3.1'

# Extract images 