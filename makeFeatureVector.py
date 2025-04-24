from colorConverter import conver_image_color_space
import statistics as stats
import cv2

# TODO: decide what features to use

# TODO: extract those features, eg with OpenCV or pillow

# TODO: return the feature vector

# Function to convert to all color spaces
def convert(img): 
    # Convert to HSV
    hsv = conver_image_color_space(img, 'HSV')
    
    # Convert to LAB
    lab = conver_image_color_space(img, 'LAB')
    
    # Convert to YCrCb
    rgb = conver_image_color_space(img, 'RGB')
    
    return hsv, lab, rgb
    

# Help function to split channels 
def split(img): 
    c1, c2, c3 = cv2.split(img)
    return c1, c2, c3


# Function to compute the harmonic mean of each image 
