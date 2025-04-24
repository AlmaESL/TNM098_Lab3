import cv2
import numpy as np

def convert_image_color_space(image, output_space):
    """
    Converts an image from BGR color space to another using OpenCV.
    """
    output_space = output_space.lower()

    # Define supported conversions in OpenCV
    conversion_code = {
        ('rgb'): cv2.COLOR_BGR2RGB,
        ('hsv'): cv2.COLOR_BGR2HSV,
        ('lab'): cv2.COLOR_BGR2LAB,
    }

    if output_space not in conversion_code:
        raise ValueError("Unsupported conversion")

    return cv2.cvtColor(image, conversion_code[output_space])
