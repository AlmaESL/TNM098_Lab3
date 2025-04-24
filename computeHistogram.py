import cv2
import numpy as np

def compute_edge_orientation_histogram(img, num_bins=9):
    """
    Computes a histogram of edge orientations using the Sobel operator.

    Parameters:
        img (np.ndarray): Input image (BGR or grayscale).
        num_bins (int): Number of bins for the orientation histogram.

    Returns:
        np.ndarray: 1D histogram with `num_bins` values.
        
        hog_vector = compute_edge_orientation_histogram(img) #TO CALL THE FUNCTION

    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Compute gradients along x and y axes
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude and orientation of gradients
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi)  # in degrees
    orientation = (orientation + 180) % 180  # restrict to [0, 180)

    # Create histogram bins
    bin_edges = np.linspace(0, 180, num_bins + 1)
    hist = np.zeros(num_bins)

    # Fill histogram: weight by gradient magnitude
    for i in range(num_bins):
        mask = (orientation >= bin_edges[i]) & (orientation < bin_edges[i+1])
        hist[i] = np.sum(magnitude[mask])

    # Normalize the histogram
    if np.sum(hist) > 0:
        hist /= np.sum(hist)

    return hist
