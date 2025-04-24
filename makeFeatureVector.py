from colorConverter import convert_image_color_space
import statistics as stats
import cv2

# TODO: decide what features to use

# TODO: extract those features, eg with OpenCV or pillow

# TODO: return the feature vector

# Function to convert to all color spaces
def convert(img): 
    # Convert to HSV
    hsv = convert_image_color_space(img, 'HSV')
    
    # Convert to LAB
    lab = convert_image_color_space(img, 'LAB')
    
    # Convert to YCrCb
    rgb = convert_image_color_space(img, 'RGB')
    
    return hsv, lab, rgb
    

# Help function to split channels 
def split(img): 
    c1, c2, c3 = cv2.split(img)
    return c1, c2, c3


# Function to compute the harmonic mean of each image 
def harmonic_mean(img_channel): 
    return stats.harmonic_mean(img_channel)


# Function to find center 9x9 patch of an image
def extract_center_patch(img):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2

    # Calculate the coordinates for the 9x9 patch
    start_x = max(center_x - 4, 0)
    end_x = min(center_x + 5, width)
    start_y = max(center_y - 4, 0)
    end_y = min(center_y + 5, height)

    # Extract the patch
    patch = img[start_y:end_y, start_x:end_x]

    return patch
   
    
# Function to create the feature vector
def makeFeatureVec(img): 
    
    # Instantiate feature vector
    feature_vector = []
    
    # Convert to all color spaces
    hsv, lab, rgb = convert(img)
    
    # Get the 9x9 center patch 
    lab_center = extract_center_patch(lab)
    rgb_center = extract_center_patch(rgb)
    
    # Split channels
    h, s, v = split(hsv)
    l, a, b = split(lab)
    r, g, bb = split(rgb)
    
    # Compute the harmonic mean of each channel
    h_mean = harmonic_mean(h)
    feature_vector.append(h_mean)
    s_mean = harmonic_mean(s)
    feature_vector.append(s_mean)
    v_mean = harmonic_mean(v)  
    feature_vector.append(v_mean)  
    l_mean = harmonic_mean(l) 
    feature_vector.append(l_mean)        
    a_mean = harmonic_mean(a)  
    feature_vector.append(a_mean)       
    b_mean = harmonic_mean(b)
    feature_vector.append(b_mean) 
    r_mean = harmonic_mean(r)
    feature_vector.append(r_mean) 
    g_mean = harmonic_mean(g)
    feature_vector.append(g_mean) 
    bb_mean = harmonic_mean(bb)
    feature_vector.append(bb_mean) 
    
    # Split center patches channels 
    l_center, a_center, b_center = split(lab_center)
    r_center, g_center, bb_center = split(rgb_center)
    
    # Compute average L in center patch of LAB 
    l_center_mean = stats.harmonic_mean(l_center)
    feature_vector.append(l_center_mean)
    
    # Compute harmonic mean of rgb center patch 
    r_center_mean = stats.harmonic_mean(r_center)
    feature_vector.append(r_center_mean)
    g_center_mean = stats.harmonic_mean(g_center)
    feature_vector.append(g_center_mean)
    bb_center_mean = stats.harmonic_mean(bb_center)
    feature_vector.append(bb_center_mean)
    
    print(feature_vector)
    
    return feature_vector
    
    
    
    
    
    
