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
   
    
# Function to create the feature vector
def makeFeatureVec(img): 
    
    # Instantiate feature vector
    feature_vector = []
    
    # Convert to all color spaces
    hsv, lab, rgb = convert(img)
    
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
    
    print(feature_vector)
    
    return feature_vector
    
    
    
    
    
    
