from typing import Tuple

import numpy as np
import cv2

import rospy

# current coeffs
# right_coeff 0.00025
# left_coeff -0.00038

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Yellow line matrix - simple bottom half, left half
    """
    height, width = shape
    matrix = np.zeros((height, width))
    
    # Bottom half, left half
    h_start = height // 2
    w_mid = width // 2
    
    matrix[h_start:, :w_mid] = rospy.get_param("/steer_matrix_left_coeff")
    
    return matrix

def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    White line matrix - simple bottom half, right half
    """
    height, width = shape
    matrix = np.zeros((height, width))
    
    # Bottom half, right half
    h_start = height // 2
    w_mid = width // 2
    
    matrix[h_start:, w_mid:] = rospy.get_param("/steer_matrix_right_coeff")
    
    return matrix

import numpy as np
import cv2
from typing import Tuple

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects lane markings in an image using color and edge-based detection.
    
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape
    
    # Convert to HSV color space for better color segmentation
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define color thresholds
    # Even more tolerant of dark grays and various lighting conditions
        # Extremely wide range for white/gray detection
    white_lower_hsv = np.array([0, 0, 107])     # Much lower value threshold, even wider hue range
    white_upper_hsv = np.array([180, 50, 255])  # Higher saturation threshold and maximum hue range
    yellow_lower_hsv = np.array([20, 100, 100])    # Lower hue threshold, lower saturation and value thresholds
    yellow_upper_hsv = np.array([40, 255, 255]) 
    
    # Create color masks
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    # Apply Gaussian blur to reduce noise
    img_gaussian = cv2.GaussianBlur(img_gray, (0, 0), 3)
    
    # Calculate gradients using Sobel
    sobelx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1)
    
    # Calculate gradient magnitude
    Gmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Create gradient magnitude mask (threshold the magnitude)
    thresh_mag = 65
    mask_mag = Gmag > thresh_mag
    
    # Create left and right region masks
    mask_left = np.ones(sobelx.shape)
    mask_left[:, int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:, 0:int(np.floor(w/2))] = 0
    
    # Create derivative sign masks
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_neg = (sobely < 0)
        
    # Combine masks for left edge (yellow line)
    mask_left_edge = (mask_left * 
                     mask_mag * 
                     mask_sobelx_neg * 
                     mask_sobely_neg * 
                     mask_yellow)
    
    # Combine masks for right edge (white line)
    mask_right_edge = (mask_right * 
                      mask_mag * 
                      mask_sobelx_pos * 
                      mask_sobely_neg * 
                      mask_white)
    
    # Convert to uint8
    mask_left_edge = mask_left_edge.astype(np.uint8)
    mask_right_edge = mask_right_edge.astype(np.uint8)
    
    return mask_left_edge, mask_right_edge