# Import statements

import numpy as np
import cv2


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ # Define a function that applies Sobel x and y, then computes the direction of the gradient
    and applies a threshold.
    To streamline code - it takes as input a grayscale Image """
    # Step1: Convert the image to grayscale
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #Step 2: Take the Sobel filter in the x and y direction
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    #Step 3: Calculate the inverse tan value
    theta = np.arctan2(sobely,sobelx)

    #Step4: Create a binary mask using the threshold values
    dir_output = np.zeros_like(theta)
    dir_output[(theta >= thresh[0]) & (theta <= thresh[1])] = 1

    return dir_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    """Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values
    To streamline code - it takes as input a grayscale Image """
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def sobel_threshold(gray,sobel_kernel, orient='x',thresh_min=0,thresh_max=255):
    """ Uses the Sobel filter of a given size to calculate gradient in either x or y direction
        To streamline code - it takes as input a grayscale Image """
    #Step1: Convert the image to grayscale
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  # mpimpg.imread() to read  then use RGB2GRAY, cv2.imread() - BGR2GRAY
    if orient == 'y':
        x_dir = 0
        y_dir = 1
    else:
        x_dir = 1
        y_dir = 0

    #Step2: Calculate the absolute value of the sobel operator in the required direction
    sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F, x_dir, y_dir, ksize = sobel_kernel))

    #Step3: Convert it to 8 bit Integer values -- Scale to between 0 and 255
    sobel = np.uint8((sobel*255)/np.max(sobel))

    #Step4: Create a binary image using the upper and lower threshold
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel >= thresh_min) & (sobel <=thresh_max)]=1

    return binary_output

def hls_select(img,thresh=(0,255)):
    """Convert an image to HLS color space and apply thresholds on the S space"""
    #Step1: Convert to HLS space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    #Step2: Apply a threshold
    s_binary = np.zeros_like(S)
    s_thresh = cv2.inRange(S.astype('uint8'), 175, 250)
    s_binary[(s_thresh == 255)] = 1

    return s_binary

def hsv_yellow(img,lower_thresh=[20, 100, 100],upper_thresh =[40, 255, 255]):
    hsv_low = np.array(lower_thresh)
    hsv_high = np.array(upper_thresh)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def hsv_white(img,lower_thresh=[0, 0, 225],upper_thresh =[255, 25, 255]):
    hsv_low = np.array(lower_thresh)
    hsv_high = np.array(upper_thresh)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def sobel_combined(image):
    """ Combine different thresholds masks to return a binary image """
    img_g_mag = mag_thresh(image,7,(20,150))
    img_d_mag = dir_threshold(image,7,(.6,1.1))
    img_abs_x = sobel_threshold(image,'x',7,(50,200))
    img_abs_y = sobel_threshold(image,'y',7,(50,200))
    sobel_combined = np.zeros_like(img_d_mag)
    sobel_combined[((img_abs_x == 1) & (img_abs_y == 1)) | \
               ((img_g_mag == 1) & (img_d_mag == 1))] = 1
    return sobel_combined

def warp(img):
    img_size = (img.shape[1], img.shape[0])


    # Four Source Co-ordinates
    src = np.float32(
                [[240, 720],
                [579, 450],
                [712, 450],
                [1165, 720]])

    # Initial four destination points

    dest = np.float32(
        [[300,720],
        [300,0],
        [900,0],
        [900,720]])

    # Compute the perspective transforms

    M = cv2.getPerspectiveTransform(src, dest)

    # Create warped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    img_size = (img.shape[1], img.shape[0])


    # Four Source Co-ordinates
    src = np.float32(
                [[240, 720],
                [579, 450],
                [712, 450],
                [1165, 720]])

    # Initial four destination points

    dest = np.float32(
        [[300,720],
        [300,0],
        [900,0],
        [900,720]])

    # Get inverse of the perspective transform
    Minv = cv2.getPerspectiveTransform(dest, src)

    # Create warped image
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return warped

def undistort_image(image,dist,mtx):
    """Undistort an input image after """
    undist_image = cv2.undistort(image,mtx,dist,mtx)
    return undist_image

def combined_threshold(image,ksize):

    yellow = hsv_yellow(image)
    white_img = hsv_white(image)
    combined = cv2.bitwise_or(src1=yellow,src2=white_img)

    gray = cv2.cvtColor(combined, cv2.COLOR_RGB2GRAY)
    gradx = sobel_threshold(gray, orient='x', thresh_min=30, thresh_max=255, sobel_kernel=ksize)

    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 255))

    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.5))

    comb_binary = np.zeros_like(dir_binary)

    comb_binary[ ((gradx == 1) & (dir_binary == 1) & (mag_binary == 1) )] = 1

    return comb_binary
