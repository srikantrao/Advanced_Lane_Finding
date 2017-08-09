import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
import pickle
import helper as hlp
from Line import Line

#Import the module for editing the movie
from moviepy.editor import VideoFileClip

# Get the camera and distortion matrices from the pickle file
matrices = pickle.load( open( "matrices.pickle", "rb" ) )
dist = matrices[0]
mtx = matrices[1]


# Create a pipepline for transforming the image

def image_transform(image):
    """Image Transform performs -
        1. Distortion removal of the image
        2. Color and gradient threshold on the image
        3. Perspective transform on the image """

    # Step 1: Undistort the image
    undist_img = hlp.undistort_image(image, dist, mtx)

    # Step2 : Use a combined threshold function ( color + gradient )
    comb_binary_img = hlp.combined_threshold(undist_img, ksize=7)

    # Step3 : Perform a perspective transform -- Images points are already selected in the src and dst in the function
    warped_img = hlp.warp(comb_binary_img)

    return undist_img,comb_binary_img,warped_img

# Create a function that you can call using the fl_image function to process images of the video

def process_image(image):
    """Process Image performs
        1. Image transformation ( distortion removal, thresholding and perspective transform
        2. Locates lines
        3. Fit lines to a second degree polynomial
        4. Draws lines on warped or unwarped images if needed """

    # Step 1 : Image tranformation
    undist_img, comb_img, warped_img = image_transform(image)

    # Step 2 : Locate the lane lines
    lines.find_lanes(warped_img)

    # Step3 : Fit the lane lines using the left and right lane images
    lines.fit_lines()

    # Step 4 : Draw the lines and unwarp the image
    final_image = lines.draw_lines(image)

    return final_image


#Initialize the line class
lines = Line()
# Create a pipeline and run it on the project video
clip = VideoFileClip("./project_video.mp4")

output_video = "./project_video_output.mp4"

output_clip = clip.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)




# fpath = glob.glob('test_images/test*.jpg')
#
# plt.figure(figsize = (24,24))
# gs1 = gridspec.GridSpec(len(fpath),6)
# gs1.update(wspace=0.0125, hspace=0.0125)
# i=0
# # Test the function
#
# for fname in fpath:
#     fimg = mpimg.imread(fname)
#
#     undist_img,comb_img,warped_img = image_transform(fimg)
#     mpimg.imsave('output_images/undistorted'+fname[-9:],undist_img)
#     mpimg.imsave('output_images/combined_threshold' + fname[-9:], comb_img)
#     mpimg.imsave('output_images/warped' + fname[-9:], warped_img)
#
#     #Initialize the class
#     lines = Line()
#     left_peak,right_peak = lines.find_init_peaks(warped_img)
#     print(left_peak)
#     print(right_peak)
#     lines.find_lanes(warped_img)
#     lines.fit_lines()
#     final_image = lines.draw_lines(fimg)
#     mpimg.imsave('output_images/final_' + fname[-9:], final_image)
#     mpimg.imsave('output_images/left_lane_' + fname[-9:], lines.left_lane_img)
#     mpimg.imsave('output_images/right_lane_' + fname[-9:], lines.right_lane_img)
#     ax1 = plt.subplot(gs1[i])
#     plt.axis('off')
#     ax1.imshow(undist_img,cmap='gray')
#     ax2 = plt.subplot(gs1[i+1])
#     plt.axis('off')
#     ax2.imshow(comb_img,cmap='gray')
#     ax3 = plt.subplot(gs1[i+2])
#     plt.axis('off')
#     ax3.imshow(warped_img,cmap='gray')
#     ax4 = plt.subplot(gs1[i + 3])
#     plt.axis('off')
#     ax4.imshow(lines.left_lane_img,cmap='gray')
#     ax5 = plt.subplot(gs1[i + 4])
#     plt.axis('off')
#     ax5.imshow(lines.right_lane_img, cmap='gray')
#     ax5 = plt.subplot(gs1[i + 5])
#     plt.axis('off')
#     ax5.imshow(final_image)
#     i +=6
#
# plt.show()
