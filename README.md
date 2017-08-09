Project Report for the Advanced Lane Finding Assignment 

Author - Srikant Rao 


The following details are provided based on the Project Specification 

1. Computing the camera calibration matrix and distortion coefficients given a set of chessboard images - 

The camera coefficients and the distortion matrix have been calculated and saved in a pickle file called matrices.pickle.
The undistorted chessboard images have been plotted in the cam_calib jupyter notebook for demonstration.

2. Apply the distortion correction to the raw image  

The distortion matrix extracted using the chessboard images of the camera have been used to undistort the test images provided. These images have been plotted in the finding_lanes jupyter notebook. Addtional details of the implementation are provided in the notebook. 

3. Use color transforms, gradients, etc., to create a thresholded binary image - 

The combined threshold function used uses the following combination.
	3.1. Gradient Threshold - It combines Sobel filter of size 7 with Gradient Magnitude and gradient direction thresholding. 
	3.2. Color Threshold - It uses the HSV color space to mask yellow and white regions (lanes) in the images. 	
A bitwise logical AND is performed on these two thresholded values to determine the regions of the lanes. 
The test image results are plotted provided results in grayscale. 
The functions which implement are in the helper.py module and are namely - 
hsv_yellow(), hsv_white(), sobel_combined(), combined_threshold()

4. Apply a perspective transform to rectify binary image ("birds-eye view") - 

A perspective tranform has been applied based on a trapezoid region in the test images where the lanes were seemingly straight (with very very large Radius of Curvature) and this was then transformed to a rectangle in the warped image. 
The function image_transform() is used to peform the following step by step - 
   Step 1. Undistort the image using the camera matrix and distortion coefficients
   Step 2. Use a combined gradient and color threshold function to determine the lane lines in the image
   Step 3. Warp the binary image provided by Step 2 using a perspective transform

5. Detect lane pixels and fit to find lane boundary - 

Detection of lane lines has been accomplished in helper functions in the class Line based on the skeleton provided by the lecture notes. 

The function find_lanes() takes as input the warped image and detects the pixels which are part of the left and right lanes and creates two new images for the left and right lane each. 
	The first step towards finding the pixels is finding the peaks of the column wise histograms.

	High Level Psuedo Code for the function - 

	5.1  If the lanes were found in the last image then use the fit parameters to find the peak for this image
	5.2  if the lanes were not found, use find_init_peak() to find the left and right peaks at the base on the image
	5.3. Then use a sliding window to determine peaks in smaller regions, keep track of the slope for the left and right peak      positions. Find all pixels =1 in this sliding window region and use them as part of the lane.
	5.4. Draw these pixels on the left and right lane images respectively. 

	Once the left and right lane line images have been created, the numpy PolyFit is used to determine a second degree polynomial fittin for the lane line. This is accomplished using fit_lines(), fit_line() functions which in turn makes use of numpy.polyfit for the fitting.

	Once this is done, the lane lines are then drawn back on the image - This is done using the image draw_lines() which in turn uses cv2.PolyFill for performing this function. 


6. Determine curvature of the lane and vehicle position with respect to center.

The Radius of Curvature and the vehicle position with respect to center are calculated based on the code provided in the lecture notes.
The formula for ROC base on the fit parameters provided is used to determine independent radius of curvature for left and right values. 

7. Warp the detected lane boundaries back onto the original image.

The draw_lines() draws the fitted polygon on the warped image and then the inverse Matrix of the perspective transform is used to "un-warp the image." This image is now superimposed using the cv2.addWeighted function on to the original undistorted image and provides a marking for the left and right lane boundaries. 

8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

This is implemented in the draw_lines function in the line class. You can see it being displayed in the project video output as well. 

The project video output has been saved to the file project_video_output.mp4. 


Addtional information pertaining to the grading rubric 

Does the pipeline established with the test images work to process the video? 

The pipeline that has been established has been documented in the cam_calib and finding_lanes jupyter notebooks. It plots the images at different stages in the pipeline. 

The plots of different stages in the pipeline of the test images have been recorded in the output_images folder as well. The image pipeline.png shows the different stages right from undistorted images to the final image with the lane lines drawn on it. 

Has some kind of search method been implemented to discover the position of the lines in the first images in the video stream?

The initial search uses the find_peaks function based on the column wise histogram that has been provided in the lecture notes. A similar kind of strategy is used in the function find_init_peaks(). This function finds the peaks in the histogram based and determines the starting values of the lane lines. 

Has some form of tracking of the position of the lane lines been implemented?

If peaks and lane lines were found ( after fulfiling the sanity checks) in the previous image, the existing fit parameters are used to determine the lane line position for the bottom of the new image. This helps save precious processing power when the lane line has been well fit and does not need recomputation of peaks from scratch. 
This also helps with smoothening of the lane lines in the actual video as it continous to use the same fitting parameters until it fails to locate the lane lines with those parameters. 

Improvements that I will make -

1.One thing I am very interested in doing is enabling a automatic perspective transform. Currently I am providing the src and dest points for the perspective transform based on my rough calculations for one of the test images where the lanes appear straight. 

2. The harder challenge videos lead to a few images where no peaks are detected. My current debugging pipeline has no intermediate results for the harder challenge video ( no test images provided for this !!) so I am going to spend some time trying to figure out how to get this to work. 

