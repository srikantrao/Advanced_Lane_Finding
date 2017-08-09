import numpy as np
import cv2
import helper as hlp
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # Were the lanes detected in the last frame
        self.left_lane_found = False
        self.right_lane_found = False
        # x values of the last 10 fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last 10 iterations
        self.best_fit = None
        #radius of curvature of the line in some units
        self.left_roc = None
        self.right_roc = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #Keep track of which frame this is
        self.frame_number = 0
        #Lane locations from previous frames
        self.last_left_peak = 0
        self.last_right_peak = 0
        # Current lane locations
        self.left_peak = 0
        self.right_peak = 0
        #Number of frames before the right and left lanes were last seen
        self.left_last = 1000
        self.right_last = 1000
        #Lane fit coordinates
        self.left_fitx = 0
        self.left_fity = 0
        self.right_fitx = 0
        self.right_fity = 0
        # Lane polynomial fits
        self.left_fit = []
        self.right_fit = []

        #Keep track of left and right lanes in each image
        self.left_lane_img = 0
        self.right_lane_img = 0

    def find_init_peaks(self,img,left_offset=200):
        """ Function to find the initial peaks starting from the bottom of an image with no prior fitting knowledge
            returns the x values of the left and right peaks"""
        histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
        # Find the left and right peaks in the x direction
        num_pixels = len(histogram)
        right_offset = int(num_pixels / 2)
        #peak finding is done ignoring the offset for left and right peaks
        left_peak = np.argmax(histogram[left_offset:int(num_pixels / 2)]) + left_offset
        right_peak = np.argmax(histogram[int(num_pixels / 2):-left_offset]) + right_offset
        return left_peak, right_peak

    def find_cont_peaks(self,hist,prev_left_peak,prev_right_peak, start_y=0,end_y=0,found_last_left=False,
                        found_last_right=False,left_trend=0,right_trend=0):
        num_pixels = len(hist)
        window = 40
        found_left = True
        found_right = True

        # make the window size bigger if you are starting from scratch
        if (not found_last_left) or (not found_last_right):
            window = 60
        if start_y == 720:
            window = 100  # If starting at the bottom of the image, need to increase the size of the window

        left_offset = 0
        #Pipeline for the left lane
        # See if lane was found in the last frame then use the best fit
        if self.left_lane_found:
            new_left_peak = int(self.left_fit[0]*(start_y**2) + self.left_fit[1]*start_y + self.left_fit[2])
        #Find peak using sliding window if previous peak does not exist
        else:
            left_window = hist[prev_left_peak + left_trend - window: prev_left_peak + left_trend + window]
            new_left_peak = 0
            if len(left_window) > 0:
                new_left_peak = np.argmax(left_window)
                left_offset = prev_left_peak + left_trend - window
            if new_left_peak == 0 or len(left_window) == 0:
                new_left_peak = prev_left_peak + left_trend
                left_offset = 0
                found_left = False
        #Set the left peak to the xmax value + left offset
        left_peak = new_left_peak + left_offset

        # Pipeline for the right lane
        # See if lane was found in the last frame then use the best fit
        right_offset = 0
        if self.right_lane_found:
            new_right_peak = int(self.right_fit[0] * (start_y ** 2) + self.right_fit[1] * start_y + self.right_fit[2])
        # Find peak using sliding window if previous peak does not exist
        else:
            right_window = hist[prev_right_peak + right_trend - window: prev_right_peak + right_trend + window]
            new_right_peak = 0
            if len(right_window) > 0:
                new_right_peak = np.argmax(right_window)
                right_offset = prev_right_peak + right_trend - window
            if new_right_peak == 0 or len(right_window) == 0:
                new_right_peak = prev_right_peak + right_trend
                right_offset = 0
                found_right = False
        # Set the right peak to the xmax value + left offset
        right_peak = new_right_peak + right_offset

        #Set the left_trend and right_trend  -- these are similar to slope values
        if start_y == 720:
            new_left_trend = 0
            new_right_trend = 0
        else:
            new_left_trend = left_peak - prev_left_peak
            new_right_trend = right_peak - prev_right_peak

        return left_peak,right_peak,found_left, found_right,new_left_trend,new_right_trend


    def find_lanes(self,image):
        """Function finds the left and right lane using fitting or detection from scratch """
        # If the lanes have not been found for a while now
        if self.frame_number ==0 or self.left_last > 3 or self.right_last > 3:
            #Use method to find the left and righ peaks from scratch
            est_left_peak,est_right_peak = self.find_init_peaks(image)
        # If lanes have been found and fit in the previous image
        else:
            est_left_peak = self.last_left_peak
            est_right_peak = self.last_right_peak

        fallback_left_peak = est_left_peak
        fallback_right_peak = est_right_peak

        # Create blank images the size of the original images for the left and right peaks
        self.left_lane_img = np.zeros(image.shape,np.float32)
        self.right_lane_img = np.zeros(image.shape,np.float32)

        #Initialize different variables for new image
        left_trend = 0
        right_trend = 0
        found_last_left = False
        found_last_right = False
        window = 40  # Size of the sliding window
        # This keeps track of the number of continuous sections
        left_conc_sections = 0
        right_conc_sections = 0

        #Divide the image up into 10 parts along the rows and use the window to find pixels belonging to the lane
        for i in range(10,0,-1):
            start_y = int(i * image.shape[0]/10)
            end_y = int((i-1) * image.shape[0]/10)
            histogram = np.sum(image[end_y:start_y,:],axis=0)
            left_peak,right_peak,found_last_left,found_last_right,new_left_trend,new_right_trend = self.find_cont_peaks(histogram,est_left_peak,est_right_peak,start_y,end_y,found_last_left,
                            found_last_right,left_trend,right_trend)

            left_trend = int((new_left_trend + left_trend) / 2)
            right_trend = int((new_right_trend + right_trend) / 2)
            # Store the left/right x values for bottom of image
            if i == 10:
                # Store the new peak values for the bottom of the image
                self.left_peak = left_peak
                self.right_peak = right_peak

            # if the last search did not yield a left peak
            if not found_last_left:
                left_peak = fallback_left_peak
                left_conc_sections = 0
            elif left_conc_sections > 1:
                fallback_left_peak = left_peak  # Fall back peak is the new peak that was found
            if not found_last_right:
                right_peak = fallback_right_peak
                right_conc_sections = 0
            elif right_conc_sections > 1:
                fallback_right_peak = right_peak

            # Check if the peaks were found for the last search
            if found_last_left:
                left_conc_sections +=1
            if found_last_right:
                right_conc_sections +=1

            # Fill in the left lane mask
            img_sect = image[end_y:start_y,:]
            left_mask = np.zeros_like(img_sect)
            left_mask[:,left_peak - window : left_peak + window] = 1 # Sliding window mask
            mask = (left_mask == 1)
            self.left_lane_img[end_y:start_y,:] = img_sect * left_mask

            # Fill in the right lane mask
            right_mask = np.zeros_like(img_sect)
            right_mask[:,right_peak-window: right_peak + window] = 1
            mask = (right_mask ==1)

            self.right_lane_img[end_y:start_y,:] = img_sect * right_mask

            est_left_peak = left_peak
            est_right_peak = right_peak

        self.frame_number +=1

    def sanity_check_roc(self,new_roc,prev_roc,new_peak,prev_peak):
        if new_roc < 580:
             return False
        #Make sure the delta is not too big
        delta = 150
        if new_roc > prev_roc + delta or new_roc < prev_roc - delta:
            return False

        max_roc = prev_roc * 100.0
        min_roc = prev_roc / 100.0

        if new_roc >= min_roc and new_roc <= max_roc:
            return True
        else:
            return False

    # Fit lines on left_lane_img and right_lane_img using the below function
    def fit_line(self,image):
        """ Function that fits a line on an image which contains only the lines after masking.
            This function will be typically used after the lane lines have been located """
        [y_coord,x_coord] = np.nonzero(image)
        #Fit a second degree polynomial
        fit = np.polyfit(y_coord, x_coord, deg=2)  # Peak fitting is for the x co-ordinate
        fitx = fit[0] * y_coord ** 2 + fit[1] * y_coord + fit[2]

        # Make sure the line extends throughout the image
        fitx = np.insert(fitx, 0, fit[2])
        fity = np.insert(y_coord, 0, 0)
        fitx = np.append(fitx, fit[0] * 719 ** 2 + fit[1] * 719 + fit[2])
        fity = np.append(fity, 719)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(y_coord)

        fit_cr = np.polyfit(y_coord * ym_per_pix, x_coord * xm_per_pix, 2)
        #fitx_cr = fit_cr[0] * (y_coord * ym_per_pix) ** 2 + fit_cr[1] * y_coord * ym_per_pix + fit_cr[2]

        # Get radius of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5)/ np.absolute(2 * fit_cr[0])

        return fit, fitx, fity, curverad

    def fit_lines(self):
        """Fit the left and right lane lines """

        # Get new lane fit for left lane
        left_fit, left_fitx, left_fity, left_curverad = self.fit_line(self.left_lane_img)
        if self.frame_number ==1 or self.sanity_check_roc(left_curverad,self.left_curverad,self.left_peak,self.last_left_peak):
            self.left_fit = left_fit
            self.left_fitx = left_fitx
            self.left_fity = left_fity
            self.left_curverad = left_curverad
            self.left_lane_found = True
            self.left_last = 0
            self.last_left_peak = self.left_peak
        else:
            self.left_last +=1

        # Get new lane fit for right lane
        right_fit, right_fitx, right_fity, right_curverad = self.fit_line(self.right_lane_img)
        if self.frame_number ==1 or self.sanity_check_roc(left_curverad,self.left_curverad,self.left_peak,self.last_left_peak):
            self.right_fit = right_fit
            self.right_fitx = right_fitx
            self.right_fity = right_fity
            self.right_curverad = right_curverad
            self.right_lane_found = True
            self.right_last = 0
            self.last_right_peak = self.right_peak
        else:
            self.right_last +=1

    def draw_lines(self,image):
        """Function to draw the fitted left and right lanes using the cv2.fillPoly function """
        #Blank image on which to draw the lines on
        warp = np.zeros_like(image).astype(dtype=np.uint8)

        left_lane = [list(x) for x in zip(self.left_fitx,self.left_fity)]
        right_lane = [list(x) for x in zip(self.right_fitx,self.right_fity)]
        right_lane.reverse()
        left_lane.extend(right_lane)
        pts = np.array([left_lane])

        #Draw the line onto the warped image
        cv2.fillPoly(warp,np.int_([pts]),(0,255,0))

        # Draw lane lines only if lane was detected this frame
        if self.left_lane_found == True:
            cv2.polylines(warp, np.int_([left_lane]), False, (0, 0, 255), thickness=20)
        if self.right_lane_found == True:
            cv2.polylines(warp, np.int_([right_lane]), False, (255, 0, 0), thickness=20)

        unwarped = hlp.unwarp(warp)

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, unwarped, 0.3, 0)

        # Write the ROC
        roc_text = "Radius of Curvature: " + str(int((self.left_curverad + self.right_curverad)/2)) + "m"
        cv2.putText(result,roc_text,org = (500,50),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,
                    color=(255,0,0),thickness=2)
        # Write the distance from Center
        # Write dist from center
        perfect_center = 1280 / 2.
        lane_x = self.last_right_peak - self.left_peak
        center_x = (lane_x / 2.0) + self.left_peak
        cms_per_pixel = 370.0 / lane_x  # US regulation lane width = 3.7m
        dist_from_center = (center_x - perfect_center) * cms_per_pixel
        dist_text = "Dist from Center: {0:.2f} cms".format(dist_from_center)
        cv2.putText(result, dist_text, (500, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        return result








