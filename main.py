# HSV Paraglider thresholding project
import numpy as np
import cv2        

# ====================================================================================================
# GLOBAL VARIABLES AND HELPER FUNCTIONS
# ====================================================================================================

# Global variable to store the current HSV image frame.
current_hsv = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for the "HSV Channels" window.
    This function is triggered by mouse events (like movement) over the pop-up window.
    Its purpose is to display the Hue, Saturation, and Value (HSV) of the pixel
    currently under the mouse cursor in the window's title bar. This used for
    interactively determining and fine-tuning the HSV color ranges for segmentation.

    Parameters:
    - event: Type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN).
    - x, y: Coordinates of the mouse pointer on the window.
    - flags: Any flags passed with the event.
    - param: Optional parameters passed to the callback function.
    """
    # Access the global HSV image variable.
    global current_hsv 
    # Ensure current_hsv is not None and coordinates are within image bounds to prevent errors.
    if current_hsv is not None and x < current_hsv.shape[1] and y < current_hsv.shape[0]:
        h, s, v = current_hsv[y, x] # Get HSV values for the pixel at (x, y).
        # Update the title of the "HSV Channels" window to show the current HSV values.
        # This provides real-time feedback for color tuning.
        cv2.setWindowTitle("HSV Channels", f"HSV - H:{h} S:{s} V:{v} at ({x},{y})")

# ====================================================================================================
# MAIN PROGRAM FUNCTION
# ====================================================================================================

# Declare current_hsv as global to modify it within main.
def main():
    global current_hsv

    # --- Video Capture Setup ---
    # Defines the absolute path to the video file.
    # Use the directory where your videos/images are stored
    video_path = "/home/falcon/Ascend-Machine-Vision/IMG1.MOV"
    # Example if using another video from the repository:
    # video_path = "/home/masslab/Desktop/MachineVision/Paraglider_Video/ParagliderVideo/IMG6.MOV"

    # Create a VideoCapture object to read frames from the video.
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully.
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return # Exit the function if video cannot be opened.

    # Get the total number of frames in the video for informational purposes.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # --- GUI Window Setup ---
    # Set up mouse callback for HSV window
    # CHANGE FROM ORIGINAL: Window name for HSV display changed to "HSV Channels" for clarity.
    cv2.namedWindow("HSV Channels")
    # Attach the mouse callback function to the "HSV Channels" window.
    # This enables interactive HSV value sampling.
    cv2.setMouseCallback("HSV Channels", mouse_callback)

     # --- Morphological Operations Kernel ---
    # NEW ADDITION: Defines a kernel (structuring element) for morphological operations.
    # This is used to clean up masks by filling small holes and removing noise,
    # making the segmentation less grainy and more robust.
    # 'kernel_size' is a tunable parameter to adjust the strength of the cleaning.
    # Experimenting with values like 3, 5, 7, 9.
    # A larger kernel has a stronger effect (fills larger holes, removes larger specks).
    kernel_size = 7 # Recommended start for balancing detail and smoothing.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # --- Control Flags ---
    running = True  # Flag to control the main video processing loop.
    paused = False  # Flag to pause/resume video playback.
    print("Controls: Press SPACEBAR to pause/resume, 'q' to quit")

    # ================================================================================================
    # MAIN VIDEO PROCESSING LOOP
    # ================================================================================================
    while running:
        # Only read a new frame if the video is not paused.
        if not paused:
            success, frame = cap.read() # Read a frame from the video.

            # If frame reading fails, reset to the beginning.
            if not success:
                print("End of video reached, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Set video position back to the first frame.
                continue # Skip the rest of the current loop iteration and read the first frame.

            # Resize the frame for consistent processing and display performance.
            frame = cv2.resize(frame, (640, 480))
            # Convert the frame from BGR (OpenCV's default) to HSV color space.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Update the global current_hsv variable for the mouse callback.
            current_hsv = hsv

            # TRACKBAR ARRAY FOR MASKING, [Hue,Staturation,Value] ranges from 0-255
            # CHANGE FROM ORIGINAL: HSV ranges are updated to better target the specific
            # red, orange, and yellow colors of the paraglider based on video analysis.
            # These are starting points, but will be fine-tuned using the mouse callback.

            # Red (Body/Middle) - Often appears as a magenta-red in HSV.
            lower_red = np.array([160, 100, 50])
            upper_red = np.array([179, 255, 255])

            # Orange (Back) - Hues typically between red and yellow.
            lower_orange = np.array([5, 100, 50])
            upper_orange = np.array([20, 255, 255])

            # Yellow (Front) - Bright yellow hues.
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])

            # --- Create Binary Masks ---
            # Create Masks (using the tuned HSV ranges)
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

            # --- Apply Morphological Operations for Mask Cleaning ---
            # This major improvement results in a smoother image and complete, accurate masks.
            #
            # NEW ADDITION: Morphological Closing.
            # Closing (Dilation followed by Erosion) is applied to each individual mask.
            # It helps to fill small black holes within the white (object) regions
            # and to connect small, nearby white regions.
            mask_red_cleaned = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
            mask_yellow_cleaned = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
            mask_orange_cleaned = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

            # Optional Tool: Morphological Opening.
            # This operation can be uncommented to remove small, isolated noise specks
            # (white dots) that might appear outside the main paraglider regions.
            # mask_red_cleaned = cv2.morphologyEx(mask_red_cleaned, cv2.MORPH_OPEN, kernel)
            # mask_yellow_cleaned = cv2.morphologyEx(mask_yellow_cleaned, cv2.MORPH_OPEN, kernel)
            # mask_orange_cleaned = cv2.morphologyEx(mask_orange_cleaned, cv2.MORPH_OPEN, kernel)

            # Combine all cleaned masks into one comprehensive mask.
            # CHANGE FROM ORIGINAL: Now combines the *cleaned* masks.
            mask_combined = mask_red_cleaned + mask_yellow_cleaned + mask_orange_cleaned

            # --- Find Contours on Cleaned Masks ---
            # Finds the outlines of the detected paraglider sections.
            # CHANGE FROM ORIGINAL: Contours are now found on the *cleaned* masks,
            # resulting in smoother and more accurate outlines.
            contours_red, _ = cv2.findContours(mask_red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_orange, _ = cv2.findContours(mask_orange_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # --- Contour Filtering by Area ---
            # NEW ADDITION: Filters out small contours that are likely noise.
            # Since paraglider sections are large, contours below a certain area
            # can be safely discarded, leading to cleaner contour visualizations.
            # 'min_contour_area' is a tunable threshold.
            min_contour_area = 1000 # Experiment with this value.

            filtered_contours_red = [c for c in contours_red if cv2.contourArea(c) > min_contour_area]
            filtered_contours_yellow = [c for c in contours_yellow if cv2.contourArea(c) > min_contour_area]
            filtered_contours_orange = [c for c in contours_orange if cv2.contourArea(c) > min_contour_area]

            # --- Prepare Images for Display ---
            # NEW ADDITION: Creates a masked version of the original frame.
            # This shows only the detected paraglider colors on the original video background.
            result_masked_original = cv2.bitwise_and(frame, frame, mask = mask_combined)

            contour_thickness = 2 # Thickness of drawn contours.
            # CHANGE FROM ORIGINAL: Contour thickness increased for visibility.

            # Create a copy of the original frame to draw contours on.
            # CHANGE FROM ORIGINAL: Drawing on a copy preserves the original frame.
            # Contours are now drawn in colors corresponding to the paraglider sections.
            frame_with_contours = frame.copy()
            for contour in filtered_contours_red:
                cv2.drawContours(frame_with_contours, [contour], -1, (0, 0, 255), contour_thickness) # Red for Red section
            for contour in filtered_contours_yellow:
                cv2.drawContours(frame_with_contours, [contour], -1, (0, 255, 255), contour_thickness) # Yellow for Yellow section
            for contour in filtered_contours_orange:
                cv2.drawContours(frame_with_contours, [contour], -1, (0, 165, 255), contour_thickness) # Orange for Orange section

        # Display windows (always show, even when paused)
        if 'frame' in locals():
            # CHANGE FROM ORIGINAL: More descriptive window names for clarity.
            cv2.imshow("Original Frame", frame) # Display the raw video frame.
            cv2.imshow("HSV Channels", hsv) # Display frame in HSV color space (for tuning).
            # NEW ADDITION: Display the cleaned, combined mask.
            cv2.imshow("Cleaned Combined Mask", mask_combined)
            # NEW ADDITION: Display the original frame with only the masked areas.
            cv2.imshow("Result (Masked Original)", result_masked_original)
            # NEW ADDITION: Display the original frame with colored, filtered contours.
            cv2.imshow("Frame with Contours", frame_with_contours)

            # Optional Tool: Uncomment these to debug individual cleaned masks if needed
            # cv2.imshow("Red Mask Cleaned", mask_red_cleaned)
            # cv2.imshow("Yellow Mask Cleaned", mask_yellow_cleaned)
            # cv2.imshow("Orange Mask Cleaned", mask_orange_cleaned)

        # Handles key presses
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'): # Break frame loop 'Q'
            break
        elif key == ord(' '): # Spacebar
            paused = not paused
            if paused:
                print("Video PAUSED - Press SPACEBAR to resume")
            else:
                print("Video RESUMED")

    # ================================================================================================
    # CLEANUP
    # ================================================================================================
    cap.release()          # Release the video capture object.
    cv2.destroyAllWindows() # Close all OpenCV display windows.

# Entry point of the script. Ensures main() is called only when the script is executed directly.
if __name__ == "__main__":
    main()
