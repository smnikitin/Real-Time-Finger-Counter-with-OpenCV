import cv2
import numpy as np
from sklearn.metrics import pairwise

# ------------------------ GLOBAL VARIABLES ------------------------

# Background model to be initialized over several frames
background = None

# Weight for running average (used to slowly build background model)
accumulated_weight = 0.5

# Coordinates for the Region of Interest (ROI) where the hand is expected
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


# ------------------------ FUNCTIONS ------------------------

def calc_accum_avg(frame, accumulated_weight):
    """
    Compute a weighted average of the background to create a stable background image over time.
    """
    global background

    # Initialize background if it's the first call
    if background is None:
        background = frame.copy().astype("float")
        return

    # Accumulate the weighted average of the background with the current frame
    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold=15):
    """
    Segments the hand region from the background.
    Returns a tuple of (thresholded image, hand contour).
    """
    global background

    # Calculate the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply binary thresholding to highlight the hand
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Find external contours from the thresholded image
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return None
    if len(contours) == 0:
        return None

    # Otherwise, return the largest contour which likely represents the hand
    hand_segment = max(contours, key=cv2.contourArea)
    return (thresholded, hand_segment)


def count_fingers(thresholded, hand_segment):
    """
    Analyzes the hand segment to count the number of extended fingers.
    """
    # Create convex hull around the hand
    conv_hull = cv2.convexHull(hand_segment)

    # Get extreme points on the convex hull (top, bottom, left, right)
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # Approximate the center of the hand
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # Calculate distances from center to extreme points
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()

    # Define a circular region of interest (ROI) to isolate fingers
    radius = int(0.7 * max_distance)
    circumference = (2 * np.pi * radius)

    # Create a circular mask
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # Apply the mask on the thresholded image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Find contours within the circular ROI
    contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Condition 1: Exclude wrist region (bottom of the hand)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))

        # Condition 2: Ignore large blobs that take up too much of the ROI
        limit_points = ((circumference * 0.25) > cnt.shape[0])

        # If both conditions are met, we count it as a finger
        if out_of_wrist and limit_points:
            count += 1

    return count


# ------------------------ MAIN PROGRAM ------------------------

# Start video capture from webcam
cam = cv2.VideoCapture(0)

# Frame counter
num_frames = 0

while True:
    # Read frame from webcam
    ret, frame = cam.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Extract region of interest (where the hand should be)
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Convert ROI to grayscale and blur it to reduce noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 60 frames, accumulate the background model
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Finger Count", frame_copy)

    else:
        # Once background is built, begin hand segmentation
        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand

            # Convert ROI-based contour coordinates to full frame coordinates
            hand_segment_shifted = hand_segment + [roi_right, roi_top]

            # Draw hand contour
            cv2.drawContours(frame_copy, [hand_segment_shifted], -1, (255, 0, 0), 1)

            # Count the number of fingers
            fingers = count_fingers(thresholded, hand_segment)

            # Display finger count on screen
            cv2.putText(frame_copy, str(fingers), (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show thresholded hand image
            cv2.imshow("Thresholded", thresholded)

    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)

    # Increase frame count
    num_frames += 1

    # Show the final frame with ROI and finger count
    cv2.imshow("Finger Count", frame_copy)

    # Exit on pressing Esc key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
