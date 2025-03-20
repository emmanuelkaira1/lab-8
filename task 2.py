import cv2
import time
def label_tracking():
    # Loading the label image
    label_img = cv2.imread('label.jpg', cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        print("Error: Unable to load the label image.")
        return

    # Initializing the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capturing frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from the camera.")
            break

        # Converting frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Matching the template (label) in the frame
        res = cv2.matchTemplate(gray_frame, label_img, cv2.TM_CCOEFF_NORMED)

        # Finding the location of the best match
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Drawing a rectangle around the detected area
        top_left = max_loc
        bottom_right = (top_left[0] + label_img.shape[1], top_left[1] + label_img.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Displaying the frame with the detected label
        cv2.imshow('Frame', frame)

        # Exiting if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Releasing the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    label_tracking()