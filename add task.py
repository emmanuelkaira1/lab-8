import cv2

def label_tracking():
    label_img = cv2.imread('label.jpg', cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        print("Error: Unable to load the label image.")
        return

    fly_img = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    if fly_img is None:
        print("Error: Unable to load the fly image.")
        return

    cap = cv2.VideoCapture(0)

    while True:
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

        # Calculating the center of the label
        label_center_x = max_loc[0] + label_img.shape[1] // 2
        label_center_y = max_loc[1] + label_img.shape[0] // 2

        # Calculating the position of the fly image to align its center with the center of the label
        fly_top_left = (label_center_x - fly_img.shape[1] // 2, label_center_y - fly_img.shape[0] // 2)
        fly_bottom_right = (fly_top_left[0] + fly_img.shape[1], fly_top_left[1] + fly_img.shape[0])

        # Checking if the fly image is completely within the frame
        if (0 <= fly_top_left[0] < frame.shape[1] and 0 <= fly_top_left[1] < frame.shape[0] and
            0 <= fly_bottom_right[0] < frame.shape[1] and 0 <= fly_bottom_right[1] < frame.shape[0]):

            # Overlaying the fly image on the frame
            overlay_image(frame, fly_img, fly_top_left, fly_bottom_right)

        # Draw a rectangle around the detected area
        top_left = max_loc
        bottom_right = (top_left[0] + label_img.shape[1], top_left[1] + label_img.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Displaying the frame with the detected label and overlayed fly
        cv2.imshow('Frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def overlay_image(background, overlay, top_left, bottom_right):
    # Extracting the alpha channel from the overlay image
    alpha = overlay[:, :, 3] / 255.0

    # Resizing the overlay image to match the size of the region of interest
    overlay_resized = cv2.resize(overlay[:, :, :3], (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

    # Computing the region of interest on the background image
    roi = background[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Overlaying the image of the fly on the frame
    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_resized[:, :, c]

if __name__ == '_main_':
    label_tracking()