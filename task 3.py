import cv2

def main():
    # Loading the label image
    label_img = cv2.imread('label.jpg', cv2.IMREAD_GRAYSCALE)

    # Initializing the camera
    cap = cv2.VideoCapture(0)

    # Defining the dimensions of the area to hit (center of the image)
    area_width = 200
    area_height = 200
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area_top_left = ((img_width - area_width) // 2, (img_height - area_height) // 2)
    area_bottom_right = (area_top_left[0] + area_width, area_top_left[1] + area_height)

    while True:
        # Capturing frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Converting frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Matching the template (label) in the frame
        res = cv2.matchTemplate(gray_frame, label_img, cv2.TM_CCOEFF_NORMED)

        # Finding the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Drawing a rectangle around the detected area
        top_left = max_loc
        bottom_right = (top_left[0] + label_img.shape[1], top_left[1] + label_img.shape[0])

        # Checking if the label hits the specified area
        if top_left[0] <= area_bottom_right[0] and bottom_right[0] >= area_top_left[0] and \
           top_left[1] <= area_bottom_right[1] and bottom_right[1] >= area_top_left[1]:
            # Label hits the area, draw a green rectangle
            cv2.rectangle(frame, area_top_left, area_bottom_right, (0, 255, 0), 2)
        else:
            # Label does not hit the area, draw a red rectangle
            cv2.rectangle(frame, area_top_left, area_bottom_right, (0, 0, 255), 2)

        # Displaying the frame
        cv2.imshow('Frame', frame)

        # Exiting if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Releasing the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()