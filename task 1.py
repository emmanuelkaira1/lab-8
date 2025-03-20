import cv2
def display_and_save_hsv_image(image_path):

    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    screen_width, screen_height = 1366, 768
    hsv_img_resized = cv2.resize(hsv_img, (screen_width, screen_height))

    cv2.imshow('HSV Image', hsv_img_resized)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()

    cv2.imwrite('variant-3_hsv_resized.jpg', hsv_img_resized)

if __name__ == "__main__":
    image_path = 'variant-3.jpg'
    display_and_save_hsv_image(image_path)