import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Failed to open camera")
    exit()

while True:
    success, img = cam.read()
    if not success:
        print("Failed to read frame")
        break

    cv2.imshow("Image Window", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cam.release()
cv2.destroyAllWindows()
