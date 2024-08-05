import cv2

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

if not cam.isOpened():
    print("Failed to open camera")
    exit()

if model.empty():
    print("Failed to load cascade classifier")
    exit()

while True:
    success, img = cam.read()

    if not success:
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.3, 5)

    print(f"Number of faces detected: {len(faces)}")  # Debugging line

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Image Window", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cam.release()
cv2.destroyAllWindows()
