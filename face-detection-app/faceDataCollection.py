import cv2
import os

# Initialize webcam
cam = cv2.VideoCapture(0)

# Get the name of the person for labeling
fileName = input("Enter the name of the person: ")

# Define dataset path and create directory if it doesn't exist
dataset_path = "face-recognition-dataset"
person_path = os.path.join(dataset_path, fileName)
os.makedirs(person_path, exist_ok=True)

# Load Haar Cascade model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

if not cam.isOpened():
    print("Failed to open camera")
    exit()

if model.empty():
    print("Failed to load cascade classifier")
    exit()

count = 0  # Counter for saved images

while True:
    success, img = cam.read()

    if not success:
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    if len(faces) > 0:
        f = faces[-1]
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y:y + h, x:x + w]

        # Save the captured face image
        face_filename = os.path.join(person_path, f"{fileName}_{count}.jpg")
        cv2.imwrite(face_filename, cropped_face)
        count += 1

        # Display cropped face only if it's defined
        cv2.imshow("cropped face", cropped_face)

    # Display the full image with rectangles
    cv2.imshow("image window", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cam.release()
cv2.destroyAllWindows()
