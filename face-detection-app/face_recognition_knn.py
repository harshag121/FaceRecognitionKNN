import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Load the training data
dataset_path = "face-recognition-npy/"
faceData = []
labels = []
nameMap = {}
classId = 0  # Initialize classId

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]  # Map classId to the name of the file (excluding the .npy extension)
        
        dataItem = np.load(os.path.join(dataset_path, f))
        m = dataItem.shape[0]
        faceData.append(dataItem)
        
        target = classId * np.ones((m,), dtype=int)  # Correctly shape the target array
        classId +=  1
        labels.append(target)

# Concatenate face data and labels
X = np.concatenate(faceData, axis=0)
y = np.concatenate(labels, axis=0)

# Reshape y to be a single column
y = y.reshape((-1, 1))

# Flatten each image in X
X_flattened = X.reshape((X.shape[0], -1))

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_flattened, y.ravel())

# Initialize webcam
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
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    if len(faces) > 0:
        f = faces[-1]
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y:y + h, x:x + w]

        # Resize and flatten the cropped face
        cropped_face = cv2.resize(cropped_face, (100, 100))
        cropped_face = cropped_face.flatten().reshape(1, -1)
        
        # Predict the label of the cropped face
        label = knn.predict(cropped_face)
        person_name = nameMap[int(label)]
        
        cv2.putText(img, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cam.release()
cv2.destroyAllWindows()
