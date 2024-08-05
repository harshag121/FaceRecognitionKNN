import cv2
import numpy as np
import os

dataset_path = "face-recognition-npy/"
faceData = []
labels = []
nameMap = {}
classId = 0  # Initialize classId

# Iterate through the dataset directory
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
y = np.concatenate(labels, axis=0).reshape((-1, 1))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print(nameMap)