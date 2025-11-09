# Read a video from web cam using openCV
# Face Detection in Video
# Click 10 oictures of the person who comes in the front of camera and save them as numpy

import cv2
import numpy as np

# Camera Object
cam = cv2.VideoCapture(0)

# Ask the name of person
fileName = input("Enter the name of the person :")

dataset_path = "./data/"

offset = 20
# Model
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

## create to list to save face image data

faceData = []
skip = 0

count =0

# Read image from Camera Object

while count < 21:

    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")
    
    # Store the image in grey color format.
    greyImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img,1.3,2)

    # Pick the face with largest bounding box
    faces = sorted(faces, key=lambda f:f[2]*f[3])
    
    # pick the largest face

    if len(faces) >0:

        f = faces[-1]
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

        # crop and save the largest face
        cropped_face = img[y - offset: y+h+offset, x-offset: x+w+offset]

        cropped_face = cv2.resize(cropped_face, (100,100))
        skip += 1

        if skip % 10 == 0:
            faceData.append(cropped_face)
            print("Saved faced images: " + str(len(faceData)))
        
        count +=1
        


   
    cv2.imshow("Image Window", img)
    #cv2.imshow("Cropped Face", cropped_face)

   
    key = cv2.waitKey(1) # Pause here for 1 ms before you read the next image.
    if key == ord('q'):
        break

# write the face data to disk
faceData = np.asarray(faceData)
#print(faceData)
m = faceData.shape[0]
faceData = faceData.reshape((m,-1))
print(faceData.shape)

# Save on disk as np array

filepath =(dataset_path + fileName + ".npy")
np.save(filepath, faceData)
print("Data Saved Successfully" + filepath)

# Release Camera and closed windows
cam.release()
cv2.destroyAllWindows()



