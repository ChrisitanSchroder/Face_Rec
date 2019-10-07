
import cv2
import sys
#from Face_reg import frame
cascPath = "cascades\data\haarcascade_frontalface_alt2.xml"
cap = cv2.VideoCapture(1)
faceCascade = cv2.CascadeClassifier(cascPath)

while(True):    

    # Capture frame-by-frame
    ret, frame = cap.read()

    
    # Get user supplied values
    imagePath = "frame"

    
    # Create the haar cascade

    
    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        color = (255, 0, 0)
        stroke = 10
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
     
    #cv2.imshow("Faces found", image)
    #cv2.waitKey(0)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.waitKey(41)
     

cap.release()
cv2.destroyAllWindows()