import cv2
import os


from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

face_cascade = cv2.cascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")

y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswidth("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            pil_image = Image.open(path).covert("L")#grayscale
            image_array = np.array(pil_image, "uin8")
            faces = face_cascade.detectMultiScale(image_array, scalefactor=1.5, minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
 