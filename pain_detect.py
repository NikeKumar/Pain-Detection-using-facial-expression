
import cv2
import numpy as np
import tensorflow as tf


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model('pain_model.h5',compile = False)

cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:

    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            img_new=cv2.resize(roi_color,(200,200))
            test = np.array(img_new).reshape(-1, 200, 200, 3) 

            prediction = model.predict(test)
            max_index = np.argmax(prediction[0])
            label = ('Pain','No pain')
            result = label[max_index]
            print(result)
            cv2.putText(img, result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Display an image in a window
    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
