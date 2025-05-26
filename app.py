# Importing essential libraries and modules
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from flask import Flask, render_template, request, redirect, Response
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------


# Loading the model
global face_cascade,pain_model,cap,x_value,x1,y1
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

pain_model = tf.keras.models.load_model('models/pain_model.h5',compile = False)
x_value=0
x1=[]
y1=[]


cap = cv2.VideoCapture(0)
# =========================================================================================

# Custom functions for calculations


def video_stream(model=pain_model):
    global x_value,x1,y1
# loop runs if capturing has been initialized.
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            faces=face_cascade.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                img_new=cv2.resize(roi_color,(200,200))
                test = np.array(img_new).reshape(-1, 200, 200, 3) 

                prediction = model.predict(test)
                p=prediction[0]
                pred = p[0]
                pred = pred*10
                y_value = int(pred)
                x_value += 1
                x1.append(x_value)
                y1.append(y_value)
                max_index = np.argmax(prediction[0])
                label = ('Pain','No pain')
                result = label[max_index]
                cv2.putText(frame, result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                dict = {'x_axis':x1,'y_axis':y1}
                df = pd.DataFrame(dict)
                df.to_csv('data_file.csv')    
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def graph_show():
    plt.style.use('fivethirtyeight')


    def animate(i):
        data = pd.read_csv('data_file.csv')
        x_label = data['x_axis']
        y_label = data['y_axis']


        plt.cla()

        plt.plot(x_label, y_label, label='Pain Intensity')
       

        plt.legend(loc='upper left')
        plt.tight_layout()


    ani = FuncAnimation(plt.gcf(), animate, interval=1000)

    plt.tight_layout()
    plt.show()

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Facial Painometer - Home'
    return render_template('index.html', title=title)




@ app.route('/Real-Time')
def Real_Time():
    title = 'Facial Painometer - RealTime'
    return render_template('livevideo.html', title=title)




@ app.route('/Upload')
def Upload():
    title = 'Facial Painometer - Upload'

    return render_template('image-upload.html', title=title)




@ app.route('/about')
def about():
    title = 'Facial Painometer - About-us'

    return render_template('about.html', title=title)



# ===============================================================================================

# RENDER PREDICTION PAGES



@ app.route('/video-feed', methods=['GET'])
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')





@ app.route('/image-prediction', methods=['GET','POST'])
def image_prediction():
    title = 'Facial Painometer - Image-Prediction'

    
    
    try:
        file = request.files.get('image')
        img =  cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        #Draw the rectangle around each face
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            img_new=cv2.resize(roi_color,(200,200))
            test = np.array(img_new).reshape(-1, 200, 200, 3) 

            prediction = pain_model.predict(test)
            max_index = np.argmax(prediction[0])
            label = ('Pain','No pain')
            result = label[max_index]
            return render_template('image-result.html', prediction=result,title=title)
    except:
        pass
  
@app.route('/requests',methods=['POST'])
def video_operation():
    global cap
    if  request.form.get('stop') == 'Stop':
            cap.release()
            return render_template('livevideo.html')
            #cv2.destroyAllWindows()
            
    else:
        cap = cv2.VideoCapture(0)
        return render_template('livevideo.html')


@app.route('/graph')
def graph():
    return Response(graph_show(), mimetype='multipart/x-mixed-replace; boundary=frame')



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
