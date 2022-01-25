from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():

    first_frame = None
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        
        if not success:
            break
        else:     
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray_image = cv2.GaussianBlur(gray,(21,21),0)

            if first_frame is None:
                first_frame=gray_image
            try:
                faces = face_cascade.detectMultiScale(gray_image,scaleFactor=1.05,minNeighbors=5)
                eyes = eye_cascade.detectMultiScale(gray_image,scaleFactor=1.05,minNeighbors=5)
                for x,y,w,h in faces:
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                for x,y,w,h in eyes:
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            finally:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)