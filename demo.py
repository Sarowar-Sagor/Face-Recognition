import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import face_recognition
import imutils
import pickle
import time
import cv2
from imutils import paths

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = 'static/image dataset/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/create',methods=['POST','GET'])
def add_image():
	if request.method=='POST':
		if request.files:
			file=request.files['file']
			if file.filename == '':
				flash('No image selected for uploading')
				return redirect(request.url)
			if not allowed_file(file.filename):
				flash('this extension is not allowed')
				return redirect(request.url)
			else:
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				flash('Image successfully uploaded')
			return redirect(request.url)
	return render_template('upload.html')	

@app.route('/update')
def update_dataset():
	imagePaths = list(paths.list_images('static/image dataset'))
	knownEncodings = []
	knownNames = []
	for (i, imagePath) in enumerate(imagePaths):
	    name = imagePath.split(os.path.sep)[-2]
	    image = cv2.imread(imagePath)
	    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    boxes = face_recognition.face_locations(rgb,model='hog')
	    encodings = face_recognition.face_encodings(rgb, boxes)
	    for encoding in encodings:
	        knownEncodings.append(encoding)
	        knownNames.append(name)
	data = {"encodings": knownEncodings, "names": knownNames}
	f = open("embedded_dataset", "wb")
	f.write(pickle.dumps(data))
	f.close()
	return render_template('update.html')

def checkimage(filename):
	cascPathface=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt2.xml"
	faceCascade = cv2.CascadeClassifier(cascPathface)
	data = pickle.loads(open('embedded_dataset', "rb").read())
	image = cv2.imread(filename)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5, minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
	 
	encodings = face_recognition.face_encodings(rgb)
	names = []
	for encoding in encodings:
	    matches = face_recognition.compare_faces(data["encodings"],
	    encoding)
	    name = "Unknown"
	    if True in matches:
	        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
	        counts = {}
	        for i in matchedIdxs:
	            name = data["names"][i]
	            counts[name] = counts.get(name, 0) + 1
	            name = max(counts, key=counts.get)
	        names.append(name)
	        for ((x, y, w, h), name) in zip(faces, names):
	            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	            cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

	    image=cv2.resize(image,(600,600))
	    cv2.imshow("Frame", image)
	    cv2.waitKey(0)


@app.route('/detectimage',methods=['POST','GET'])
def detectimage():
	if request.method=='POST':
		if request.files:
			file=request.files['file']
			if file.filename == '':
				flash('No image selected for uploading')
				return redirect(request.url)
			if not allowed_file(file.filename):
				flash('this extension is not allowed')
				return redirect(request.url)
			else:
				filename = secure_filename(file.filename)
				checkimage(filename)
				# #print('upload_image filename: ' + filename)
				flash('Image successfully checked')
			return redirect(request.url)
	return render_template('fromimage.html')	

@app.route('/detectwebcam',methods=['POST','GET'])
def detectwebcam():
	cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
	faceCascade = cv2.CascadeClassifier(cascPathface)
	data = pickle.loads(open('embedded_dataset', "rb").read())
	video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	while True:
	    ret,frame = video_capture.read()
	    if ret==True:
	        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
	        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	        encodings = face_recognition.face_encodings(rgb)
	        names = []
	        for encoding in encodings:
	            matches = face_recognition.compare_faces(data["encodings"],encoding)
	            name = "Unknown"
	            if True in matches:
	                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
	                counts = {}
	                for i in matchedIdxs:
	                    name = data["names"][i]
	                    counts[name] = counts.get(name, 0) + 1
	                name = max(counts, key=counts.get)
	            names.append(name)
	            for ((x, y, w, h), name) in zip(faces, names):
	                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
	        cv2.imshow("Frame", frame)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break
	video_capture.release()
	cv2.destroyAllWindows()
	flash('Streaming finished')
	return render_template('fromwebcam.html')


VIDEO_EXTENSIONS = set(['mp4', 'mkv', 'mpeg', '3gp'])
def allowed_videofile(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in VIDEO_EXTENSIONS


def checkvideo(filename):
	cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
	faceCascade = cv2.CascadeClassifier(cascPathface)
	data = pickle.loads(open('embedded_dataset', "rb").read())
	video_capture = cv2.VideoCapture(filename)
	while True:
	    ret,frame = video_capture.read()
	    if ret==True:
	        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
	        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	        encodings = face_recognition.face_encodings(rgb)
	        names = []
	        for encoding in encodings:
	            matches = face_recognition.compare_faces(data["encodings"],encoding)
	            name = "Unknown"
	            if True in matches:
	                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
	                counts = {}
	                for i in matchedIdxs:
	                    name = data["names"][i]
	                    counts[name] = counts.get(name, 0) + 1
	                name = max(counts, key=counts.get)
	     
	            names.append(name)
	            for ((x, y, w, h), name) in zip(faces, names):
	                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
	        image=cv2.resize(frame,(650,700))
	        cv2.imshow("press q to exit",image)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break
	video_capture.release()
	cv2.destroyAllWindows()

@app.route('/detectvideo',methods=['POST','GET'])
def detectvideo():
	if request.method=='POST':
		if request.files:
			file=request.files['file']
			if file.filename == '':
				flash('No file selected for uploading')
				return redirect(request.url)
			if not allowed_videofile(file.filename):
				flash('this extension is not allowed')
				return redirect(request.url)
			else:
				filename = secure_filename(file.filename)
				checkvideo(filename)
				# #print('upload_image filename: ' + filename)
				flash('video successfully checked')
				#return render_template('upload.html', filename=filename)
			return redirect(request.url)
	return render_template('fromvideo.html')	