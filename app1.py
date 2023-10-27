import cv2
import os
from flask import Flask, request, render_template, flash, session    
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import requests

# Defining Flask App
app = Flask(__name__)

app.secret_key = 'Csk@123'  # Replace 'your_secret_key_here' with a strong and unique secret key
# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Create the attendance CSV file with headers if it doesn't exist
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if attendance_file not in os.listdir('Attendance'):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# Function to get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract the face from an image
def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except:
        return []

# Function to identify a face using the ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Function to train the model on all the faces available in the 'faces' folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Function to extract info from today's attendance file in the 'Attendance' folder
def extract_attendance():
    df = pd.read_csv(attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Function to add attendance of a specific user
def add_attendance(name, newuserphone):  # Pass newuserphone as an argument
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(attendance_file)
    if int(userid) not in list(df['Roll']):
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        
        # Send an SMS
        message_text = f"Attendance taken for {username} ({userid}) at {current_time}"
        phone_number = newuserphone  # Use the newuserphone value
        
        sms_result = send_sms(message_text, phone_number)
        
        # Check if the SMS was sent successfully
        if "error" in sms_result:
            flash(f"Attendance taken successfully, but SMS sending failed: {sms_result['error']}", "warning")
        else:
            flash("Attendance taken successfully and SMS sent!", "success")

# Function to send an SMS
def send_sms(message, phone):
    # Replace with your actual API secret and device ID
    apisecret = "23c5d1b1e743d1ca84d43eb92af681dff5e2dcf3"
    deviceId = "00000000-0000-0000-bd8f-1d0f920d4268"
    
    # Create the message payload
    payload = {
        "secret": apisecret,
        "mode": "devices",
        "device": deviceId,
        "sim": 1,
        "priority": 1,
        "phone": phone,
        "message": message,
    }
    
    # Send the SMS
    response = requests.post(url="https://www.cloud.smschef.com/api/send/sms", params=payload)
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        # Handle the error, e.g., by logging it or returning an error message
        return {"error": "Failed to send SMS"}

# Function to get all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    
    return userlist, names, rolls, l

# Function to delete a user's folder
def deletefolder(duser):
    pics = os.listdir(duser)
    
    for i in pics:
        os.remove(duser + '/' + i)

    os.rmdir(duser)

# Routing Functions
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        # Flash an error message
        flash("There is no trained model in the static folder. Please add a new face to continue.", "danger")
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2) 

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person, 'newuserphone')  # Replace 'newuserphone' with the actual phone number
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()    

    # Check if "success" flash message is in the session and render the success.html template accordingly
    if "success" in session:
        return render_template('success.html')  # Render the success template
    else:
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        newuserphone = request.form['newuserphone']  # Retrieve the phone number
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 10 == 0:
                    name = newusername + '_' + str(i) + '.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == 500:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()
        
        # Pass newuserphone to add_attendance function
        add_attendance(f"{newusername}_{newuserid}", newuserphone)
        
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return render_template('add.html')  # Render the form to capture new user data

if __name__ == '__main__':
    app.run(debug=True)
