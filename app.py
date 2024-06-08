from flask import Flask, render_template, Response, flash, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib
import glob

if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if f'Attendance_sheet.xlsx' not in os.listdir('Attendance'):
    columns = ['No.', 'Surname', 'Name', 'M/F', 'Standard']
    # Create a DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    # Save the DataFrame to an Excel file
    df.to_excel('Attendance/Attendance_sheet.xlsx', index=False)

attendance_sheet = r'Attendance/Attendance_sheet.xlsx'

url = 'http://192.168.1.101/cam-hi.jpg'  
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
nimgs = 10
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



@app.route('/find_student', methods =["GET", "POST"])
def find_student():
    if request.method == "POST":
        first_name = request.form.get("fname").title().strip()
        last_name = request.form.get("lname").title().strip()
        df = pd.read_excel(attendance_sheet)
        for index, row in df.iloc[:].iterrows():
            if (row['Name'] == first_name) and (row['Surname'] == last_name):
                return redirect(url_for('update_standard', index=index, first_name=first_name, last_name=last_name))
                
        
        else:
            print("student not found")
            flash('student not found. Please add the student.', 'danger')
    return render_template("find_student.html")


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []
    
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

def update_photo(first_name, surname):
        print("function updatephoto called", first_name, surname)
        cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
        name_drive = 'static/faces/' + first_name + " " + surname
        if not os.path.isdir(name_drive):
            os.makedirs(name_drive)
        else:
            files = glob.glob(name_drive + '/*')
            for f in files:
                os.remove(f)
        
        i, j = 0, 0
        nimgs = 10  # Define the number of images to capture
        
        while True:
            # Capture frame from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Extract faces from the frame
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = first_name + "_" + surname + '_' + str(i) + '.jpg'
                    cv2.imwrite(name_drive + '/' + name, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:  # Press 'ESC' to break
                break
        cv2.destroyAllWindows()
        train_model()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def add_data(first_name, last_name, student, gender):
    df = pd.read_excel(attendance_sheet)
    for index, row in df.iloc[:].iterrows():
        if (row['Name'] == first_name) and (row['Surname'] == last_name):
            flash("Student already in the list", "info")
            break
    else:
        name_drive = 'static/faces/' + first_name + " " + last_name
        if not os.path.isdir(name_drive):
            os.makedirs(name_drive)
        i, j = 0, 0
        while True:
            # Capture frame from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Extract faces from the frame
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = first_name + "_" + last_name + '_' + str(i) + '.jpg'
                    cv2.imwrite(name_drive + '/' + name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                max_no = df['No.'].max() if not df.empty else 0
                new_student_data = {'No.': max_no + 1, 'Surname': last_name, 'Name': first_name, 'M/F': gender, 'Student': student}
                df = df._append(new_student_data, ignore_index=True)
                df.to_excel(attendance_sheet, index=False)
                train_model()
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:  # Press 'ESC' to break
                break
        cv2.destroyAllWindows()
    return

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/update_standard/<int:index>/<first_name>/<last_name>', methods =["GET", "POST"])
def update_standard(index, first_name, last_name):
    print("Standard function>>>>", index, first_name, last_name)
    if request.method == "POST":
        standard = request.form.get("standard")
        print("Standard:", standard)
        df = pd.read_excel(attendance_sheet)
        df.at[index, 'Standard'] = standard
        df.to_excel(attendance_sheet, index=False)
        update_photo(first_name, last_name)
    return render_template('update_standard.html', index=index, first_name=first_name, last_name=last_name)

@app.route('/add_student', methods =["GET", "POST"])
def add_student():
    if request.method == "POST":
        first_name = request.form.get("fname").title().strip()
        last_name = request.form.get("lname").title().strip()
        standard = request.form.get("standard").title().strip()
        gender = request.form.get("gender").title().strip()
        add_data(first_name, last_name, standard, gender)
        return render_template("add_student.html")
    return render_template("add_student.html")


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def add_attendance(name):
        df = pd.read_excel(attendance_sheet)
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d 00:00:00')
        if date_string not in df.columns:
            df[date_string] = ""

        # Find the student in the dataframe and mark the attendance
        for index, row in df.iloc[:].iterrows():
            student_name = f"{row['Name']} {row['Surname']}"
            if student_name.upper() == name.upper():
                df.at[index, date_string] = "Present"
                print(f"Attendance marked for {name} on {date_string}")
                break
        else:
            print(f"Student '{name}' not found in the attendance sheet.")
        df.to_excel(attendance_sheet, index=False)

@app.route('/mark_attendance')
def mark_attendance():

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print('There is no trained model in the static folder. Please add a new face to continue.')

    else:
        while True:
            # Continuously capture frames from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Resize and convert the frame to RGB
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
            facesCurFrame = extract_faces(imgS)
        
            for (x, y, w, h) in facesCurFrame:
                face = imgS[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                face_flatten = face.flatten().reshape(1, -1)
                identified_person = identify_face(face_flatten)[0]
                add_attendance(identified_person)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        cv2.imread
    return render_template("mark_attendance.html")

if __name__=="__main__":
    app.run(debug=True)