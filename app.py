from __future__ import division, print_function
import os
import cv2
import sqlite3
import random
import smtplib
import pathlib

from email.message import EmailMessage
from ultralytics import YOLO
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= FLASK CONFIG =================
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['CUSTOM_MODEL'] = 'best.pt'      # For image upload detection
app.config['HUMAN_MODEL'] = 'yolov8n.pt'    # For live detection

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ================= LOAD MODELS =================
custom_model = YOLO(app.config['CUSTOM_MODEL'])
human_model = YOLO(app.config['HUMAN_MODEL'])

# =====================================================
#                   BASIC ROUTES
# =====================================================

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('signin.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/notebook')
def notebook():
    return render_template('Notebook.html')

# =====================================================
#                   SIGNUP + OTP
# =====================================================

@app.route('/signup')
def signup():
    global otp, username, name, email, number, password

    username = request.args.get('user', '')
    name = request.args.get('name', '')
    email = request.args.get('email', '')
    number = request.args.get('mobile', '')
    password = request.args.get('password', '')

    otp = random.randint(1000, 5000)

    msg = EmailMessage()
    msg.set_content(f"Your OTP is : {otp}")
    msg['Subject'] = 'OTP'
    msg['From'] = "vandhanatruprojects@gmail.com"
    msg['To'] = email

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("vandhanatruprojects@gmail.com", "pahksvxachlnoopc")
    server.send_message(msg)
    server.quit()

    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    message = request.form['message']

    if int(message) == otp:
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute(
            "INSERT INTO info (user,email,password,mobile,name) VALUES (?,?,?,?,?)",
            (username, email, password, number, name)
        )
        con.commit()
        con.close()
        return render_template("signin.html")

    return render_template("signup.html")

@app.route('/signin')
def signin():
    user = request.args.get('user', '')
    password = request.args.get('password', '')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("SELECT user,password FROM info WHERE user=? AND password=?",
                (user, password))
    data = cur.fetchone()
    con.close()

    if data:
        return render_template("index.html")

    return render_template("signin.html")

# =====================================================
#              IMAGE UPLOAD DETECTION
# =====================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_image(input_path, output_path):
    img = cv2.imread(input_path)
    results = custom_model.predict(source=img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{custom_model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

@app.route('/predict2', methods=['POST'])
def predict2():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

        file.save(input_path)
        process_uploaded_image(input_path, output_path)

        return redirect(url_for('show_output', filename=filename))

    return redirect(url_for('index'))

@app.route('/output/<filename>')
def show_output(filename):
    return render_template('result.html', image_file=f'output/{filename}')

# =====================================================
#                LIVE CAMERA DETECTION
# =====================================================

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = human_model(frame, conf=0.5)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if human_model.names[cls].lower() == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"Person {conf*100:.1f}%"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =====================================================
#                     RUN
# =====================================================

if __name__ == '__main__':
    app.run(debug=False)
