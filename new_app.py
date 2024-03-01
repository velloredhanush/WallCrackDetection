from flask import Flask, request, render_template, send_from_directory,flash
import os
import numpy as np
import pandas as pd
import cv2
import mysql.connector
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the random string'

classes = ['System Detected Image as Normal', 'System Detected Image as Cracked']

# Function to calculate average width of crack branches
def calculate_average_width(crack_branches):
    widths = []
    for branch in crack_branches:
        if isinstance(branch, np.ndarray):
            branch = branch.tolist()
        elif not isinstance(branch, list):
            continue
        branch_array = np.array(branch)
        if len(branch_array) < 2:
            continue
        try:
            x, y, w, h = cv2.boundingRect(branch_array)
            width = w
            widths.append(width)
        except Exception as e:
            print(f"Error processing contour: {e}")
    if widths:
        average_width = np.mean(widths)
    else:
        average_width = 0
    return average_width

# Function to calculate centroid of a contour
def calculate_centroid(contour):
    if isinstance(contour, list):
        contour_np = np.array(contour)
    elif isinstance(contour, np.ndarray):
        contour_np = contour
    else:
        return 0, 0  # Return default values if contour format is not recognized

    if len(contour_np) == 0:
        return 0, 0  # Return default values if contour is empty

    try:
        M = cv2.moments(contour_np)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
    except Exception as e:
        print(f"Error calculating centroid: {e}")
        cX, cY = 0, 0

    return cX, cY




# Function to calculate average distance between crack branches
def calculate_average_distance(crack_branches):
    distances = []
    num_branches = len(crack_branches)
    for i in range(num_branches - 1):
        centroid1 = calculate_centroid(crack_branches[i])
        centroid2 = calculate_centroid(crack_branches[i + 1])
        distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
        distances.append(distance)
    average_distance = np.mean(distances)
    return average_distance

# Function to assess severity level based on width and distance
def assess_severity_level(average_width, average_distance):
    threshold1 = 0.01
    threshold2 = 0.02
    threshold3 = 0.03
    threshold4 = 0.04
    if average_width > threshold1 and average_distance > threshold2:
        severity = "High"
    elif average_width > threshold3 or average_distance > threshold4:
        severity = "Medium"
    else:
        severity = "Low"
    return severity

# Function to classify crack branches into types
def classify_crack_branches(crack_branches):
    threshold1 = 0.01
    threshold2 = 0.02
    threshold3 = 0.03
    threshold4 = 0.04
    classified_branches = {'transversal': [], 'longitudinal': [], 'block': [], 'alligator': []}
    for branch in crack_branches:
        area = cv2.contourArea(branch)
        if area < threshold1:
            classified_branches['transversal'].append(branch)
        elif area < threshold2:
            classified_branches['longitudinal'].append(branch)
        elif area < threshold3:
            classified_branches['block'].append(branch)
        else:
            classified_branches['alligator'].append(branch)
    return classified_branches


# Function to perform crack segmentation
def perform_crack_segmentation(img):
    img_array = image.img_to_array(img)

    # Convert to BGR color space (assuming image is in RGB format)
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    blurred_image = np.uint8(blurred_image)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    classified_branches = classify_crack_branches(contours)
    formatted_contours = []
    for contour in contours:
        if isinstance(contour, list):
            contour = np.array(contour)
        if contour.size > 0:
            formatted_contours.append(contour)
    return classified_branches, formatted_contours

# Function to assess severity based on classification result
def assess_severity(classification_result, classified_branches):
    if classification_result == 'The model predicts this image has a crack.':
        crack_branches = classified_branches['alligator']  # Assuming you are interested in a specific type of crack
        average_width = calculate_average_width(crack_branches)
        average_distance = calculate_average_distance(crack_branches)
        severity_level = assess_severity_level(average_width, average_distance)
        return severity_level
    else:
        return "Not Cracked"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/reg")
def reg():
    return render_template("ureg.html")
@app.route('/regback',methods = ["POST"])
def regback():
    if request.method=='POST':
        name=request.form['name']
        email=request.form['email']
        pwd=request.form['pwd']
        cpwd=request.form['cpwd']
        pno=request.form['pno']



    #email = request.form["email"]

        print("**************")
        mydb = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            passwd="",
            database="pavement"
        )
        mycursor = mydb.cursor()
        print("**************")
        sql = "select * from ureg"
        result = pd.read_sql_query(sql, mydb)
        email1 = result['email'].values
        print(email1)
        if email in email1:
            flash("email already exists","warning")
            return render_template('ureg.html')
        if(pwd==cpwd):
            sql = "INSERT INTO ureg (name,email,pwd,pno) VALUES(%s,%s,%s,%s)"
            val = (name, email, pwd, pno)
            mycursor.execute(sql, val)
            mydb.commit()
            flash("You registered successfully", "success")

            return render_template('user.html')
        else:
            flash("Password and Confirm Password are not same", "danger")
            return render_template('ureg.html')
    flash("Something wrong", "danger")
    return render_template('user.html', msg="registered successfully")


@app.route('/userlog',methods=['POST', 'GET'])
def userlog():
    global name, name1
    global user
    if request.method == "POST":

        username = request.form['email']
        password1 = request.form['pwd']
        print('p')
        mydb = mysql.connector.connect(host="localhost",port=3306, user="root", passwd="", database="pavement")
        cursor = mydb.cursor()
        sql = "select * from ureg where email='%s' and pwd='%s'" % (username, password1)
        print('q')
        x = cursor.execute(sql)
        print(x)
        results = cursor.fetchall()
        print(results)
        if len(results) > 0:
            print('r')
            # session['user'] = username
            # session['id'] = results[0][0]
            # print(id)
            # print(session['id'])
            flash("Welcome to website", "success")
            return render_template('userhome.html', msg=results[0][1])
        else:
            flash("Invalid Email/password", "danger")
            return render_template('user.html', msg="Login Failure!!!")

    return render_template('user.html')

@app.route("/userhome")
def userhome():
    return render_template("userhome.html")

# Route for uploading image
@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('images/', fn)
        myfile.save(mypath)
        print("{} is the file name".format(fn))
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)

        model = load_model(r"model\fine_tune_model.h5")
        img = image.load_img(mypath, target_size=(224, 224))
      
        img_array = img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        preprocessed_image = preprocess_input(img_array_expanded_dims)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.where(prediction > 0.5, 1, 0)  # Assuming binary classification: 1 for 'crack', 0 for 'no crack'
        print(predicted_class)
        classes = ['The model predicts this image does not have a crack.','The model predicts this image has a crack.']
        # predicted_class = classes[prediction]
        predicted_class1 = classes[predicted_class[0][0]]
        print(predicted_class1)
        # predicted_probability = prediction[0][predicted_class]
        # print("Predicted cass:", predicted_class)
        # print("Probability:", predicted_probability)

        if predicted_class1 == 'The model predicts this image has a crack.':
            classified_branches, _ = perform_crack_segmentation(img)
            severity_level = assess_severity(predicted_class, classified_branches)
            print(severity_level)
            if severity_level=="High":
                flash(f'Severity level is {severity_level}, it will urgently need to fix',"danger")
            elif severity_level=="Medium":
                flash(f'Severity level is {severity_level}, it will need to fix',"warning")
            else:
                flash(f'Severity level is {severity_level}, its a cracked but not dangerous so fix it as soon as possible',"primary")


            return render_template("template.html", image_name=fn, text=predicted_class1, severity=severity_level,classified_branches=classified_branches)
        else:
            severity_level = "Not Applicable"
            return render_template("template.html", image_name=fn, text=predicted_class1, severity=severity_level)

    return render_template('upload.html')
# Route for serving images
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/upload1')
def upload1():
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True)
