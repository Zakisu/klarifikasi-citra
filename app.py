from flask import Flask, render_template, request, flash, url_for
import os
import numpy as np
from skimage.io import imread
import warnings
import glob
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename, redirect

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Declare path
path = os.getcwd() + "/data_training/"

# Declare Upload Folder
UPLOAD_FOLDER = os.getcwd() + '/upload'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

# Method for processing the model
def model(image_upload):
    global path
    data_image = []
    data_label = []
    for i in range(10):
        if i == 9:
            sleep(1)
        files = glob.glob(path + str(i) + '/*.jpg')
        for myFile in files:
            image = imread(myFile)
            data_image.append(image)
            data_label.append(i)

    data_image = np.array(data_image)
    data_label = np.array(data_label)

    data_image = data_image.reshape((data_image.shape[0], -1))

    X = data_image
    y = data_label

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=555)

    rf = RandomForestClassifier(random_state=555).fit(X_train, y_train)

    test_image = imread(os.getcwd() + '/upload/' + image_upload, as_gray=True)

    test_flat = test_image.flatten()
    test_flat.shape
    test_data = [test_flat]
    test_data = np.array(test_data)

    return rf.predict(test_data)

# Method to check allowed extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main route to proccess
@app.route('/', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    status = 0
    output = ''
    # If request method is POST, here
    if request.method == 'POST':
        # check file exist or not
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # get the prediction
            output = model(filename)[0]
            status = 1
            os.remove(os.getcwd()+'/upload/'+filename)
        return render_template('index.html', output=output, status=status)
    # If request method is GET, here
    else:
        return render_template('index.html', status=status)



if __name__ == '__main__':
    app.run()
