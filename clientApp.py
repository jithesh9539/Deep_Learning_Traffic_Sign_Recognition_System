from doctest import debug
from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

app = Flask(__name__)
# app.config["DEBUG"] = True

# Classes of trafic signs
classes={
    0:'Speed limit (20Km/Hr)',
    1:'Speed limit (30Km/Hr)',
    2:'Speed limit (50Km/Hr)',
    3:'Speed limit (60Km/Hr)',
    4:'Speed limit (70Km/Hr)',
    5:'Speed limit (80Km/Hr)',
    6:'End of Speed limit (80Km/Hr)',
    7:'Speed limit (100Km/Hr)',
    8:'Speed limit (120Km/Hr)',
    9:'No passing',
    10:'No passing vehicle over 3.5 tons',
    11:'Right of way at the intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No Vehicles',
    16:'Vehicles more than 3.5 tons prohibited',
    17:'No entry',
    18:'General Caution',
    19:'Dangerous Curve Left',
    20:'Dangerous Curve Right',
    21:'Double Curve',
    22:'Bumpy Road',
    23:'Slippery Road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic Signals',
    27:'Pedestrians',
    28:'Children Crossing',
    29:'Bicycles Crossing',
    30:'Beware of Ice/Snow',
    31:'Wild Animals Crossing',
    32:'End Speed and passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep Right',
    39:'Keep Left',
    40:'Roundabout Mandatory',
    41:'End of no passing',
    42:'End of passing vehicles more than 3.5 tons' }

def image_processing(img):
    print("in img_processing")
    print(img)
    model = load_model('./model_traffic_sign.h5')
    data=[]

    image=cv2.imread(img)
    image_fromarray=Image.fromarray(image,'RGB')
    resize_image=image_fromarray.resize((30,30))
    data.append(np.array(resize_image))
    X_test=np.array(data)
    X_test=X_test/255
    pred=np.argmax(model.predict(X_test),axis=1)
    return pred

    # image = Image.open(img)
    # image = image.resize((30,30))
    # data.append(np.array(image))
    # X_test=np.array(data)
    # Y_pred = np.argmax(model.predict(X_test),axis=1)
    # return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        print("filepath"+file_path)
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted Traffic Sign is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)