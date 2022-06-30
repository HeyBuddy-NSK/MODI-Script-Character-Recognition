# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:40:04 2022

@author: DVSP with Nitin
"""

from flask import Flask, render_template, request, url_for, send_from_directory
import cv2 as cv
import keras
import numpy as np
import os 


app = Flask(__name__)

model = keras.models.load_model("trained_model/MODI_CHR_REC")


ALLOWED_EXTENSIONS = ['png','jpg','jpeg']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


## Transliterating to marathi
modi_to_marathi = {1:'अ', 2:'आ', 3:'ई', 4:'ऊ', 5:'ए', 6:'ऐ', 7:'ओ', 8:'औ', 9:'अं', 10:'अः', 11:'क', 12:'ख', 13:'ग', 14:'घ', 15:'ङ', 16:'च', 17:'छ', 18:'ज', 19:'झ', 20:'ञ', 21:'ट', 22:'ठ', 23:'ड', 24:'ढ', 25:'ण', 26:'त', 27:'थ', 28:'द', 29:'ध', 30:'न', 31:'प', 32:'फ', 33:'ब', 34:'भ', 35:'म', 36:'य', 37:'र', 38:'ल', 39:'व', 40:'श', 41:'ष', 42:'स', 43:'ह', 44:'ळ', 45:'क्ष', 46:'ज्ञ'} 

## Creating upload folder for saving uploaded images
path = os.getcwd()

UPLOAD_FOLDER = os.path.join(path, 'uploads\\')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


## Final Code for app
@app.route('/',methods=['GET'])
def HomePage():
    return render_template("upload.html")


def predict_img(img_path):
    img = cv.imread(img_path,0)
    img = cv.resize(img,(96,96))
    
    # reshaping image for model
    img = img.reshape((1,96,96,1)).astype('float32')
    
    # converting to range between 0-1
    img = img/255
    
    result = model.predict(img)
    
    perc = np.amax(result)
    pred = np.argmax(result[0])
    
    return f"Recognized Character in Marathi Language : {modi_to_marathi[pred]}", f" Confidence : {perc*100:.2f}"


@app.route('/prediction',methods=['GET','POST'])
def upload_page():
    if request.method=='POST':
        if 'file' not in request.files:
            return render_template("upload.html", msg='No File Selected!')
        
        file = request.files['file']
        
        if file.filename=='':
            return render_template("upload.html", msg='No File!')
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            
            img_src = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
            
            ans, confidence = predict_img(img_src)
            
            return render_template("upload.html", msg = "Character Recognition Completed!",
                                   answer = ans,
                                   confidence = confidence,
                                   user_image=file.filename)
            
        else:
            return render_template('upload.html')
        
@app.route("/uploads/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
        

if __name__=="__main__":
    # app.run(debug=True,use_reloader=False)
    app.run(debug=True)