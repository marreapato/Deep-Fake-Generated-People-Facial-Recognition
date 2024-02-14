from flask import Flask, render_template, request
import numpy as np
import os
from model import image_pre,predict

app = Flask(__name__, template_folder='template')

UPLOAD_FOLDER = 'D:\\deepfacedetec\\app\\uploads\\static'
ALLOWED_EXTENSIONS = set(['jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/',methods = ['GET','POST'])

def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'],'input.jpg')
        file1.save(path)
        data = image_pre(path)
        s = predict(data)
        s = s*100
        #if s == 1:
        result = 'Probability of not being an AI generated fake person = '+str(s)[0:5]+"%"
        #else:
        #   result = 'Fake Face'
    return render_template('index.html',result = result)


if __name__=='__main__':
    app.run(debug=True)