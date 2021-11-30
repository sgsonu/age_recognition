from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
import timm
from utils import Ranger



from fastai import *
from fastai.vision import *

from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)

learn = load_learner('./path/models/')


def model_predict(img_path):
    img = open_image(img_path)
    ans = learn.predict(img)
    print(ans)
    return int(ans[1].item())



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path)
        return str(preds)

    return None


if __name__ == '__main__':    
    app.run()


