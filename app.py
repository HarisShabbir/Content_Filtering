import numpy as np
from skimage.io import imread
from flask import Flask, request, jsonify, render_template
from skimage.transform import resize
import pickle

from matplotlib import pyplot as plt

# Create flask app
Content_Filtering = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@Content_Filtering.route("/", methods=['GET'])
def Home():
    return render_template("index.html")



@Content_Filtering.route("/", methods= ["POST"])
def predict():


    imagefile =request.files['imagefile']

    imagepath ="./images/" + imagefile.filename

    imagefile.save(imagepath)

    # return render_template("index.html", prediction_text="The flower species is {}".format(prediction))

#New Code for prediction
    Categories = ['Adult', 'Sensitive', 'Non_Adult']

    img = imread(imagepath)
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    ans = model.predict(l)
    probability = model.predict_proba(l)
    if ans == 0:
        classification='Adult Image'
        print('Adult Image')

    elif ans == 1:
        classification = 'Sensitive Image'
        print('Sensitive Image')

    else:
        classification = 'Non-Adult Image'
        print('Non-Adult Image')

    # return render_template('index.html', prediction= classification, user_image=imagepath)
    return render_template('index.html', prediction=classification)

    print(classification)

if __name__ == "__main__":
    # Content_Filtering.run(debug=True)
     Content_Filtering.run( debug=True)
