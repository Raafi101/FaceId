# Machine Learning
import tensorflow as tf

# Common Imports
import numpy as np
import cv2
import os

from flask import Flask, render_template, request

def contrastive_loss2(y_true, y_pred):
  square_pred = tf.math.square(y_pred)
  margin_square = tf.math.maximum(1 - (y_pred), 0)
  
  return(tf.math.reduce_mean(
      (1 - y_true) * square_pred + (y_true) * margin_square
  ))

model = tf.keras.models.load_model('lessFP.h5', custom_objects={'contrastive_loss2': contrastive_loss2})

dim = 128

path1 = 'Data/raafi.jpg'
path2 = 'Data/raafi2.jpg'

def predict(path1, path2):
  img1 = cv2.imread(path1)
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

  img2 = cv2.imread(path2)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

  img1 = cv2.resize(img1, (dim, dim), interpolation = cv2.INTER_AREA)/255
  img2 = cv2.resize(img2, (dim, dim), interpolation = cv2.INTER_AREA)/255

  pred = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])

  if pred >= .5:
    pred = 'Match'
  else:
    pred = 'No Match'

  return(pred)

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods = ["GET", "POST"])
def index():
  if request.method == "POST":
    img1 = request.files["img1"]
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], img1.filename)
    img1.save(path1)
    img2 = request.files["img2"]
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], img2.filename)
    img2.save(path2)
    pred = predict(path1, path2)
    return render_template("index.html", prediction = pred)
  return render_template("index.html", prediction = "N/A")