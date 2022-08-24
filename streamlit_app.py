print('Hello')

from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

# Machine Learning
import tensorflow as tf

# Common Imports
import numpy as np
import cv2

#============================================================================

def contrastive_loss2(y_true, y_pred):
  square_pred = tf.math.square(y_pred)
  margin_square = tf.math.maximum(1 - (y_pred), 0)
  
  return(tf.math.reduce_mean(
      (1 - y_true) * square_pred + (y_true) * margin_square
  ))

model = tf.keras.models.load_model('lessFP/SiameseNetwork', custom_objects={'contrastive_loss2': contrastive_loss2})

dim = 128

img1 = cv2.imread('Data/tests/raafi.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('Data/tests/kevin.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img1 = cv2.resize(img1, (dim, dim), interpolation = cv2.INTER_AREA)/255
img2 = cv2.resize(img2, (dim, dim), interpolation = cv2.INTER_AREA)/255

pred = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])

#===========================================================================

a = 55
b = 34

z = 2*a + b

print(z)

f"""
# Face Identification with Siamese Neural Networks!
## Made by Raafi Rahman
answer = {z}
prediction = {pred}
"""


with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
