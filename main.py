import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras as keras
import cv2
import tensorflow as tf
import math
from keras.utils import Sequence
import streamlit as st
import efficientnet.keras as efn
from PIL import Image

class predict_gen(Sequence):
  
  def __init__(self, tup , batch_size):
    self.tup = tup
    self.batch_length = batch_size
    self.size = 1

  def __getitem__(self, idx):
    start = idx*self.batch_length
    end = min((idx+1)*self.batch_length, self.size)
    (one,two) = self.tup
    
    lis = []
    for i in range(start,end):
      img = cv2.imread('img.jpg')
      
      if len(img.shape)==2:
            img = np.stack((img,)*3, axis=-1)
          
      #scaling
      (l,b,n_c) = img.shape
      mini = min(l,b)
      fraction = 224.0/mini
      img_scaled = cv2.resize( img , ( int((b/mini)*224) , int((l/mini)*224)  )  )
      
      #cropping
      (l,b,n_c) = img_scaled.shape
      ran = one
      if l>b:
        img_scaled_crpd = img_scaled[ (l-224)*ran + 0  : (l-224)*ran  + 224 , : ]
      else:
        img_scaled_crpd = img_scaled[ :, (b-224)*ran + 0  : (b-224)*ran  + 224 ]
        
      #flipping  
      if two:
            img_scaled_crpd = img_scaled_crpd[:,::-1]
          
      lis.append(img_scaled_crpd/255.0)
    lis = np.array( lis )
  
    
    
    return lis
  def __len__(self):
    return math.ceil(self.size / self.batch_length)


label_dict = np.array(['antelope', 'bat', 'beaver', 'bobcat', 'buffalo', 'chihuahua',
       'chimpanzee', 'collie', 'dalmatian', 'german+shepherd',
       'grizzly+bear', 'hippopotamus', 'horse', 'killer+whale', 'mole',
       'moose', 'mouse', 'otter', 'ox', 'persian+cat', 'raccoon', 'rat',
       'rhinoceros', 'seal', 'siamese+cat', 'spider+monkey', 'squirrel',
       'walrus', 'weasel', 'wolf'])


def predict():
    y1 = model.predict_generator( predict_gen( (0,0),64) )
    y2 = model.predict_generator( predict_gen( (0,1),64) )
    y3 = model.predict_generator( predict_gen( (1,0),64) )
    y4 = model.predict_generator( predict_gen( (1,1),64) )
    y = (y1 + y2 + y3 + y4)/4.0
    y = y[0]
    z = {}
    i = 0
    for x in y:
      z[i] = '%.4f'%(x)
      z[i] = round( float(x) , 4 )
      i = i+1
    x = z
    z = {k: v for k, v in sorted(x.items(),reverse = True, key=lambda item: item[1])}
    return z


from keras.models import load_model
model = load_model('model.h5')

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title( "Animal Classifcation" )
st.header( "Test Example" )

img = st.file_uploader("Upload an image of an animal", type="jpg")

if img is not None:
    img = Image.open(img)
    st.image(img, caption="User Input", width=500, use_column_width=False)
    img.save('img.jpg')

    
if st.button('Execute'):
    z = predict()
    answer = label_dict[ list(z.keys())[0] ]
    answer = answer.upper()
    df = pd.DataFrame( {'Animal': label_dict[list(z.keys())] , 'Probability' : list(z.values()) } )
    st.write(" It's a " ,answer)
    st.write( 'The predictions are as follows ; ')
    st.write(df)
