import tensorflow as tf
model = tf.keras.models.load_model('./model.h5')
import streamlit as st
st.write("""
        # 💻PNEUMONIA PREDICTOR💻
        """
        )
st.write("THIS IS A WEBAPP THAT PREDICTS WHETHER THE X-RAY IMAGE IS PNEUMONIA OR NOT")
st.write("🚨WARNING🚨 : PLEASE NOT THAT WE DO NOT TAKE ANY RESPONSIBILITY OR LIABILITY FOR ANY PROBLEMS CAUSED THROUGH THIS SERVICE.")
st.write("🚨WARNING🚨 : PLEASE DO NOT USE THIS APP FOR DIANOSIS PURPOSE.")
st.write("📧NOTICE📧 : IF YOU HAVE ANY PROBLEM OR REQUEST PLEASE CONTACT ME VIA TWITTER(@risakoml)")
st.write("📧NOTICE📧 : [LINK](https://medium.com/tmi-datascience/%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%A7%E8%83%B8%E9%83%A8x%E7%B7%9A%E7%94%BB%E5%83%8F%E3%82%92-4b86f79d145a) FOR SOURCE CODE OF TRAINING&MODELING(JAPANESE ONLY)")


file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])


import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(200,200),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    classes = ['NORMAL','PNEUMONIA']
    
    if classes[int(np.round((prediction[0])))] == 'PNEUMONIA':
        st.write("IT'S PNEUMONIA")
    else:
        st.write("IT'S NORMAL")
    
    st.text("Probability (0: NORMAL, 1: PNEUMONIA)")
    st.write(prediction)
