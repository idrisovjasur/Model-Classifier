import streamlit as st

import pathlib

from fastai.vision.core import PILImage

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


####streamlit
from fastai.learner import load_learner

st.title('Transport klassifkatsiya qiluvchi model_AI')
file = st.file_uploader('Rasim yuklash' , type = ['png','jpeg','jpg','gif','svg'])

if file:
    st.image(file)
    image = PILImage.create(file)
    model = load_learner('model_classes.pkl')

    pred , pred_id , probs = model.predict(image)
    st.success(f"Bashorat:{pred}")
    st.info(f'Ehtimolligi:{probs[pred_id]}')

    