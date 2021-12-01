import os
import streamlit as st
from autotab.TabPrediction import load_model_and_weights,load_x_new, make_full_tab, make_squeezed_tab, web_tabs
import pandas as pd
import numpy as np 
from autotab.TabDataReprGen import TabDataReprGen
from PIL import Image
import base64

st.set_page_config(
    page_title="AutoTab tab generator",
    layout="centered", # centered
    initial_sidebar_state="auto") # collapsed

# @st.cache
# def load_image(path):
#     with open(path, 'rb') as f:
#         data = f.read()
#     encoded = base64.b64encode(data).decode()
#     return encoded

# def background_image_style(path):
#     encoded = load_image(path)
#     style = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{encoded}");
#         background-size: contain;
#     }}
#     </style>
#     '''
#     return style
# image_path = 'guitar_icon.png'
# st.write(background_image_style(image_path), unsafe_allow_html=True)
# # guitar_icon = Image.open(image_path)
# st.image(image_path, use_column_width=False)


st.title("""
        AutoTab tab generator    
""")

st.title('Model')

model = load_model_and_weights()
model.load_weights('./h5-model/full_val0_75acc_weights.h5')

st.write(model.weights[0].shape)

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("choose a music file", type="wav")

st.title("""
            The predicted Tabs are: 
""")

if uploaded_file is not None: 
    genrep = TabDataReprGen()
    x_new = genrep.load_rep_from_raw_file(uploaded_file)
    st.write(x_new.shape)
    y_pred = model.predict(x_new)
    
    expanded_tab = make_full_tab(y_pred, len(y_pred))
    display_tab = make_squeezed_tab(expanded_tab)
    st.text(web_tabs(display_tab))
    
#     st.write(pd.DataFrame(display_tab))
# test_string = """
# e|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|-
# B|-----------88---|-------8888-----|---------------8|8---------------|-------66------6|----------------|---------------8|----------------|----------------|----------------|----------------|----------------|-
# G|-----------7773-|----------------|----------------|-------7--------|----------------|----------------|----------------|----------------|----------------|---5------------|----------------|----------------|-
# D|--------------88|888--------88---|---------8----5-|-----------88-6-|6666--66------66|--------66666---|----------------|-------88-----88|----------3---88|---------------6|----------------|----------------|-
# A|----------------|----6-----------|---------------6|6---------------|----------------|----------------|66--------------|----------------|----------------|----------------|----------------|----------------|-
# E|----------------|----------------|----------------|----------------|444-------------|----------------|----------------|----------------|----------------|---------44-----|----------------|----------------|-
# """

# # Using readlines()
# with open('.tab.txt', 'r') as f:
#     lines = f.readlines()
# f.close()
# st.text(lines)


