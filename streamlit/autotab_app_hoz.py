import os
import streamlit as st
import autotab.TabPrediction as tp
import pandas as pd
import numpy as np
from autotab.TabDataReprGen import TabDataReprGen
from PIL import Image

st.set_page_config(
    page_title="AutoTab tab generator",
    layout="centered",  # centered
    initial_sidebar_state="auto")  # collapsed

image_path = "https://images.unsplash.com/photo-1535587566541-97121a128dc5?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80"
st.write(
    f'<div class="banner" style="background-image: linear-gradient(rgba(0,0,0,0.4),rgba(0,0,0,0.4)), url({image_path});"><h1>Autotab</h1><p>Learning the guitar the easy way</p></div>',
    unsafe_allow_html=True)
CSS = """
.banner {
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
    position: relative;
    height: 300px;
    text-align: center;
    margin-top: -100px;
    margin-left: -480px;
    margin-right: -480px;
}
.banner h1 {
    padding-top: 120px;
    margin: 0;
    color: white;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    font-size: 56px;
    font-weight: bold;
}
.banner p {
    font-size: 32px;
    color: white;
    opacity: .7;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
}
"""


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def init_model():
    model = tp.load_model_and_weights()
    model.load_weights('./h5-model/full_val0_75acc_weights.h5')
    genrep = TabDataReprGen()
    return model, genrep


@st.cache(suppress_st_warning=True)
def make_pred(uploaded_file):
    model, genrep = init_model()
    x_new = genrep.load_rep_from_raw_file(uploaded_file)
    y_pred = model.predict(x_new)
    #processed_filename = uploaded_file.name
    return y_pred


@st.cache(suppress_st_warning=True)
def get_all_frame_tabs(uploaded_file):
    y_pred = make_pred(uploaded_file)
    all_frames_tab = tp.make_smart_tab(y_pred, len(y_pred))
    return all_frames_tab


st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("choose a music file:", type="wav")

processed_filename = ""

mode = st.radio('Choose Mode of Tab production:',
                ('Ergonomic Simple', 'Ergonomic Melody', 'All Frames'))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
         unsafe_allow_html=True)

st.session_state.init_counter = 1
st.session_state.pred_counter = 1

if uploaded_file is not None:
    ##############load as an audio file################
    #read from beginnning of file (position 0)
    uploaded_file.seek(0)
    st.audio(uploaded_file.read(), format='audio/wav')
    #seek back to the beginning of the file, so that other actions can be done on it
    uploaded_file.seek(0)
    ####################################################
    y_pred = make_pred(uploaded_file)

if uploaded_file is not None and mode == 'Ergonomic Simple':
    st.title(f"""
            Ergonomic Simple predicted Tabs:
    """)
    # if uploaded_file.name != processed_filename:
    #     make_pred()

    all_frames_tab = get_all_frame_tabs(uploaded_file)
    simple_tabs = tp.make_squeezed_tab(all_frames_tab)
    simple_text = tp.web_tabs(simple_tabs)
    st.text(simple_text)

if uploaded_file is not None and mode == 'Ergonomic Melody':
    st.title(f"""
            Ergonomic Melody predicted Tabs:
    """)
    # if uploaded_file.name != processed_filename:
    #     make_pred()

    all_frames_tab = get_all_frame_tabs(uploaded_file)
    melody_tabs = tp.make_dynamic_tab(all_frames_tab)
    melody_text = tp.web_tabs(melody_tabs)
    st.text(melody_text)

if uploaded_file is not None and mode == 'All Frames':
    st.title(f"""
            All Frames predicted Tabs:
    """)
    # if uploaded_file.name != processed_filename:
    #     make_pred()

    all_frames_tab = get_all_frame_tabs(uploaded_file)
    ################################################
    # all_frames_tab.to_csv('all_frames_tab.csv')
    ################################################
    all_frames_text = tp.web_tabs(all_frames_tab)
    st.text(all_frames_text)
