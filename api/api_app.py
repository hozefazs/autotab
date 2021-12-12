from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from autotab.TabDataReprGen import TabDataReprGen
import autotab.TabPrediction as tp
from autotab import param
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = tp.load_model_and_weights()
model.load_weights('./h5-model/full_val0_75acc_weights.h5')
genrep = TabDataReprGen()



# get filename from User Interface! 

# @app.post('/filename')
# async def post_filename(filename):
#     uploaded_file = uploaded_file + filename #'./data/GuitarSet/audio/audio_mic/00_BN1-129-Eb_comp_mic.wav'  #'./test-music/rec1.wav' #TODO:
#     return filename
    
#########################
# print(uploaded_file)

# root endpoint 
@app.get('/ergonomic_simple')
def ergonomic_simple(uploaded_file):
    y_pred = load_predict(uploaded_file)   
    all_frames_tab = tp.make_smart_tab(y_pred, len(y_pred))
    simple_tabs = tp.make_squeezed_tab(all_frames_tab)
    simple_text = tp.web_tabs(simple_tabs) 
    
    ###################################################
    return {'simple_text': simple_text}

@app.get('/ergonomic_rhythm')
def ergonomic_rhythm(uploaded_file):
    y_pred = load_predict(uploaded_file)
    all_frames_tab = tp.make_smart_tab(y_pred, len(y_pred))
    rythm_tabs = tp.make_dynamic_tab(all_frames_tab)
    rythm_text = tp.web_tabs(rythm_tabs)
    
    ###################################################
    return {'simple_text': rythm_text}

@app.get('/all_frames')
def all_frames(uploaded_file):
    y_pred = load_predict(uploaded_file)
    all_frames_tab = tp.make_smart_tab(y_pred, len(y_pred))
    all_frames_text = tp.web_tabs(all_frames_tab)
    print(all_frames_text)
    
    ###################################################
    return {'simple_text': all_frames_text}

def load_predict(uploaded_file):
    print(uploaded_file)
    if uploaded_file is not None:     
        get_gcloud_file(uploaded_file)   
        x_new = genrep.load_rep_from_raw_file(uploaded_file)
        y_pred = model.predict(x_new)
    return y_pred

# Testing out of of the server in docker
print(os.getcwd())

#########################
def get_gcloud_file(filename):
    
#gcloud storage
    try: 
        gcloud_path = f'gsutil cp -n gs://{param.BUCKET_NAME}/{filename} .'
    except:
        time.sleep(5)
    print(gcloud_path)
    os.popen(gcloud_path)
