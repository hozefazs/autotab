###### ENVIRONMENT VARIABLES FOR AUTOTAB #########
import os

GCP = True
BUCKET_NAME = "wagon-data-737-sadriwala"
DATA_BUCKET_FOLDER = "data/spec_repr"
LOCAL_DATA = os.path.abspath("data/spec_repr") + "/"
LOCAL_MODEL = os.path.abspath("h5-model/full_val0_75acc_weights.h5")

if __name__== '__main__':
    print(LOCAL_DATA)
    print(LOCAL_MODEL)