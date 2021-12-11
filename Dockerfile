FROM python:3.8.12-slim-buster

COPY h5-model /h5-model
COPY autotab /autotab
COPY data/spec_repr /data/spec_repr
COPY scripts /scripts
COPY streamlit /streamlit
COPY Makefile /Makefile
COPY MANIFEST.in /MANIFEST.in
COPY README.md /README.md
COPY Procfile /Procfile
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY api/api_app.py /api/api_app.py
COPY le-wagon-737-a89ea614b8b4.json /credentials.json


RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 curl


# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

ENV GOOGLE_APPLICATION_CREDENTIALS /credentials.json
RUN gcloud config set project le-wagon-737
RUN gcloud auth activate-service-account --key-file=/credentials.json



CMD uvicorn api.api_app:app --host 0.0.0.0 --port $PORT