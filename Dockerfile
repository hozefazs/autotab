FROM python:3.8.6-buster

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

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 

CMD uvicorn api.api_app:app --host 0.0.0.0 --port $PORT