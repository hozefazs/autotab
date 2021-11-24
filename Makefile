# Environment variables

# project id - replace with your GCP project id
PROJECT_ID=wagon-bootcamp-328205

# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-data-737-sadriwala

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
		@gcloud config set project ${PROJECT_ID}

create_bucket:
		@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="/mnt/d/data-science/le-wagon/autotab/data"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
#BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})
BUCKET_FILE_NAME=.

##### Machine configuration - - - - - - - - - - - - - - - -

REGION=europe-west1

PYTHON_VERSION=3.7
FRAMEWORK=TensorFlow
RUNTIME_VERSION=2.6

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=autotab
FILENAME=TabCNN

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=autotab_$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
		# @gsutil -m cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
		@gsutil -m cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}


# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
first_npz:
	@python autotab/TabDataReprGen.py

run_first_model:
	@python autotab/TabCNN.py

install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* autotab/*.py

black:
	@black scripts/* autotab/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr autotab-*.dist-info
	@rm -fr autotab.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
