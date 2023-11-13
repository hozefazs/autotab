# Steps to test the installation

- pip install -e .
  The above line installs autotab onto your machine

- make first_npz
  the above line makes the first npz with two annotation files already stored in the guitarset annotation folder and two audio files already stored in your guitarset audio mic folder
  The output of this command is the npz files saved in data/spec_repr/c folder

- make run_first_model
  The above line runs the first train of the model, only on the one file provided to you. NOTE : The Test phase of the model will fail, because you have only one file

# Next steps

- Download the entire GuitarSet from
  https://zenodo.org/record/1422265#.YZ0JEdBBwnK
  Make sure to select the GuitarSet_audio_and_annotation.zip 7.5GB
  Unzip and place into the GuitarSet Folder inside data folder. A place holder has already been made for you

- If you wish to make new npz files, at the bottom of TabDataReprGen file, provide two numbers between 0 and 359 as n(must be even) and n + 1 in place of the 0 and 1 currently provided. Then
- make first_npz

- If you wish to train on this new npz file, search from the beginning of the file name in the data/spec_repr/full_id.csv file, and copy all lines connected to that file into the data/spec_repr/id.csv file. You can then
- make run_first_model

- (In Progress/TO DO) Parallel_Tab generation for all wav files
- (In Progress/TO DO) Train model on multiple files

# Data analysis

- Document here the project: autotab
- Description: This Project focuses on creating a CNN-Model to predict guitar tabluature based on a .wave input file.
So far the model only detects a single instrument.
- Data Source: Data from Audio Research Lab, along the Center for Digital Music at Queen Mary University.
GuitarSet, a dataset that provides high quality guitar recordings alongside rich annotations and metadata. https://guitarset.weebly.com/
- Type of analysis: Training a model based on the Data and trying to predict an accurate tabluatur. 

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:

```bash
make clean install test
```

Check for autotab in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/autotab`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "autotab"
git remote add origin git@github.com:{group}/autotab.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
autotab-run
```

# Install

Go to `https://github.com/{group}/autotab` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/autotab.git
cd autotab
pip install -r requirements.txt
make clean install test                # install and test
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
autotab-run
```
