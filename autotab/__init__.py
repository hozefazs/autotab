from os.path import isfile
from os.path import dirname
from os.path import join
from dotenv import load_dotenv  # for using the .env file fo renvironment

version_file = '{}/version.txt'.format(dirname(__file__))

if isfile(version_file):
    with open(version_file) as version_file:
        __version__ = version_file.read().strip()

env_path = join(dirname(dirname(__file__)), '.env')  # ../.env
load_dotenv(dotenv_path=env_path)  # for using the .env file fo renvironment
