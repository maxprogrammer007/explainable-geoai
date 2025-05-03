import os

# Base directory of the project (one level up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'voting_2021.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'voting_clean.csv')
