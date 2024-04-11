import os
import json
import datetime
import nltk
from nltk.corpus import stopwords
import string

def download_assets():
    nltk_assets = ['punkt', 'wordnet', 'stopwords']
    for asset in nltk_assets:
        try:
            nltk.data.find(f'corpora/{asset}')
        except LookupError:
            nltk.download(asset)

def preprocess_text(text):
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text.translate(translator)
    tokens = nltk.word_tokenize(text_no_punctuation)
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def write_list_to_file(directory, filename, lst):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(lst, file)

def read_list_from_file(directory, filename):
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return None  # or an empty list, depending on your needs

def logger(correct, incorrect):
    logging_path = r"C:\Users\J C SINGLA\Downloads\External - take_home_challenge_(withJSONs)\take_home_challenge_(withJSONs)\logs"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_folder_path = os.path.join('logs', f'log_{timestamp}')
    os.makedirs(new_folder_path, exist_ok=True)
    with open(os.path.join(new_folder_path, "correct.json"), 'w') as file:
        json.dump(correct, file, indent=4)

    with open(os.path.join(new_folder_path, "incorrect.json"), 'w') as file:
        json.dump(incorrect, file, indent=4)