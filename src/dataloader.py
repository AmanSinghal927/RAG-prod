import pandas as pd
import json 
from sentence_transformers import SentenceTransformer
import glob
import faiss
from fuzzywuzzy import fuzz
from llmsherpa.readers import LayoutPDFReader
import matplotlib.pyplot as plt
import numpy as np

class DataLoader:
    def __init__(self, raw_text_path, pdf_path):
        self.directory_path = raw_text_path
        self.pdf_path = pdf_path
        
    def split_data(self, data):
        return data["raw_text"].split("\n")

    def read_json_from_folders(self):
        """
        Reads all JSON files from each folder in the specified directory.

        :param directory_path: Path to the directory containing folders of JSON files.
        :return: A list of dictionaries where each dictionary contains data from a single JSON file.
        """
        directory_path = self.directory_path
        all_data = []
        filenames = []
        search_pattern = f"{directory_path}/*/*.json"
        for file_path in glob.glob(search_pattern):
            with open(file_path, 'r') as file:
                data = json.load(file)
                data = self.split_data(data)
                filenames.extend([file_path]*len(data))
                all_data.extend(data)
        return all_data, filenames