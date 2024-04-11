import hydra
from omegaconf import OmegaConf
import pandas as pd
import os
import json
import importlib
from src.rechunker import Rechunker
from src.faiss.flat_idx import flatIdx
from utils.utils import read_list_from_file, logger, download_assets
from src.eval import eval_retrieval
from src.post_processing import idk
from pathlib import Path

# TODO: Add Hydra logging
class Workspace: 
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.setup()
        self.encoder = self.load_encoder(cfg.encoder)
        self.encoder.load_model()
        self.ground_truth, self.all_data, self.test_data, self.test_labels = self.load_data(cfg)

    def faiss_index(self):
        embeddings = self.encoder.get_embeddings(self.all_data)
        self.index = flatIdx(embeddings.shape[1])
        self.index.add_idx(embeddings)
        return self.index

    def load_encoder(self, encoder_cfg):
        return hydra.utils.instantiate(encoder_cfg)

    def load_data(self, cfg):
        ground_truth_path = cfg.ground_truth_path
        ground_truth = pd.read_excel(ground_truth_path)
        all_data = read_list_from_file(cfg.data_path, cfg.data_name)
        test_data = list(ground_truth["relevant questions"])
        test_labels = list(ground_truth["answer"])
        return ground_truth, all_data, test_data, test_labels 

    def setup(self):
        download_assets()
    
    def retrieve(self):
        query = self.encoder.get_query(self.test_data, clean = self.cfg.clean)
        self.retrieved_items, self.similarity = self.index.get_idx(self.all_data, query, self.cfg.k)
        if self.cfg.idk:
            self.test_labels, self.retrieved_items, self.test_data = idk(self.test_labels, self.retrieved_items, self.similarity, self.test_data)
        return self.retrieved_items
    
    def eval(self):
        metric = eval_retrieval()
        recall, incorrect, correct = metric.recall_k(self.test_labels, self.retrieved_items, k = self.cfg.recall_k)
        return recall

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    workspace = Workspace(cfg)
    index = workspace.faiss_index()
    retrieved_items = workspace.retrieve()
    print (workspace.eval())

if __name__ == "__main__":
    main()
