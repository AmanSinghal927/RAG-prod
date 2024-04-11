import faiss
import torch

class flatIdx:
    def __init__(self, d, type = "L2"):
        self.d = d
        self.type = type
        if type == "L2":
            self.index = faiss.IndexFlatL2(d)
        elif type == "Cosine":
            self.index = faiss.IndexFlatIP(d)
    
    def add_idx(self, sentence_embeddings):
        if isinstance(sentence_embeddings, torch.Tensor):
            sentence_embeddings = sentence_embeddings.detach().numpy()
        self.index.add(sentence_embeddings)

    def get_idx(self, indexed_data, query_data, k=4):
        ret_context = []
        if isinstance(query_data, torch.Tensor):
            query_data = query_data.detach().numpy()
        D, I = self.index.search(query_data, k)
        ret_context = [[indexed_data[i] for i in row] for row in I]
        return ret_context, D  
        


