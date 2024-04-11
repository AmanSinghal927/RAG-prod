from transformers import AutoTokenizer, AutoModel
import torch


class Encoder:
    def __init__(self, context_model_name, query_model_name):
        self.context = None
        self.query = None
        self.q_embedding = None
        self.c_embedding = None
        self.tokenizer = None
        self.context_encoder = None
        self.query_encoder = None

        self.context_model_name = context_model_name
        self.query_model_name = query_model_name

        self.tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.context_model_name)
        self.context_encoder = AutoModel.from_pretrained(self.context_model_name)
        self.context_encoder = AutoModel.from_pretrained(self.context_model_name)
        self.query_encoder = AutoModel.from_pretrained(self.query_model_name)

    def concat_embeddings(self, data, typ, sz = (0, 768)):
        ctx_emb = torch.empty(sz)
        for i in range(len(data)):
            ctx_input = self.tokenizer(data[i:i+1], padding=True, truncation=True, return_tensors='pt', max_length = 512)
            if typ == "context":
                temp_emb = self.context_encoder(**ctx_input).last_hidden_state[:, 0, :]
            else:
                temp_emb = self.context_encoder(**ctx_input).last_hidden_state[:, 0, :]
            ctx_emb = torch.cat((ctx_emb, temp_emb), dim=0)
        return ctx_emb

    def get_embeddings(self, data, clean = False):
        return self.concat_embeddings(data, typ="context")
    
    def get_query(self, query, clean = False):
        return self.concat_embeddings(query, typ = "query")

