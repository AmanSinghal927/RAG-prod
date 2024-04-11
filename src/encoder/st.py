from sentence_transformers import SentenceTransformer

class Encoder:
    def __init__(self, context_model_name='bert-base-nli-mean-tokens', batch_size = 128, device = None):
        self.embedding = None
        self.device = device
        self.batch_size = batch_size
        self.context_model_name = context_model_name

    def load_model(self):
        self.context_encoder = SentenceTransformer(self.context_model_name, self.device)

    def get_embeddings(self, data):
        embedding = self.context_encoder.encode(data, batch_size = self.batch_size)
        return embedding
    
    def get_query(self, query):
        embedding = self.context_encoder.encode(query, batch_size = self.batch_size)
        return embedding
    
        