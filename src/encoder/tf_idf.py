from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text.translate(translator)
    tokens = nltk.word_tokenize(text_no_punctuation)
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word not in stop_words]
    return ' '.join(tokens)

class Encoder:
    def __init__(self):
        self.data = None
        self.embedding = None
        self.model = None
    
    def load_model(self):
        self.model = TfidfVectorizer()

    def get_embeddings(self, data, clean = False):
        if clean:
            data = [preprocess_text(x) for x in data]
        tfidf_vectors = self.model.fit_transform(data)
        embedding = tfidf_vectors.toarray()
        return embedding

    def get_query(self, query, clean = False):
        tfidf_vectors = self.model.transform(query)
        embedding = tfidf_vectors.toarray().astype('float32')
        return embedding
        
        