import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def main() -> None:
    with open("sample.txt", "r") as f:
        tokens = list(filter(lambda x: x.isalpha(), word_tokenize(f.read())))
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokens)]
        model = Doc2Vec(tagged_data, vector_size = 5, window = 2, min_count = 1, workers = 4)
        
        test_doc = word_tokenize("I had pizza and pasta".lower())
        test_doc_vector = model.infer_vector(test_doc)
        print(test_doc_vector)

if __name__ == "__main__":
    main()