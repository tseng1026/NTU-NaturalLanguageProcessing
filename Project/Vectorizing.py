import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def Vectorize(data):
	modl = Word2Vec(data, size=80, window=15, min_count=10, workers=5)
	vect = modl.wv
	for k, sent in tqdm(enumerate(data), 'Vectorization I'):
		data[k] = np.array([vect[word] for word in sent if word in modl]).flatten()
	
	max_len = np.max([len(sent) for sent in data])
	for k, sent in tqdm(enumerate(data), 'Vectorization II'):
		data[k] = np.array(sent.tolist() + [0] * (max_len - len(sent)))
	return data
