import re
from tqdm import tqdm
from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def PartOfSpeech(word):
	tag = pos_tag([word])[0][1]
	if tag.startswith('J'):   return wordnet.ADJ
	elif tag.startswith('V'): return wordnet.VERB
	elif tag.startswith('N'): return wordnet.NOUN
	elif tag.startswith('R'): return wordnet.ADV
	else: return wordnet.NOUN

def Tokenize(data):
	for k, sent in tqdm(enumerate(data), 'Tokenization'):
		data[k] = word_tokenize(sent.lower())
	return data

def Garbages(data):
	garb = ['url', 'user', '@', '\'ve', 'n\'t', '\'s', '\'m']
	for k, sent in tqdm(enumerate(data), 'Garbages Removing'):
		data[k] = [word for word in sent if word not in garb and not re.match(r'[^a-zA-Z]+', word)]
	return data

def Stopword(data):
	stop = set(stopwords.words("english"))
	for k, sent in tqdm(enumerate(data),'Stopwords Removing'):
		data[k] = [word for word in sent if word not in stop]
	return data

def Lematize(data):
	lema = WordNetLemmatizer()
	for i, sent in tqdm(enumerate(data), 'Lemmatization'):
		for j, word in enumerate(sent):
			data[i][j] = lema.lemmatize(word, pos=PartOfSpeech(word))
	return data
