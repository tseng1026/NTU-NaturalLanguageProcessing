import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import Parsing
import Loading
import Preprocessing
import Vectorizing
import Classifying
from argparse import ArgumentParser

### Parsing Arguments
args = Parsing.Args()
dataname = args.d
waysname = args.m
taskname = args.t

### Loading Data
data, task = Loading.LoadData(dataname, taskname)

### Preprocessing
data = Preprocessing.Tokenize(data)
data = Preprocessing.Garbages(data)
data = Preprocessing.Stopword(data)
data = Preprocessing.Lematize(data)
data = Vectorizing.Vectorize(data)
data = np.array(data)

### Classifying
accuracy, f1_score = Classifying.Classify(data, task, waysname)
print ("accuracy", accuracy)
print ("f1_score", f1_score)
