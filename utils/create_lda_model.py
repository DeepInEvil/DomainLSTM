import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models
import gc
import sys
import os.path
import re
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#get stopwords
stop = stopwords.words('english')
file_n = sys.argv[1]
lda_model = './lda_models/amazon_lda'
lda_dict = './lda_models/amazon_dict'
#read all data
print "read the data"
data = pd.read_table(file_n, error_bad_lines=False)
data = data.fillna('na')
#data = np.genfromtxt('./new_dat/all_reviews.txt')
data = np.array(data)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9()\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()
    
   
print data[0]
print "processing the data now"
#new_dat = np.empty(len(data))
#print new_dat.shape
#data = [sent[0].split() for sent in data]
#data = [[word for word in sent if word not in stop] for sent in new_dat]
data = [clean_str(sent[0]) for sent in data]
print data[0]
data = [[word for word in sent if word not in stop] for sent in data]
print data[0]
data = [[word for word in sent if word not in ('\\)', '\\(')] for sent in data]
print data[0]
#load the lda dictionary
dictionary = gensim.corpora.Dictionary.load(lda_dict)
#create bow with the dictionary
bow_dat = [dictionary.doc2bow(sent) for sent in data]
if not os.path.isfile(lda_model):
#create the lda model
   print "creating new lda models"
   lda = models.LdaMulticore(bow_dat, id2word=dictionary, num_topics=50, workers=30, passes=1)
   print "Done training the lda model"
   lda.save('./lda_models/amazon_lda')
else:
   print "updating previous ones"
   lda = models.LdaModel.load(lda_model)
   lda.update(bow_dat)
   lda.save(lda_model)
