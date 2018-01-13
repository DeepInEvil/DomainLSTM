import pandas as pd
import numpy as np
import re
from collections import defaultdict
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, GlobalAveragePooling1D, GRU, MaxPooling1D, AveragePooling1D, merge
from keras.layers.merge import Concatenate, Add, concatenate
from keras.preprocessing import sequence
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping
from gensim.models import word2vec
from sklearn.model_selection import KFold
from collections import defaultdict
from collections import Counter
from keras.utils import np_utils
from gensim import models
import gensim
from keras.layers.normalization import BatchNormalization
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from keras.models import *
from keras.layers.core import *
from keras import initializers
from LSTM_topic import LSTMCustom
from itertools import product
import functools
# Custom loss function with costs

#weighted categorical cross entropy loss function
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
w_array = np.ones((2,2))
w_array[0, 1] = 1.2
ncce = functools.partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'

#define required variables
vocab_path='/data/dchaudhu/all_unlabelled/all_vocab.npy'
#word_vec_file='/data/dchaudhu/all_unlabelled/CNN_T_learning/word_vecs/corpus_word2vec'
#word_vec_file='/data/dchaudhu/rel_ent_pred/lexvec.commoncrawl.300d.W.pos.vectors'
word_vec_file='/data/dchaudhu/all_unlabelled/CNN_T_learning/data/word_vectors/word2vec_amazon'
pos_revw_file = '/data/dchaudhu/all_dat/positive/positive_revws.tsv'
neg_revw_file = '/data/dchaudhu/all_dat/negative/negative_revws.tsv'
reviews1k = '/data/dchaudhu/metaphor2vec/data/reviews/'
reviews = '/data/dchaudhu/all_unlabelled/CNN_T_learning/data/'
model_weights_d = '/data/dchaudhu/all_unlabelled/CNN_T_learning/saved_models/300_itr/'

#lda_model = '/data/dchaudhu/all_unlabelled/CNN_T_learning/lda_models/lda_50/amazon_lda'
lda_model = '/data/dchaudhu/all_unlabelled/CNN_T_learning/new_dat/lda_models/amazon_lda'
saved_dict = '/data/dchaudhu/all_unlabelled/CNN_T_learning/new_dat/lda_models/amazon_dict'
#saved_dict = '/data/dchaudhu/all_unlabelled/CNN_T_learning/lda_models/lda_50/amazon_dict'

domains = ['electronics', 'books', 'kitchen', 'dvd']
#domains = ['kitchen', 'dvd']
stop = set(stopwords.words('english'))
words_not_found = []

#custom Attention Class
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


def get_index_to_embeddings_mapping(vocab, word_vecs):
    """
    get word embeddings matrix
    :param vocab:
    :param word_vecs:
    :return:
    """
    embeddings = {}
    for word in vocab.keys():
        try:
            embeddings[word] = word_vecs[word]
        except KeyError:
            # map index to small random vector
            # print "No embedding for word '"  + word + "'"
            words_not_found.append(word)
            embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    return embeddings


def get_word2id(word, w2idx_dict):
    """
    get word2id mapping
    :param word:
    :param w2idx_dict:
    :return:
    """
    if word.isdigit():
        return w2idx_dict['DIGIT']
    else:
        try:
            return w2idx_dict[word]
        except KeyError:
            return w2idx_dict['unk']


def get_lda_vec(lda_dict, max_len):
    """
    get lda vector
    :param lda_dict:
    :return:
    """
    lda_vec = np.zeros(50)
    for id, val in lda_dict:
        lda_vec[id] = val
    return lda_vec


def get_id2word(idx, idx2w_dict):
    """
    get id2word mappings
    :param idx:
    :param idx2w_dict:
    :return:
    """
    try:
        return idx2w_dict[idx]
    except KeyError:
        return 'unk'


def get_alpha(texts, lda, dictionari, idx2word, max_len):
    """
    get doc-topic distribution vector for all reviews
    :param texts:
    :param lda:
    :param dictionari:
    :param idx2word:
    :return:
    """
    texts = [[get_id2word(idx, idx2word) for idx in sent] for sent in texts] 
    print "sample lda vector:"
    #print texts[0]
    #print get_lda_vec(lda[dictionari.doc2bow(texts[0])]).reshape((1, 50))
    #print lda[dictionari.doc2bow(texts[0])]
    review_alphas = np.array([get_lda_vec(lda[dictionari.doc2bow(sentence)], max_len) for sentence in texts])
    #review_alphas = np.random.randn(len(texts), 50)
    return review_alphas

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    try:
    	string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)
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
    except Exception:
        return str(string).strip().lower().split()


def get_domain_revs(domain, test_t):
    """
    get data for each domain
    :param domain:
    :param word2idx:
    :return: domain reviews
    """
    print "Getting reviews for domain:" + domain
    if test_t == 'train':
	rev_d = reviews
        rev_ext = '.csv'
    elif test_t == 'test':
	rev_d = reviews1k
        rev_ext = '.tsv'

    pos_revw_file = rev_d + domain + '/' + 'positive' + rev_ext
    neg_revw_file = rev_d + domain + '/' + 'negative' + rev_ext
    positive_reviews = pd.read_table(pos_revw_file, nrows=200000)
    negative_reviews = pd.read_table(neg_revw_file, nrows=200000)

    positive_reviews['label'] = 1
    negative_reviews['label'] = 0

    #positive_reviews = np.array(positive_reviews)
    #negative_reviews = np.array(negative_reviews)

    review_text = np.concatenate((np.array(positive_reviews['review']), np.array(negative_reviews['review'])), axis=0)
    #review_text = np.array(positive_reviews['review']), np.array(negative_reviews['review'])
    #review_labels = np_utils.to_categorical(np.concatenate((np.array(positive_reviews['label']), np.array(negative_reviews['label'])), axis=0))
    review_labels = np.concatenate((np.array(positive_reviews['label']), np.array(negative_reviews['label'])))
    print review_text[0]
    review_text = [clean_str(sent) for sent in review_text]
    print review_text[0]
    #review_text = [[word for word in sent if word not in stop] for sent in review_text]
    #print review_text[0]

    #review_text = np.array([[get_word2id(word, word2idx) for word in sent] for sent in review_text])

    #print review_text.shape, review_labels.shape
    return review_text, review_labels


def get_vocab(docs):
    """
    create vocab for the training data
    :param docs:
    :return:
    """
    vocab = defaultdict(float)

    for sent in docs:
        for word in sent:
            if word.isdigit():
                vocab['DIGIT'] += 1.0
            else:
                vocab[word] += 1.0


    #vocab = { k : v for k,v in vocab.iteritems() if v < 2.0}
    try:
        print vocab['DIGIT']
    except KeyError:
        vocab['DIGIT'] = 1.0
    #word2idx = dict(zip(vocab.keys(), range(0, len(vocab))))
    #idx2word = {v: k for k, v in word2idx.iteritems()}

    return vocab


def cnn_model(embedding_weights, cv_dat, max_len, model_w, lda, dictionary, idx2word, alpha):
    max_len = 1000 if max_len > 1000 else max_len
    #max_len = 1000
    dropout = 0.8
    print max_len

    json_file = open(model_w+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_lda = model_from_json(loaded_model_json)
    # load weights into new model
    #print layer_dict
    train_x, test_x, train_y, test_y = cv_dat
    test_lda = get_alpha(test_x, lda, dictionary, idx2word)

    print "Maximum length of sentence:" + str(max_len)
    print "Distribution of labels in training set:"
    print Counter([np.argmax(dat) for dat in train_y])
    print "Distribution of labels in testing set:"
    print Counter([np.argmax(dat) for dat in test_y])

    test_x = np.array(sequence.pad_sequences(test_x, maxlen=max_len), dtype=np.int)

    #print (train_x.shape)
    #print train_y.shape
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.166, random_state=666, stratify=train_y)

    train_lda = get_alpha(train_x, lda, dictionary, idx2word)
    val_lda = get_alpha(val_x, lda, dictionary, idx2word)
    #defining the model architecture now
  

    train_x = np.array(sequence.pad_sequences(train_x, maxlen=max_len), dtype=np.int)
    val_x = np.array(sequence.pad_sequences(val_x, maxlen=max_len), dtype=np.int)


    review_text = Input(shape=(max_len ,), dtype='int64', name="body_input")
     
    embedded_layer_body = Embedding(embedding_weights.shape[0], embedding_weights.shape[1], mask_zero=False,
                                    input_length=max_len, weights=[embedding_weights], trainable=True)(review_text)
    lda_input = Input(shape=(30, ), dtype='float32', name="lda_inp")
    #load the weights from pre-trained model
    lrelu = LeakyReLU(alpha=0.1)
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation=lrelu, weights=layer_dict['conv1d_1'].get_weights())
    conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation=lrelu, weights=layer_dict['conv1d_2'].get_weights())
    conv3 = Conv1D(filters=128, kernel_size=5, padding='same', activation=lrelu, weights=layer_dict['conv1d_3'].get_weights())
  
    #conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    #conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    #conv3 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')
 

    conv1a = conv1(embedded_layer_body)
    glob1a = GlobalAveragePooling1D()(conv1a)
    #max1 = AveragePooling1D()(conv1a)
    conv2a = conv2(embedded_layer_body)
    glob2a = GlobalAveragePooling1D()(conv2a)
    #max2 = AveragePooling1D()(conv2a) 
    conv3a = conv3(embedded_layer_body)
    glob3a = GlobalAveragePooling1D()(conv3a)
    #max3 = AveragePooling1D()(conv3a)

    merge_pooling = concatenate([glob1a, glob2a, glob3a])
    #merge_pooling = concatenate([max1, max2, max3])

    hidden_layer = Dense(1200, activation='tanh', kernel_initializer="glorot_uniform")(merge_pooling)
    #hidden_concat = concatenate([hidden_layer, lda_vec])
    dropout_hidden = Dropout(dropout)(hidden_layer)
    #merge_hidden = concatenate([dropout_hidden, lda_input]) 
    batch_norm = BatchNormalization()(dropout_hidden)


    #hidden_layer_2 = Dense(600, activation='tanh', kernel_initializer="glorot_uniform")(batch_norm)
    #dropout_hidden_2 = Dropout(0.6)(hidden_layer_2)
    #batch_n_2 = BatchNormalization()(dropout_hidden_2)
   
    hidden_layer_3 = Dense(600, activation='tanh', kernel_initializer="glorot_uniform")(batch_norm)
    dropout_hidden_3 = Dropout(0.5)(hidden_layer_3)
    batch_n_3 = BatchNormalization()(dropout_hidden_3)


    output_layer = Dense(2, activation='softmax', name='out_sent')(batch_n_3)
    output_lda = Dense(30, activation='softmax', name='out_lda')(batch_n_3)

    model = Model([review_text], output=[output_layer, output_lda])
    layer_dict_nu = dict([(layer.name, layer) for layer in model.layers])
    
    adam = Adam(lr=0.001)



    model.compile(loss=['categorical_crossentropy', 'kullback_leibler_divergence'], optimizer=adam, metrics=['accuracy'], loss_weights={'out_sent': (1 - alpha), 'out_lda': alpha})
    #model.compile(loss=ncce, optimizer=adam, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_out_sent_loss', min_delta=0.0001, patience=9,
                          verbose=1, mode='auto')
    callbacks_list = [earlystop]
    print model.summary()
    model.fit([train_x], [train_y, train_lda], batch_size=32*2, epochs=50,
              verbose=1, shuffle=True, callbacks=callbacks_list, validation_data=[[val_x], [val_y, val_lda]])
    #model.fit([train_x, train_lda], [train_y, train_lda], batch_size=64, epochs=25,
    #          verbose=1, shuffle=True)
    test_predictions = model.predict([test_x], verbose=False)
    #test_y = [np.argmax(pred) for pred in test_y]
    test_pred = [np.argmax(pred) for pred in test_predictions[0]]
    #print test_pred
    test_y = [np.argmax(label) for label in test_y]
    error_preds = [i for i in range(0,len(test_pred)) if (test_y[i] != test_pred[i])]
    print len(error_preds)
    misclassified = [test_x[i] for i in error_preds] 
    misclassified = [[get_id2word(idx, idx2word) for idx in sent if idx != 0] for sent in misclassified]
    labels = [(test_y[i], test_pred[i]) for i in error_preds]
    #acc = accuracy_score(test_y, test_pred)
    print acc
    return acc, misclassified, labels

def tmlstm_model(embedding_weights, cv_dat, max_len, model_w, lda, dictionary, idx2word, alpha, gru_time_steps):
    """
    topic aware lstm model
    :param embedding_weights:
    :param cv_dat:
    :param max_len:
    :param model_w:
    :param lda:
    :param dictionary:
    :param idx2word:
    :param alpha:
    :return:
    """
    max_len = 1000 if max_len > 1000 else max_len
    #max_len = 1000
    dropout = 0.8
    print max_len

    #json_file = open(model_w+'model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model_lda = model_from_json(loaded_model_json)
    # load weights into new model
    #layer_dict = dict([(layer.name, layer) for layer in model_lda.layers])
    #print layer_dict.keys()
    ##print layer_dict
    train_x, test_x, train_y, test_y, test_x1k, test_y1k = cv_dat
    #test_lda = get_alpha(test_x, lda, dictionary, idx2word)

    print "Maximum length of sentence:" + str(max_len)
    print "Distribution of labels in training set:"
    print Counter([np.argmax(dat) for dat in train_y])
    print "Distribution of labels in testing set:"
    print Counter([np.argmax(dat) for dat in test_y])
    #print "Distribution of labels in testing set 1k:"
    #print Counter([np.argmax(dat) for dat in test_y1k])



    #print (train_x.shape)
    #print train_y.shape
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.166667, random_state=666, stratify=train_y)

    train_lda = get_alpha(train_x, lda, dictionary, idx2word, max_len)
    val_lda = get_alpha(val_x, lda, dictionary, idx2word, max_len)
    print val_lda[0]
    test_lda = get_alpha(test_x, lda, dictionary, idx2word, max_len)
    test_lda1k = get_alpha(test_x1k, lda, dictionary, idx2word, max_len)
    print val_lda.shape
    #defining the model architecture now


    train_x = np.array(sequence.pad_sequences(train_x, maxlen=max_len), dtype=np.int)
    val_x = np.array(sequence.pad_sequences(val_x, maxlen=max_len), dtype=np.int)
    test_x = np.array(sequence.pad_sequences(test_x, maxlen=max_len), dtype=np.int)
    test_x1k = np.array(sequence.pad_sequences(test_x1k, maxlen=max_len), dtype=np.int)


    review_text = Input(shape=(max_len ,), dtype='int64', name="body_input")
    lda_input = Input(shape=(50, ), dtype='float32', name="lda_inp")

    embedded_layer_body = Embedding(embedding_weights.shape[0], embedding_weights.shape[1], mask_zero=False,
                                    input_length=max_len, weights=[embedding_weights], trainable=True)(review_text)

    #define the GRU with attention
    gru_l = LSTMCustom(gru_time_steps, init='glorot_uniform',  activation='tanh',dropout_W=0.3, dropout_U=0.3, topic_distribution=lda_input)(embedded_layer_body)
    #gru_l = LSTM(gru_time_steps, init='glorot_uniform',  activation='tanh',dropout_W=0.3, dropout_U=0.3)(embedded_layer_body)
    print gru_l
    #gru_lda = merge([gru_l, lda_input], name='add_lda', mode='sum')
    #gru_lda = Lambda(lambda x: merge([x, lda_input], name='add_lda', mode='sum'))(gru_l)

    #attn_wghts = Permute((2, 1))(gru_lda)
    #print attn_wghts
    #ttn_wghts = Reshape((300, gru_time_steps))(attn_wghts)
    #attn_wghts = Dense(gru_time_steps, activation='softmax')(attn_wghts)
    #alpha_wghts = Permute((2, 1), name='attention_vec')(attn_wghts)
    #output_attention_mul = merge([attn_wghts, alpha_wghts], name='attention_mul', mode='mul')
    #attention_mul = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
    #add_lda =  merge([attention_mul, lda_input], name='add_lda', mode='sum')
    #print attention_mul
    #l_dense = TimeDistributed(Dense(gru_time_steps*2))(gru_l)
    #l_att = Attention(lda=lda_input)(gru_l)
    

    #output_lda = Dense(30, activation='softmax', name='out_lda')(attention_mul)

    hidden_layer = Dense(1200, activation='tanh', kernel_initializer="glorot_uniform")(gru_l)
    #hidden_concat = concatenate([hidden_layer, lda_vec])
    dropout_hidden = Dropout(dropout)(hidden_layer)
    #merge_hidden = concatenate([dropout_hidden, lda_input])
    batch_norm = BatchNormalization()(dropout_hidden)


    hidden_layer_3 = Dense(600, activation='tanh', kernel_initializer="glorot_uniform")(batch_norm)
    dropout_hidden_3 = Dropout(0.5)(hidden_layer_3)
    batch_n_3 = BatchNormalization()(dropout_hidden_3)

    output_layer = Dense(2, activation='softmax', name='out_sent')(batch_n_3)

    model = Model([review_text, lda_input], output=[output_layer])
    layer_dict_nu = dict([(layer.name, layer) for layer in model.layers])

    adam = Adam(lr=0.0001)

    #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'], loss_weights={'out_sent': (1 - alpha), 'out_lda': alpha})
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.compile(loss=ncce, optimizer=adam, metrics=['accuracy'])



    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3,
                          verbose=1, mode='auto')
    callbacks_list = [earlystop]
    print model.summary()
    model.fit([train_x, train_lda], [train_y], batch_size=128, epochs=100,
              verbose=1, shuffle=True, callbacks=callbacks_list, validation_data=[[val_x, val_lda], [val_y]])
    #model.fit([train_x, train_lda], [train_y, train_lda], batch_size=64, epochs=25,
    #          verbose=1, shuffle=True)
    test_predictions = model.predict([test_x, test_lda], verbose=False)
    test_predictions1k = model.predict([test_x1k, test_lda1k], verbose=False)
    #test_y = [np.argmax(pred) for pred in test_y]
    test_pred = [np.argmax(pred) for pred in test_predictions]
    test_pred1k = [np.argmax(pred) for pred in test_predictions1k]
    #print test_pred1k
    test_y = [np.argmax(label) for label in test_y]
    test_y1k = [np.argmax(label) for label in test_y1k]
    error_preds = [i for i in range(0,len(test_pred)) if (test_y[i] != test_pred[i])]
    print len(error_preds)
    misclassified = [test_x[i] for i in error_preds]
    misclassified = [[get_id2word(idx, idx2word) for idx in sent if idx != 0] for sent in misclassified]
    labels = [(test_y[i], test_pred[i]) for i in error_preds]
    acc = accuracy_score(test_y, test_pred)
    print acc
    acc1k = accuracy_score(test_y1k, test_pred1k)
    print acc1k
    return acc, acc1k, misclassified, labels

def run_model():
    """
    Run the models
    :return:
    """
    #Loading the vocabulary for all domains
    #vocab = np.load(vocab_path).item()
    #word to index mapping, +1 for oov words
    #word2idx = dict(zip(vocab.keys(), range(0, len(vocab) + 1)))
    #idx2word = {v: k for k, v in word2idx.iteritems()}

    print "Loading the LDA model....."
    lda = models.LdaModel.load(lda_model)
    dictionri = gensim.corpora.Dictionary.load(saved_dict)


    #print len(word2idx)

    print "Loading the word_vecs"
    corpus_wordvec = word2vec.Word2Vec.load(word_vec_file)
    #corpus_wordvec = gensim.models.KeyedVectors.load_word2vec_format(word_vec_file)
    print corpus_wordvec.most_similar('automobile')
    #index_to_vector_map = get_index_to_embeddings_mapping(word2idx, corpus_wordvec)

    acc_dict = {}
    acc_dict1k = {}
    test_acc = 0.0

    for domain in domains:
        x_tr = []
        y_tr = []

        train_domains = [dom for dom in domains if dom != domain]
        print train_domains
        for dom in train_domains:
           revws, labels = get_domain_revs(dom, 'train')
           #print labels
           for r in revws:
	   	x_tr.append(r)
	   for l in labels:
	   	y_tr.append(l)
      	   #other_dom = [d for d in train_domains if d != dom]
	   #for od in other_dom:
           #    od_r, od_l = get_domain_revs(od)
           #    for i in range(0, len(od_r)):
	   #	   	x_tr.append(od_r[i])
	   #		y_tr.append(0)
 	   rev1k, label1k = get_domain_revs(dom, 'test')
	   for r in rev1k:
		x_tr.append(r)
           for l in label1k:
		y_tr.append(l)
        y_tr = np_utils.to_categorical(np.array(y_tr).flatten())
        x_tr = np.array(x_tr).flatten()
        
        vocab = get_vocab(x_tr)
       
        print "Lenght of vocabulary:"
        print len(vocab)
        x_te, y_te = get_domain_revs(domain, 'train')
        x_te1k, y_te1k = get_domain_revs(domain, 'test')
        #y_te =  np_utils.to_categorical(y_te)
        y_te1k = np_utils.to_categorical(y_te1k) 
        word2idx = dict(zip(vocab.keys(), range(0, len(vocab))))
        word2idx['unk'] = len(word2idx) + 1
        word2idx['UNK'] = len(word2idx) + 1
        print len(word2idx)
        idx2word = {v: k for k, v in word2idx.iteritems()}   
        x_tr = np.array([[get_word2id(word, word2idx) for word in sent] for sent in x_tr])
        index_to_vector_map = get_index_to_embeddings_mapping(word2idx, corpus_wordvec)
        n_symbols = len(word2idx) + 1  # adding 1 to account for masking
        embedding_weights = np.zeros((n_symbols, 300))
        for word, index in word2idx.items():
            try:
                embedding_weights[index, :] = index_to_vector_map[word]
            except KeyError:
                embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, 300)


        x_te = np.array([[get_word2id(word, word2idx) for word in sent] for sent in x_te])
        x_te1k = np.array([[get_word2id(word, word2idx) for word in sent] for sent in x_te1k])
        print "length of training samples:" + str(len(x_tr))
        print "getting scores for domain:" + domain
        max_sent_len = int(np.max([len(sent) for sent in x_tr]))
        print "Maximum lenght of sentence in test data:" + str(max_sent_len)
        cv_dat = np.array(x_tr).flatten(), x_te, (np.array(y_tr)), y_te, x_te1k, y_te1k
        #cv_dat = np.array(x_tr).flatten(), x_te1k, (np.array(y_tr)), y_te1k
        #wrong_file = open(domain+'_mis.txt', 'w')
        acc, acc1k, misclassified, org_labels = tmlstm_model(embedding_weights, cv_dat, max_sent_len, model_weights_d, lda, dictionri, idx2word, 0.15, 256)
        #for i in range(0, len(misclassified)):
        #    wrong_file.write(str(misclassified[i]) + str(org_labels[i])+'\n')

        test_acc = test_acc + acc
        acc_dict[domain] = acc 
        acc_dict1k[domain] = acc1k
        #wrong_file.close()

	K.clear_session()

    print test_acc/4
    print len(set(words_not_found))
    print acc_dict
    print "Accuracies for 1k corpus:"
    print acc_dict1k
    return acc_dict, test_acc/4
if __name__ == '__main__':
    run_model()
