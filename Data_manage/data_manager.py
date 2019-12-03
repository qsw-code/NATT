'''
@author: Hao Wu, Yongxin Wang, Shaowei Qin
'''

import os
import pickle
import random
import sys
import numpy as np
from operator import itemgetter
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tflearn.data_utils import pad_sequences


class Data_Factory():

    def load(self, path):
        D = pickle.load(open(path + "/document.all", "rb"))
        print ("Load preprocessed document data - %s" % (path + "/document.all"))
        return D

    def save(self, path, D):
        print ("Saving preprocessed document data - %s" % (path + "/document.all"))
        pickle.dump(D, open(path + "/document.all", "wb"))
        print ("Done!")

    def read_pretrained_word2vec(self, path, vocab, dim):
        print(f'Read pre_trained word vectors from {path}')
        if os.path.isfile(path):
            raw_word2vec = open(path, 'r', encoding='UTF-8')
        else:
            print ("Path (word2vec) is wrong!")
            sys.exit()

        word2vec_dic = {}
        all_line = raw_word2vec.read().splitlines()
        mean = np.zeros(dim)
        count = 0
        for line in all_line:
            tmp = line.split()
            _word = tmp[0]
            _vec = np.array(tmp[1:], dtype=float)
            if _vec.shape[0] != dim:
                print ("Mismatch the dimension of pre-trained word vector with word embedding dimension!")
                sys.exit()
            word2vec_dic[_word] = _vec
            mean = mean + _vec
            count = count + 1
        mean = mean / count
        W = np.zeros((len(vocab) + 1, dim))
        count = 0
        for _word, i in vocab:
            if _word in word2vec_dic:
                W[i + 1] = word2vec_dic[_word]
                count = count + 1
            else:
                W[i + 1] = np.random.normal(mean, 0.1, size=dim)

        print ("%d of [%d] words exist in the given pretrained model" % (count, len(vocab)))
        return W
    
    def preprocess(self, path_itemtext, path_itemtag, _max_df=0.5, _vocab_size=8000, ratio=0.9):
        '''
        Preprocess rating and document data.
        Input:
            - path_itemtext: path for textual data of items(data format - item_id::sentence,sentence....)
            - path_itemtag: path for tagging data of items(data format - item_id::tag,tag,tag....)
            - _max_df: terms will be ignored that have a document frequency higher than the given threshold (default = 0.5)
            - vocab_size: vocabulary size (default = 8000)
            - ratio: (1-ratio),  will be training and test set, respectively
        Output:
            - D['X_train'],D['X_test']: list of sequence of word index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - D['Y_train'],D['Y_test']: list of sequence of tag index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - D['X_vocab'],D['Y_tag']: list of tuple (word|tag, index) in the given corpus
        '''
        # Validate data paths
        if os.path.isfile(path_itemtext):
            raw_content = open(path_itemtext, 'r',encoding='utf-8')
            print ("Path - document data: %s" % path_itemtext)
        else:
            print ("Path(item text) is wrong!")
            sys.exit()

        # 1st scan document file to filter items which have documents
        tmp_iid_plot1 = set()
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            i = tmp[0]
            tmp_plot = tmp[1].split(' ')
            if tmp_plot[0] == '':
                continue
            tmp_iid_plot1.add(i)
        raw_content.close()
        
        # Validate data paths
        if os.path.isfile(path_itemtag):
            raw_content = open(path_itemtag, 'r',encoding='utf-8')
            print ("Path - document data: %s" % path_itemtext)
        else:
            print ("Path(item text) is wrong!")
            sys.exit()

        # 2nd scan document file to filter items which have tags
        tmp_iid_plot2 = set()
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            i = tmp[0]
            tmp_plot = tmp[1].split(',')
            if tmp_plot[0] == '':
                continue
            tmp_iid_plot2.add(i)
        raw_content.close()
        
        tmp_iid_plot = tmp_iid_plot1 & tmp_iid_plot2
        
        # 3nd scan document file to make idx2plot dictionary according to
        # indices of items in rating matrix
        print ("Preprocessing item document...")

        # Read Document File
        raw_content = open(path_itemtext, 'r',encoding='utf-8')
        map_item2txt = {}  # temporal item_id to text
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if tmp[0] in tmp_iid_plot:
                map_item2txt[tmp[0]] = tmp[1]
        raw_content.close()
        
        # 
        raw_content = open(path_itemtag, 'r',encoding='utf-8')
        map_item2tag = {}  # temporal item_id to tag
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if tmp[0] in tmp_iid_plot:
                map_item2tag[tmp[0]] = tmp[1]
        raw_content.close()

        print ("\tRemoving stop words...")
        print ("\tFiltering words by TF-IDF score with max_df: %.1f, vocab_size: %d" % (_max_df, _vocab_size))

        # Make vocabulary by document
        corpus = [value for value in map_item2txt.values()]
        vectorizer1 = TfidfVectorizer(max_df=_max_df, stop_words='english', max_features=_vocab_size)
        tfidf_matrix = vectorizer1.fit_transform(corpus)
        X_vocab = vectorizer1.vocabulary_
        idf_weight=vectorizer1.idf_
        
        # Make train/test data for run
        X_train, X_test = [], []  # #for item text
        # X_train_raw, X_test_raw = [], []
        Y_train, Y_test = [], []  # #for item tag
        
        np.random.seed(2019)
        index = 0
        train_index = []
        test_index = []
        for item in map_item2txt.keys():
            wordid_list = [X_vocab[word] + 1 for word in map_item2txt[item].split() if word in X_vocab]
            
            toss = np.random.uniform()
            if toss <= ratio:
                train_index.append(index)
                X_train.append(wordid_list)
                Y_train.append(map_item2tag[item])
            else:
                test_index.append(index)
                X_test.append(wordid_list)
                Y_test.append(map_item2tag[item])
            index = index + 1
        X_train_wv, X_test_wv = tfidf_matrix[train_index], tfidf_matrix[test_index]
        
        # process data in training set
        vectorizer2 = CountVectorizer()
        tf_matrix=vectorizer2.fit_transform(Y_train)
        Y_tag = vectorizer2.vocabulary_
        # assign identifier for tags 
        for i in range(len(Y_train)):
            Y_train[i] = [Y_tag[tag] for tag in Y_train[i].split(',') if tag in Y_tag]
        for i in range(len(Y_test)):
            Y_test[i] = [Y_tag[tag] for tag in Y_test[i].split(',') if tag in Y_tag]

        X_vocab = sorted(X_vocab.items(), key=itemgetter(1))
        Y_tag = sorted(Y_tag.items(), key=itemgetter(1))
        
        D = {'X_train': X_train, 'X_test': X_test, 'X_vocab': X_vocab,
             'Y_train': Y_train, 'Y_test': Y_test, 'Y_tag': Y_tag,
             'X_train_wv':X_train_wv, 'X_test_wv':X_test_wv, 'X_train_tv':tf_matrix,
             'train_index':train_index,'test_index':test_index
             }
            
        print ("Done!")
        return D
    
    def pad_sequence(self, X_train, X_test, threshold=0.95):
        '''
        threshold: 0.95 means that maxlen_doc we taken covers 95% percentage of documents
        '''
        len_doc = [len(profile) for profile in X_train]
        len_doc.extend([len(profile) for profile in X_test])
        len_doc = sorted(len_doc)
        maxlen_doc = len_doc[(int)(len(len_doc) * threshold)]
        print ("X_train, X_test, maxlen_doc:%d " % (maxlen_doc))
        X_train = pad_sequences(X_train, maxlen=maxlen_doc)
        X_test = pad_sequences(X_test, maxlen=maxlen_doc)
        return X_train, X_test, maxlen_doc
    
    def generate_tag_item_matrix(self, D):
        n_items = len(D['Y_train'])
        n_tags = len(D['Y_tag'])
        A = dok_matrix((n_tags, n_items), dtype=np.float32)
                 
        for item_id in range(n_items):
            for tag_id in D['Y_train'][item_id]:
                A[tag_id, item_id] = 1.0
        return A

     