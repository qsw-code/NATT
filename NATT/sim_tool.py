'''
@author: Hao Wu, ShaoWei Qin
'''
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import pickle

class SimTool():

    def est_similarity(self, term_doc_matrix_train, term_doc_matrix_test, K):
        print("Create item coefficient matrix W_train/W_test & NN_train/NN_test...")
        W_train, NN_train = [], []
        n_items = term_doc_matrix_train.shape[0]
        item_ids = [idx for idx in range(n_items)]
        # initialize W_train with K-nearest-neighbors of items
        for i in tqdm(range(n_items), desc='Estimate similarity of items among train set...'):
            sims = np.squeeze(cosine_similarity(term_doc_matrix_train[i], term_doc_matrix_train))
            sim_i = sorted(zip(sims, item_ids), reverse=True)
            #k-nearest neighbors
            knn_sims, knn_items = zip(*sim_i[:K])
            W_train.extend(knn_sims)
            NN_train.append(knn_items)

        # transform shape of W_train
        W_train = np.array(W_train)
        W_train = W_train[:, np.newaxis].tolist()


        W_test, NN_test = [], []
        n_items = term_doc_matrix_test.shape[0]
        item_ids = [idx for idx in range(n_items)]
        # initialize W_train with K-nearest-neighbors of items.0.


        
        for i in tqdm(range(n_items), desc='Estimate similarity of items between test set and train set...'):
            sims = np.squeeze(cosine_similarity(term_doc_matrix_test[i], term_doc_matrix_train))
            sim_i = sorted(zip(sims, item_ids), reverse=True)
            #k-nearest neighbors
            knn_sims, knn_items = zip(*sim_i[:K])
            W_test.extend(knn_sims)
            NN_test.append(knn_items)

        # transform shape of W_train
        W_test = np.array(W_test)
        W_test = W_test[:, np.newaxis].tolist()

        C = {'W_train':W_train, 'NN_train':NN_train,
             'W_test':W_test, 'NN_test':NN_test}

        return C

    def load(self, path):
        C = pickle.load(open(path + "/similarity.all", "rb"))
        print ("Load similarity data - %s" % (path + "/similarity.all"))
        return C

    def save(self, path, C):
        print ("Saving similarity data - %s" % (path + "/similarity.all"))
        pickle.dump(C, open(path + "/similarity.all", "wb"))
        print ("Done!")