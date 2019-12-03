'''
@author: Hao Wu, ShaoWei Qin
'''

import os
import time, datetime
import math
import numpy as np
from scipy.sparse import dok_matrix
from tqdm import tqdm
from data_manager import Data_Factory
from eval_metrics import AP, NDCG, PrecisionRecall, val_format,HD
from sklearn.metrics.pairwise import cosine_similarity


class ItemK():               
       
    def evaluate(self, A, X_train_wv, X_test_wv, Y_test, K=100, TOP_N=[5, 10]):
        '''
        Input:
            - A: the tag-item matrix
            - X_train_wv/X_test_wv: the term vector (by tfidf or tf) for each document in train/test set
            - Y_test: list of sequence of tag index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - K: the number of K-nearest-neighbors
        '''
            
        _precision = [0] * len(TOP_N)
        _recall = [0] * len(TOP_N)
        _ndcg = [0]* len(TOP_N)
        _hd = [0]* len(TOP_N)
        hd=[]
        n_tags = A.shape[0]
        n_items = A.shape[1]
        
        count = 0
        tag_ids = [idx for idx in range(n_tags)]
        tag_score = [0] * n_tags
        
        train_item_ids = [idx for idx in range(n_items)]
        # recommend tags for current item
        for i in tqdm(range(len(Y_test)), desc='Estimate...'):
            # the case of having no warm-start tags, also, there is no ground_truth
            if len(Y_test[i]) == 0: continue
            
            sims = np.squeeze(cosine_similarity(X_test_wv[i], X_train_wv))
            sim_i = sorted(zip(sims, train_item_ids), reverse=True)[:K]
            knn_sims, knn_items = zip(*sim_i)
            
            for j in range(n_tags):
                tag_score[j] = A[j, knn_items].dot(knn_sims)

            
            # build TOP_N ranking list         
            ranklist = sorted(zip(tag_score, tag_ids), reverse=True)
            
            sc,hd_id=zip(*ranklist)
            hd.append(hd_id)

            for n in range(len(TOP_N)):
                sublist = ranklist[:TOP_N[n]]
                score, test_decision = zip(*sublist)
                p_at_k, r_at_k = PrecisionRecall(Y_test[i], test_decision)
                # sum the metrics of decision
                _precision[n] += p_at_k
                _recall[n] += r_at_k

                _ndcg[n] += NDCG(Y_test[i], test_decision)
            count += 1
            
        # calculate the final scores of metrics      
        for n in range(len(TOP_N)):
            _precision[n] /= count
            _recall[n] /= count
            _ndcg[n] /= count
            cc = 0
            hd_sum = 0
            for i in range(len(hd)):
                for j in range(i + 1, len(hd)):
                    p = HD(hd[i], hd[j], TOP_N[n])
                    hd_sum += p
                    cc += 1
            _hd[n] = hd_sum / cc
        
        print ("[%s] Top-%s, K: %d, Precision: %s Recall: %s NDCG: %s HD: %s\n" % (datetime.datetime.now(), TOP_N, K, val_format(_precision, ".5"), val_format(_recall, ".5"), val_format(_ndcg, ".5"), val_format(_hd, ".5")))



if __name__ == '__main__':

    datapath = 'programmableweb'
    
    data_factory = Data_Factory()
    D = data_factory.preprocess(f'/data/{datapath}/item_text.txt', f'/data/{datapath}/item_tag.txt')
    A = data_factory.generate_tag_item_matrix(D)
    # baseline
    k=50
    itemknn = ItemK()
    itemknn.evaluate(A, D['X_train_wv'], D['X_test_wv'], D['Y_test'], k)
  
