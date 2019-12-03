'''
@author: Hao Wu, ShaoWei Qin
'''
from scipy.sparse import dok_matrix, lil_matrix
import numpy as np
from queue import Queue

class Sampler():

    def __init__(self, X_train, NN_train,Y_train, n_tags, batch_size, n_negative, K,check_negative=True):
        """
    
        :param X_train, Y_train: the training data
        :param batch_size: number of samples to return
        :param n_negative: number of negative samples per item-positive-tag pair
        :param result_queue: the output queue
        :return: None
        """
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.check_negative = check_negative
        self.X_train = X_train
        self.NN_train = NN_train
        self.K = K
        self.Y_train=Y_train
        
        n_items = len(Y_train)
        self.item_tag_matrix = dok_matrix((n_items, n_tags), dtype=np.float32)
        for i in range(n_items):
            for j in Y_train[i]:
                self.item_tag_matrix[i, j] = 1.0
                
        self.result_queue = Queue()
        self.item_tag_matrix = lil_matrix(self.item_tag_matrix)
        self.item_tag_pairs = np.asarray(self.item_tag_matrix.nonzero()).T
        self.item_to_positive_set = {i: set(row) for i, row in enumerate(self.item_tag_matrix.rows)}
    
    def generate_batches(self): 
        np.random.shuffle(self.item_tag_pairs)
        for i in range(int(len(self.item_tag_pairs) / self.batch_size) + 1):
            item_positive_tag_pairs = None
            if (i + 1) * self.batch_size < len(self.item_tag_pairs):
                item_positive_tag_pairs = self.item_tag_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]
            else:  # for the last mini_batch where the size is less than self.batch_size
                item_positive_tag_pairs = self.item_tag_pairs[i * self.batch_size:, :]
            
            item_to_doc = []
            pos_tag = []
            neg_tags=[]
            for (i, j) in item_positive_tag_pairs:
                pos_id=[]
                pos_id.extend(self.Y_train[i])
                tmp = self.X_train[self.NN_train[i][0]]
                for k in range(1, self.K):
                    nn_id=self.NN_train[i][k]
                    tmp = np.concatenate((tmp, self.X_train[nn_id]))
                    pos_id.extend(self.Y_train[nn_id])


                neg_tmp = np.random.randint(0, self.item_tag_matrix.shape[1], size=(self.n_negative))
                for k in range(len(neg_tmp)):
                    while neg_tmp[k] in pos_id:
                        neg_tmp[k]=np.random.randint(0, self.item_tag_matrix.shape[1])

                neg_tags.append(neg_tmp)
                item_to_doc.append(tmp)
                pos_tag.append([j])
            self.result_queue.put((item_to_doc, pos_tag, neg_tags)) 


    def is_empty(self):
        return self.result_queue.empty()
    
    def next_batch(self):
        return self.result_queue.get()