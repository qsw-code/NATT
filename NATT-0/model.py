'''
@author: Hao Wu, Shaowei Qin
'''
import os, time
import tflearn
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from eval_metrics import ID2ID, AP, NDCG, PrecisionRecall, val_format,HD
from text_encoder import text_encode

TOP_N = [5,10]

                        
class NATT_0():
    
    def __init__(self,res_dir, n_tags, doc_len, init_WV=None, vocab_size=8000, word_dim=300, 
        num_kernel=150, margin=0.1,learning_rate=0.001, random_seed=2019):
        self.doc_len=doc_len
        self.vocab_size = vocab_size  # the number of distinct words 
        self.word_dim = word_dim  # dimension of word embeddings
        self.tag_dim = word_dim  # dimension of tag embeddings
        self.num_kernel = num_kernel # number of convolutional kernels per window
        self.n_tags = n_tags  
        self.learning_rate = learning_rate
        self.margin = margin
        self.random_seed = random_seed
        self.init_WV = init_WV
        self._init_graph()
    
    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            
            # placeholder for inputs
            self.doc = tf.placeholder(tf.int32, [None, self.doc_len])
            self.pos_tag = tf.placeholder(tf.int32, [None, 1])
            self.neg_tags = tf.placeholder(tf.int32, [None, None])

            # embeddings of words
            if init_WV is None:
                self.word_embeddings = tflearn.embedding(self.doc, input_dim=self.vocab_size, output_dim=self.word_dim, trainable=True, name="EmbeddingLayer")
            else:
                self.word_embeddings = tflearn.embedding(self.doc, input_dim=self.vocab_size, output_dim=self.word_dim, trainable=False, name="EmbeddingLayer")
                 # Retrieve embedding layer weights (only a single weight matrix, so index is 0)
                embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
                # Assign your own weights (for example, a numpy array [input_dim, output_dim])
                tf.assign(embeddingWeights, init_WV)
            
            # embeddings of tags
            self.tag_embeddings = tf.Variable(tf.random_normal([self.n_tags, self.tag_dim], stddev=0.1))
            
            doc_encode=text_encode(self.word_embeddings,self.num_kernel)


            self.doc_emb = doc_encode
        
            # positive tag embedding
            pos_tag_emb = tf.nn.embedding_lookup(self.tag_embeddings, self.pos_tag)#batch_size*1*tag_dim
            pos_tag_emb = tf.squeeze(pos_tag_emb)#batch_size*tag_dim
            
            # inner production of document to positive tag
            self.pos_production = tf.reduce_sum(tf.multiply(self.doc_emb, pos_tag_emb),1)
    
            # negative tag embeddings
            neg_tag_embs = tf.nn.embedding_lookup(self.tag_embeddings, self.neg_tags)
            neg_tag_embs = tf.transpose(neg_tag_embs, (0, 2, 1))
            # inner production of document to negative tags
            production_to_neg_items = tf.reduce_sum(tf.multiply(tf.expand_dims(self.doc_emb, -1), neg_tag_embs), 1)
    
            # best negative item
            self.max_neg_production = tf.reduce_max(production_to_neg_items, 1)
    
            # compute hinge loss per-pair
            loss_per_pair = tf.maximum(self.margin-self.pos_production+ self.max_neg_production, 0)
            
            # loss
            self.loss = tf.reduce_sum(loss_per_pair)
            
            # optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_opt = optimizer.minimize(self.loss)
          
            '''
            for evaluation part
            '''
            self.test_doc = tf.expand_dims(self.doc_emb, 1)
            self.test_tags = tf.expand_dims(self.tag_embeddings, 0)
            self.prod=tf.multiply(self.test_doc, self.test_tags)
            self.tag_scores = tf.reduce_sum(self.prod, 2)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()
            
    def train(self, sampler, X_test, Y_test, max_iter):

        for iteration in range(max_iter):

            print ("%d iteration ..." % (iteration))

            
            total_loss = 0
            tic = time.time()
            sampler.generate_batches()
            
            while not sampler.is_empty():
                batch_doc, batch_pos_tag, batch_neg_tags = sampler.next_batch()
                _, batch_loss = self.sess.run((self.train_opt, self.loss), {self.doc: batch_doc,
                                                                            self.pos_tag: batch_pos_tag,
                                                                            self.neg_tags:batch_neg_tags})
                total_loss += batch_loss
                
            toc = time.time()
            elapsed = toc - tic
            
            print("Loss: %.5f Elpased: %.4fs \n" % (total_loss, elapsed))
            
            _precision, _recall, _ndcg, _hd = self.evaluate(X_test, Y_test, TOP_N)
            print("Top-%s, Precision: %s Recall: %s NDCG: %s  HD: %s\n" % (TOP_N, val_format(_precision, ".5"), val_format(_recall, ".5"), val_format(_ndcg, ".5"),val_format(_hd, ".5")))

    def evaluate(self, X_test, Y_test, TOP_N):
        _precision = [0] * len(TOP_N)
        _recall = [0] * len(TOP_N)
        _hd = [0] * len(TOP_N)
        _ndcg = [0] * len(TOP_N)
        hd = []
        count = 0
        tag_ids = [k for k in range(self.n_tags)]
        
        for i in tqdm(range(len(Y_test)), desc='Estimate...'):
            # the case of having no warm-start tags, also, there is no ground_truth
            if len(Y_test[i]) == 0: continue
            
            t1 = time.time()
            # recommend tags for current item         
            tag_score = self.sess.run((self.tag_scores), {self.doc: np.array([X_test[i]])})
            tag_score = np.squeeze(tag_score)
            

            
            # build TOP_N ranking list         
            ranklist = sorted(zip(tag_score, tag_ids), reverse=True)

            sc, hd_id = zip(*ranklist)
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

        return _precision, _recall, _ndcg, _hd


