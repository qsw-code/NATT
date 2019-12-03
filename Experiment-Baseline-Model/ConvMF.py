'''
@author: Hao Wu,ShaoWei Qin
'''
import os, time
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tqdm import tqdm
from data_manager import Data_Factory
from eval_metrics import ID2ID, AP, NDCG, PrecisionRecall, val_format,HD
TOP_N = [5, 10]
a = 1
b = 0.01


class CNN():
    '''
    classdocs
    '''
    batch_size = 128
    epochs = 1
    vanila_dimension = 200
    
    def __init__(self, doc_len, vocab_size, word_dim, out_dim, num_kernel_per_ws, dropout_rate, init_WV=None):

        '''Input Layer'''
        cnn = input_data(shape=[None, doc_len], name='input')
        
        '''Embedding Layer'''
        if init_WV is None:
            cnn = tflearn.embedding(cnn, input_dim=vocab_size, output_dim=word_dim, trainable=True, name="EmbeddingLayer")
        else:
            cnn = tflearn.embedding(cnn, input_dim=vocab_size, output_dim=word_dim, trainable=False, name="EmbeddingLayer")
        
        '''Convolution Layer & Max Pooling Layer'''
        branch1 = conv_1d(cnn, num_kernel_per_ws, 3, padding='valid', activation='relu')  # window size (ws=3)
        branch2 = conv_1d(cnn, num_kernel_per_ws, 4, padding='valid', activation='relu')  # window size (ws=4)
        branch3 = conv_1d(cnn, num_kernel_per_ws, 5, padding='valid', activation='relu')  # window size (ws=5)
        cnn = merge([branch1, branch2, branch3], mode='concat', axis=1)
        cnn = tf.expand_dims(cnn, 2)
        cnn = global_max_pool(cnn)
        
        '''fully connected layer'''
        cnn = fully_connected(cnn, self.vanila_dimension, activation='tanh')
        
        '''Dropout Layer'''
        cnn = dropout(cnn, dropout_rate)
        
        '''Projection Layer'''
        cnn = fully_connected(cnn, out_dim, activation='tanh')
        
        # model
        cnn = regression(cnn, optimizer='adam', loss='mean_square')
        
        self.model = tflearn.DNN(cnn, tensorboard_verbose=0)
        
        if init_WV is not None:
            # Retrieve embedding layer weights (only a single weight matrix, so index is 0)
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
            # Assign your own weights (for example, a numpy array [input_dim, output_dim])
            self.model.set_weights(embeddingWeights, init_WV)
            
    def load_model(self, model_path):
        self.model.load(model_path)

    def save_model(self, model_path, overwrite=True):
        self.model.save(model_path, overwrite)
        
    def train(self, X, V, weight_of_sample, seed):
        print("Train CNN...")
        self.model.fit(X, V, n_epoch=self.epochs, shuffle=True, show_metric=True, batch_size=self.batch_size)

    def get_projection_layer(self, X):
        theta = []
        for i in range(int(len(X) / self.batch_size) + 1):
            X_i = None
            if (i + 1) * self.batch_size < len(X):
                X_i = X[i * self.batch_size: (i + 1) * self.batch_size]
            else:  # for the last mini_batch where the size is less than self.batch_size
                X_i = X[i * self.batch_size:]
            theta.extend(self.model.predict(X_i))
        return np.array(theta)
    

class ConvMF():            
    
    def __init__(self, datapath,res_dir, doc_len=100, vocab_size=8000, word_dim=200, out_dim=50, num_kernel_per_ws=50, dropout_rate=0.2, init_WV=None):
        '''
        doc_len: max length of input sentences of item documents
        vocab_size: total number of words in the dataset of item documents
        word_dim: embedding size for words in CNN-based similarity network
        out_dim: dimensionality of output vector of CNN 
        num_kernel_per_ws: number of kernels per window size
        dropout_rate:dropout rate for CNN-based similarity network
        init_WV: a flag to determine whether or not initialize the word embeddings using Glove word embeddings which can be downloaded from Internet
        '''
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        self.log_path = res_dir + '/convmf.log'
        log = open(self.log_path, 'a')
        log.write(f"### CNN ###\n\n data: {datapath}\n")
        log.write("word_dim=%d, out_dim=%d, num_kernel_per_ws=%d, dropout_rate=%.2f\n\n" % (word_dim, out_dim, num_kernel_per_ws, dropout_rate))
        log.close()
        
        np.random.seed(2019)
        # a CNN module for extraction deep features of items' textual content   
        self.cnn_module = CNN(doc_len, vocab_size, word_dim, out_dim, num_kernel_per_ws, dropout_rate, init_WV) 
        self.out_dim = out_dim
    
    def train(self, X_train, Y_train, X_test, Y_test, n_tags, lambda_u=1, lambda_v=100, max_iter=50, give_weight=False):
        '''
        Input:
        res_dir: path for storing the datasets
        X_train: the data used for CNN-based similarity network
        Y_train: the data used for building item-tag matrix
        lambda_u: parameter for L2 regularization of tags
        lambda_v: parameter for L2 regularization of items
        '''
        
        log = open(self.log_path, 'a')
        log.write("### ConvMF###\n\n")
        log.write("lambda_u=%f, lambda_v=%f\n" % (lambda_u, lambda_v))
        log.close()
        '''re-create rating data between items and tags'''
        self.n_tags = n_tags
        self.n_items = len(Y_train)
        print('#items:%d #tags:%d in training!' % (self.n_items, self.n_tags))
        
        # create train_item
        tag_id_list = [[] for i in range(self.n_items)]
        rating_list1 = [[] for i in range(self.n_items)]
        
        for i in range(self.n_items):
            # for each item-tag interaction, we take a binary value 0 or 1
            rating_list1[i] = [1.0] * len(Y_train[i])
            tag_id_list[i] = Y_train[i]
        
        # create train_tag
        item_id_list = [[] for i in range(self.n_tags)]
        rating_list2 = [[] for i in range(self.n_tags)]
        
        for i in range(self.n_items):
            for j in range(len(tag_id_list[i])):
                item_id_list[tag_id_list[i][j]].append(i)
                rating_list2[tag_id_list[i][j]].append(rating_list1[i][j])
        
        train_item = [tag_id_list, rating_list1]
        train_tag = [item_id_list, rating_list2]
        
        # initialization
        if give_weight is True:
            item_weight = np.array([math.sqrt(len(i))  for i in train_item[1]], dtype=float)
            item_weight = (float(self.n_items) / item_weight.sum()) * item_weight
            
            tag_weight = np.array([math.sqrt(len(u))  for u in train_tag[1]], dtype=float)
            tag_weight = (float(self.n_tags) / tag_weight.sum()) * tag_weight
            
        else:
            item_weight = np.ones(self.n_items, dtype=float)
            tag_weight = np.ones(self.n_tags, dtype=float)
            
        # transform X_train
        X_train = np.array(X_train)
        # projection vectors of items' textual contents
        theta = self.cnn_module.get_projection_layer(X_train)
        # latent vectors of tags
        U = np.random.uniform(size=(self.n_tags, self.out_dim))
        # latent vectors of items
        V = theta
        # some variables
        count = 0
        PREV_LOSS = 1e-50
        
        # iterations
        for iteration in range(max_iter):
            loss = 0
            tic = time.time()
            print ("%d iteration..." % (iteration))
    
            VV = b * (V.T.dot(V))
            sub_loss = np.zeros(self.n_tags)
            # update U
            for i in range(self.n_tags):
                idx_item = train_tag[0][i]
                V_i = V[idx_item]
                R_i = train_tag[1][i]
                
                tmp_A = VV + (a - b) * (V_i.T.dot(V_i))
                A = tmp_A + lambda_u * tag_weight[i] * np.eye(self.out_dim)
                B = (a * V_i * (np.tile(R_i, (self.out_dim, 1)).T)).sum(0) 
                U[i] = np.linalg.solve(A, B)
                # -\frac{\lambda_u}{2}\sum_i u_i^Tu_i
                sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])
                    
            loss += np.sum(sub_loss)
    
            sub_loss_dev = np.zeros(self.n_items)
            sub_loss = np.zeros(self.n_items)
            # update V
            UU = b * (U.T.dot(U))
            for j in range(self.n_items):
                idx_user = train_item[0][j]
                U_j = U[idx_user]
                R_j = train_item[1][j]
    
                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + lambda_v * item_weight[j] * np.eye(self.out_dim)
                B = (a * U_j * (np.tile(R_j, (self.out_dim, 1)).T)).sum(0)
                B = B + lambda_v * item_weight[j] * theta[j]
                   
                V[j] = np.linalg.solve(A, B)
                # -\sum_i\sum_j\frac{c_{i,j}}{2}(r_{ij}-u_i^T v_j)^2
                sub_loss_dev[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss_dev[j] += a * np.sum((U_j.dot(V[j])) * R_j)
                sub_loss_dev[j] += -0.5 * np.dot(V[j].dot(tmp_A), V[j])
                    
            loss += np.sum(sub_loss_dev)
            loss += np.sum(sub_loss)
            
            self.cnn_module.train(X_train, V, item_weight, np.random.randint(100000))
            theta = self.cnn_module.get_projection_layer(X_train)
            # -\frac{\lambda_v}{2}\sum_j(v_j-\theta_j)^T(v_j-\theta_j)
            # cnn_loss = history.history['loss'][-1]
            # loss += -0.5 * lambda_v * cnn_loss * self.n_items
            
            toc = time.time()
            elapsed = toc - tic
    
            if iteration == 0:
                converge = -1
            else: 
                converge = abs((loss - PREV_LOSS) / PREV_LOSS)
    
            print("Loss: %.5f Elpased: %.4fs Converge: %.6f\n" % (loss, elapsed, converge))
            if iteration!=0:
                log = open(self.log_path, 'a')
                log.write("%d iteration..." % (iteration))
                _precision, _recall, _ndcg, _hd = self.evaluate(U, X_test, Y_test, TOP_N)
                print ("Top-%s, Precision: %s Recall: %s NDCG: %s HD: %s\n" % (TOP_N, val_format(_precision, ".5"), val_format(_recall, ".5"), val_format(_ndcg, ".5"), val_format(_hd, ".5")))
                log.write("Top-%s, Precision: %s Recall: %s NDCG: %s HD: %s\n\n" % (TOP_N, val_format(_precision, ".5"), val_format(_recall, ".5"), val_format(_ndcg, ".5"), val_format(_hd, ".5")))
            
            if iteration > 5 and converge < 1e-4:
                break
            
            PREV_LOSS = loss
        log.close()
    
    def evaluate(self, U, X_test, Y_test, TOP_N):
        _precision = [0] * len(TOP_N)
        _recall = [0] * len(TOP_N)
        _ndcg = [0] * len(TOP_N)
        _hd=[0] * len(TOP_N)
        hd=[]

        
        count = 0
        tag_ids = [k for k in range(self.n_tags)]
        tag_score = [0] * self.n_tags
        
        for i in tqdm(range(len(Y_test)), desc='Estimate...'):
            # the case of having no warm-start tags, also, there is no ground_truth
            if len(Y_test[i]) == 0: continue
            
            t1 = time.time()
            # get projection vectors for cold-start items
            theta_i = self.cnn_module.get_projection_layer(np.array([X_test[i]]))
            # print(theta_i[0])
            # recommend tags for current item
            for j in range(self.n_tags):
                tag_score[j] = theta_i[0].T.dot(U[j])
           
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
        
        return _precision, _recall, _ndcg, _hd
    
    
if __name__ == '__main__':
   
    min_rating = 1
    max_df = 0.5
    vocab_size = 10000
    split_ratio = 0.9
    datapath = 'programmableweb'

    # Load document
    data_factory = Data_Factory()
    D = data_factory.preprocess(f'/data/{datapath}/item_text.txt', f'/data/{datapath}/item_tag.txt', max_df,vocab_size, split_ratio)
    
    # Read Glove word vectors
    word_dim = 300
    pretrain_w2v = f'../glove/glove.6B.{word_dim}d.txt'
    if pretrain_w2v is None:
        init_WV = None
    else:
        init_WV = data_factory.read_pretrained_word2vec(pretrain_w2v, D['X_vocab'], word_dim)
    
    # Padding
    X_train, X_test, maxlen_doc= data_factory.pad_sequence(D['X_train'], D['X_test'], 0.8)
   
    # ConvMF
    lambda_u = 10
    lambda_v = 0.1
    convmf = ConvMF(datapath=datapath,res_dir='log', doc_len=maxlen_doc, vocab_size=len(D['X_vocab']) + 1, word_dim=word_dim, out_dim=200, num_kernel_per_ws=128, dropout_rate=0.2, init_WV=init_WV)
    convmf.train(X_train=X_train, Y_train=D['Y_train'], X_test=X_test, Y_test=D['Y_test'], n_tags=len(D['Y_tag']), lambda_u=lambda_u, lambda_v=lambda_v, max_iter=32)
