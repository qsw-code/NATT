from sample import Sampler
from data_manager import Data_Factory
from sim_tool import SimTool
from model import NATT


min_rating = 1
max_df = 0.5
vocab_size = 8000
split_ratio = 0.9
k=10
datapath = 'programmableweb'

# Load document
data_factory = Data_Factory()
D = data_factory.preprocess(f'/Data/{datapath}/item_text.txt', f'/Data/{datapath}/item_tag.txt', max_df,vocab_size, split_ratio)

# Load K-Nearnest Neighboor
sim_tool = SimTool()
#C = sim_tool.est_similarity(D['X_train_wv'],D['X_test_wv'], k)
C = sim_tool.load(f'../data/{datapath}/wv')
# Padding
X_train, X_test, maxlen_doc= data_factory.pad_sequence(D['X_train'], D['X_test'], 0.8)

# Read Glove word vectors+
word_dim=300
pretrain_w2v = f'../glove/glove.6B.{word_dim}d.txt'
if pretrain_w2v is None:
    init_WV = None
else:
    init_WV = data_factory.read_pretrained_word2vec(pretrain_w2v, D['X_vocab'], word_dim)

# Sampling
sampler = Sampler(X_train=X_train, NN_train=C['NN_train'],Y_train=D['Y_train'], n_tags=len(D['Y_tag']), batch_size=128, K=k+1,n_negative=50)

# NATT-model
natt = NATT(datapath=datapath,n_tags=len(D['Y_tag']), doc_len= maxlen_doc, K=k+1,init_WV=init_WV,vocab_size=len(D['X_vocab']) + 1,
            word_dim=word_dim,num_kernel=150, margin=0.1)
# train
natt.train(sampler=sampler, NN_test=C['NN_test'],X_test=X_test, Y_test=D['Y_test'], max_iter=10)