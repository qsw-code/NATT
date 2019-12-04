from data_manager import Data_Factory
from sample import Sampler
from model import NATT_0

min_rating = 1
max_df = 0.5
vocab_size = 8000
split_ratio = 0.9

datapath = 'programmableweb'

#Load document
data_factory = Data_Factory()
D = data_factory.preprocess(f'/Data/{datapath}/item_text.txt', f'/Data/{datapath}/item_tag.txt', max_df,vocab_size, split_ratio)

# Padding
X_train, X_test, maxlen_doc= data_factory.pad_sequence(D['X_train'], D['X_test'],0.8)

word_dim=300
# Read Glove word vectors
pretrain_w2v = f'/glove/glove.6B.{word_dim}d.txt'
if pretrain_w2v is None:
    init_WV = None
else:
    init_WV = data_factory.read_pretrained_word2vec(pretrain_w2v, D['X_vocab'], word_dim)

sampler = Sampler(X_train=X_train, Y_train=D['Y_train'], n_tags=len(D['Y_tag']), batch_size=128, n_negative=150)

natt = NATT_0(n_tags=len(D['Y_tag']), doc_len= maxlen_doc, init_WV=init_WV,  
                    vocab_size=len(D['X_vocab']) + 1, word_dim=word_dim, num_kernel=150, margin=0.1)
natt.train(sampler=sampler, X_test=X_test, Y_test=D['Y_test'], max_iter=10)