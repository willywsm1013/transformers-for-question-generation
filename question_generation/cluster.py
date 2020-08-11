import sys
import numpy as np
import os
import re
import json
import logging
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import  PCA
from sklearn.cluster import  KMeans, DBSCAN

def load_universal_sentence_encoder():
    logger.info('load universal-sentence-encoder')
    return hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

def load_inferSent(sentences):
    logger.info('load InferSent')
    V = 2
    MODEL_PATH = 'Infersent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    if torch.cuda.is_available():
        infersent.cuda()
    
    # set word vector
    if V == 1:
        W2V_PATH = 'Infersent/Glove/glove.840B.300d.txt'
        logger.warning('Use Glove Embedding')
    elif V ==2 :
        W2V_PATH = 'Infersen/fastText/crawl-300d-2M.vec'
        logger.warning('Use fastText Embedding')
    else:
        raise NotImplementedError
    infersent.set_w2v_path(W2V_PATH)
    
    # build voceb
    infersent.build_vocab(sentences, tokenize=True)

    return infersent

def post_process(elem, cluster_num, pred_cluster, name, p_a, p_qa, indices=None, sort=False) :
    if indices is None:
        indices = list(range(len(pred_cluter)))

    cluster = [[] for _ in range(cluster_num)]
    for c, i in zip(pred_cluster, indices):
        cluster[c].append(i)

    if sort:
        sort_by_a = []
        sort_by_qa = []
        for idx in cluster:
            # sort by p(a|c,q)
            idx = sorted(idx, key=lambda x:p_a[x], reverse=True)
            sort_by_a.append(idx[:])

            # sort by p(a,q|c)=p(a|q,c)*p(q|c)
            idx = sorted(idx, key=lambda x:p_qa[x], reverse=True)
            sort_by_qa.append(idx[:])

        elem['%s_sort_by_a_idx'%name] = sort_by_a
        elem['%s_sort_by_qa_idx'%name] = sort_by_qa
    else:
        elem[name] = cluster

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# read data
logger.info('read data')
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

# load model
model_name=sys.argv[3]
if model_name == 'use':
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text

    model = load_universal_sentence_encoder()
    encode = lambda x:model.signatures['question_encoder'](tf.constant(x))['outputs'].numpy()
elif model_name == 'infersent':
    import torch
    from models import  InferSent

    model = load_inferSent([q for elem in data for q in elem['pred_question']])
    encode = lambda x:model.encode(x, tokenize=True)
else:
    raise Exception('Unknown model_name %s'%model_name)

# encode and clustering
def cluster(elem, embed, filtered_idx, sort=False):
    if sort:
        qa_prob = np.array(elem['qa_prob'])
        answer_rank = np.array(elem['answer_rank'])

        cq_log_prob = np.array(elem['lm_log_prob'])
        cq_token_num = np.array(elem['lm_token_num'])
        cq_lm_prob = np.exp(cq_log_prob/cq_token_num)
        p_qa = qa_prob*cq_lm_prob
    else:
        qa_prob = None
        p_qa = None

    for cluster_num in [2,3]:
        if len(embed) < cluster_num:
            pred_cluster = [i for i in range(len(embed))]
        else:
            pred_cluster = KMeans(n_clusters=cluster_num).fit_predict(embed)
        post_process(elem, cluster_num, pred_cluster, 
                     'cluster_by_%s-%d'%(model_name, cluster_num), qa_prob, p_qa, filtered_idx, sort=sort)

accu_num=5
elems = []
sentences = []
elem_idx = []
all_filtered_idx = []
for i, elem in enumerate(tqdm(data, ncols=50)):
    elems.append(elem)
    questions = elem['pred_question']
    # filtering 
    filtered_idx = list(range(len(questions)))
    all_filtered_idx.append(filtered_idx)
    sentences.extend([questions[i] for i in filtered_idx])
    elem_idx.extend([len(elems)-1]*len(filtered_idx))

    if len(elems) == accu_num or i == len(data)-1:
        embed = encode(sentences)
        embeds = [[] for _ in range(len(elems))]
        for elem_id, embd in zip(elem_idx, embed) :
            embeds[elem_id].append(embd)
        
        for elem, embed, f_idx in zip(elems, embeds, all_filtered_idx):
            cluster(elem, embed, f_idx, sort=True)
        elems = []
        sentences = []
        elem_idx = []
        all_filtered_idx = []

with open(sys.argv[2], 'w') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
