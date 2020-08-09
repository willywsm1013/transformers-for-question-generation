import sys
import numpy as np
import os
import re
import json
import logging
from models import  InferSent
from tqdm import tqdm
import torch

# setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# load model
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
if torch.cuda.is_available():
    infersent.cuda()

# set word vector
if V == 1:
    W2V_PATH = 'Glove/glove.840B.300d.txt'
    logger.info('Use Glove Embedding')
elif V ==2 :
    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    logger.info('Use fastText Embedding')
else:
    raise NotImplementedError
infersent.set_w2v_path(W2V_PATH)

# read data
refs = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        refs.append(line[:-1])

hyps = []
with open(sys.argv[2], 'r') as f:
    for line in f:
        hyps.append(line[:-1])

# build voceb
infersent.build_vocab(refs+hyps, tokenize=True)

# get embeddings
refs_embeds = infersent.encode(refs, tokenize=True)
hyps_embeds = infersent.encode(hyps, tokenize=True)

# compute cosine similarity
refs_norm = np.linalg.norm(refs_embeds, ord=2, axis=1)
hyps_norm = np.linalg.norm(hyps_embeds, ord=2, axis=1)  

cosine = np.sum((refs_embeds*hyps_embeds), axis=1)/refs_norm/hyps_norm

print ('%s,%f'%(sys.argv[1].split('/')[-2], np.mean(cosine)))

'''
# visualize importance 
infersent.visualize('A man plays an instrument.', tokenize=True)
'''
