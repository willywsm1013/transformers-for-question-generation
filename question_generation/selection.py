import json
import os
import sys
import re
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
nltk_tokenizer = TreebankWordTokenizer()

def process(text):
    toks = nltk_tokenizer.tokenize(text)
    return ' '.join([tok.replace("''", '"').replace("``", '"') for tok in toks]).lower()

def get_log_a_prob(elem):
    '''
        a_log_porb = log(P(start)) + log(P(end))
        
        return : 
            average log prob of start position and end position = a_log_prob / 2
    '''
    a_prob = np.array(elem['qa_prob'])
    a_log_prob = np.log(a_prob+1e-12)/2
    return a_log_prob

def get_log_q_prob(elem):
    '''
        q_log_prob = log(P(q))
        
        return :
            average log prob = q_log_prob / len(q)
    '''
    cq_log_prob = np.array(elem['lm_log_prob'])
    cq_token_num = np.array(elem['lm_token_num'])
    cq_log_prob_ln = cq_log_prob/cq_token_num
    return cq_log_prob_ln

def sort_by_score(idx, scores):
    idx = sorted(idx, key=lambda i:scores[i], reverse=True)
    return idx

with open(sys.argv[1]) as f:
    data = json.load(f)

input_file=sys.argv[1]
selection = sys.argv[2]
num = int(sys.argv[3])

d, filename = os.path.split(input_file)
tmp_dir = os.path.join(d, 'tmp')
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)


golden = os.path.join(tmp_dir, 'golden.txt')
generated = os.path.join(tmp_dir, 'generated.txt')
squad_file = os.path.join(tmp_dir, 'generated.json')

with open(golden,'w') as gt_file,  open(generated, 'w') as pred_file:
    select_data = []
    for elem in data:
        log_a_prob = get_log_a_prob(elem)
        log_q_prob = get_log_q_prob(elem)
        log_qa_prob = log_a_prob+log_q_prob

        idx = list(range(len(elem['pred_question'])))
        if selection == 'qg':
            idx = sort_by_score(idx, elem['pred_question_log_prob'])
        elif selection == 'qg_ln':
            probs = np.array(elem['pred_question_log_prob'])
            token_nums = np.array(elem['pred_question_token_num'])
            idx = sort_by_score(idx, probs/token_nums)
        elif selection == 'a':
            idx = sort_by_score(idx, log_a_prob)
        elif selection == 'qa':
            idx = sort_by_score(idx, log_qa_prob)
            
        pred_question = [process(elem['pred_question'][i]) for i in idx[:num]]

        if num == 1:
            print (process(elem['question']), file=gt_file)
            print (pred_question[0], file=pred_file)
        
        select_data.append({'context':elem['context'], 'answers':elem['answers'], 'question':elem['question'], 'pred_question':pred_question, 'id':elem['id']})

with open(squad_file, 'w') as f:
    json.dump(select_data, f, indent=1, ensure_ascii=False)
