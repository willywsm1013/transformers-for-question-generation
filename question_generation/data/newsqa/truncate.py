import json
import sys
import re
from collections import defaultdict

def read_data(path):
    print ('read %s'%path)
    with open(path) as f:
        return json.load(f)

def paragraph_tokenize(text):
    eos=False
    paras=[]
    start_pos = []
    para = ''
    start = 0
    for i, ch in enumerate(text):
        if ch == '\n':
            if not eos:
                paras.append(para)
                start_pos.append(start)
                para = ''
                eos = True
        else:
            if eos:
                start = i
                eos = False
            para += ch
    if para != '':
        paras.append(para)
        start_pos.append(start)

    return paras, start_pos

def truncate(data, accept_sent_num):
    new_data = []
    for elem in data['data']:
        title = elem['title']
        new_paragraphs = []
        for para in elem['paragraphs']:
            context = para['context']
            paragraphs, para_start_pos = paragraph_tokenize(context)
            new_context_cache = defaultdict(list)
            for qa in para['qas']:
                a = qa['consensus']
                a_text = a['text']
                a_start = a['answer_start']
                para_id = [ i for i, p in enumerate(para_start_pos) if p<=a_start<p+len(paragraphs[i])][0]
                
                start_para_id = max(para_id-accept_sent_num, 0)
                end_para_id = min(para_id+accept_sent_num, len(paragraphs)-1)

                context_start_pos = para_start_pos[start_para_id]
                context_end_pos = para_start_pos[end_para_id]+len(paragraphs[end_para_id])
                new_context = context[context_start_pos:context_end_pos]
                new_answers = [{'text':a['text'], 'answer_start':a['answer_start']-context_start_pos} for a in qa['answers']
                                if a['answer_start']-context_start_pos >= 0]
                
                new_context_cache[new_context].append({'question':qa['question'],
                                                      'id':qa['id'],
                                                      'answers':new_answers,
                                                      'consensus':{'text':a_text, 'answer_start':a_start-context_start_pos}})
            for context, qas in new_context_cache.items():
                new_paragraphs.append({'context':context,
                                 'qas':qas})
        new_data.append({'title':title,
                         'paragraphs':new_paragraphs})
    return new_data

def truncate_train_context(data, accept_sent_num):
    new_data = []
    for elem in data['data']:
        title = elem['title']
        new_paragraphs = []
        for para in elem['paragraphs']:
            context = para['context']
            paragraphs, para_start_pos = paragraph_tokenize(context)
            for i in range(accept_sent_num, len(paragraphs)-accept_sent_num, accept_sent_num):
                start_para_id = max(i - accept_sent_num, 0)
                end_para_id = min(i + accept_sent_num, len(paragraphs)-1)

                context_start_pos = para_start_pos[start_para_id]
                context_end_pos = para_start_pos[end_para_id]+len(paragraphs[end_para_id])
                
                context_start_pos = para_start_pos[start_para_id]
                context_end_pos = para_start_pos[end_para_id]+len(paragraphs[end_para_id])
                new_context = context[context_start_pos:context_end_pos]
                new_paragraphs.append({'context':new_context})
        new_data.append({'title':title,
                         'paragraphs':new_paragraphs})
    return new_data


data = read_data(sys.argv[1])
accept_sent_num = int(sys.argv[2])
new_data = truncate_train_context(data, accept_sent_num)

#if 'train_context' in sys.argv[1]:
#    new_data = truncate_train_context(data, accept_sent_num)
#elif 'dev' in sys.argv[1] or 'train' in sys.argv[1] or 'test' in sys.argv[1]:
#    new_data = truncate(data, accept_sent_num)
#else:
#    raise Exception

with open('%s_%s_context.json'%(re.sub('\.json', '', sys.argv[1]), accept_sent_num), 'w') as f: 
    json.dump({'version':data['version'],
               'data':new_data}, f, indent=1, ensure_ascii=False)
