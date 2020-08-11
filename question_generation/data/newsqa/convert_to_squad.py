import json

def read_ids(path):
    ids = []
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            ids.append(line.strip('\n'))

    return set(ids)

def read_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

def convert_to_squad_format(data, mode):
    squad_format_data=[]
    qas_num = 0
    bad_qas_num = 0
    for elem in data:
        context = elem['text']

        qas = []
        for q_id, question in enumerate(elem['questions']):
            q_text = question['q']
            answers = []
            if 'noAnswer' in question['consensus'] or 'badQuestion' in question['consensus']:
                continue
            
            # consensus
            consensus = question['consensus']
            answer_start = consensus['s']
            answer_end = consensus['e']
            text = context[answer_start:answer_end].strip(' \n')
            consensus = {'text':text, 'answer_start':answer_start}

            if mode == 'train':
                answers.append(consensus)
            else:
                for answer in question['answers']:
                    answer_start = answer['sourcerAnswers'][0].get('s', None)
                    if answer_start is None:
                        continue
                    answer_end = answer['sourcerAnswers'][0]['e']
                    text = context[answer_start:answer_end].strip(' \n')
                    answers.append({'text':text, 'answer_start':answer_start})
            
                if len(answers) == 0:
                    continue
            qas.append({'question':q_text, 'answers':answers, 'id':'%s_%d'%(elem['storyId'], q_id), 
                        'consensus':consensus})
            qas_num += 1
        squad_format_data.append({'title':elem['storyId'],
                                  'paragraphs':[{'context':context,
                                                 'qas':qas}]})
    print ('bad_qas/total_qas : %d/%d'%(bad_qas_num, qas_num))
    return squad_format_data

train_story_id = read_ids('maluuba/newsqa/train_story_ids.csv')
dev_story_id = read_ids('maluuba/newsqa/dev_story_ids.csv')
test_story_id = read_ids('maluuba/newsqa/test_story_ids.csv')

data = read_data('combined-newsqa-data-v1.json')

train_data=[]
dev_data=[]
test_data=[]
for elem in data['data']:
    story_id = elem['storyId']
    if story_id in train_story_id:
        train_data.append(elem)
    elif story_id in dev_story_id:
        dev_data.append(elem)
    elif story_id in test_story_id:
        test_data.append(elem)
    else:
        sys.exit()

train_data = convert_to_squad_format(train_data, mode='train')
dev_data = convert_to_squad_format(dev_data, mode='dev')
test_data = convert_to_squad_format(test_data, mode='test')


save_data({'version':data['version'], 'data':train_data}, 'train.json')
save_data({'version':data['version'], 'data':dev_data}, 'dev.json')
save_data({'version':data['version'], 'data':test_data}, 'test.json')
