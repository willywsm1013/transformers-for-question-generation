import json

def count_num_qas(data):
    qa_num = 0
    for elem in data['data']:
        for para in elem['paragraphs']:
            qa_num += len(para['qas'])
    return qa_num

# split QG training set and QG generating set
with open('train.json', 'r') as f:
    data = json.load(f)

num = len(data['data'])

QG_train = {'version':'%s_qg-train'%data['version'], 'data':data['data'][:num//2]}
QG_gen = {'version':'%s_qg-generation'%data['version'], 'data':data['data'][num//2:]}
with open('QG_train.json', 'w') as f:
    json.dump(QG_train, f, indent=1, ensure_ascii=False)
with open('QG_gen.json', 'w') as f:
    json.dump(QG_gen, f, indent=1, ensure_ascii=False)

print ('num of qas in QG_train : ', count_num_qas(QG_train))
print ('num of qas in QG_gen : ', count_num_qas(QG_gen))

# split QG dev set and RC dev set
with open('dev.json', 'r') as f:
    data = json.load(f)

num = len(data['data'])

QG_dev = {'version':'%s_qg-dev'%data['version'], 'data':data['data'][:num//2]}
RC_dev = {'version':'%s_rc-dev'%data['version'], 'data':data['data'][num//2:]}

with open('QG_dev.json', 'w') as f:
    json.dump(QG_dev, f, indent=1, ensure_ascii=False)
with open('RC_dev.json', 'w') as f:
    json.dump(RC_dev, f, indent=1, ensure_ascii=False)

print ('num of qas in QG_dev : ', count_num_qas(QG_dev))
print ('num of qas in RC_dev : ', count_num_qas(RC_dev))
