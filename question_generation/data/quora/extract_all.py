import sys
import json
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm

random.seed(42)
# read data
df = pd.read_csv('train.csv')

examples = []
for i, row in tqdm(df.iterrows(), ncols=50, total=df.shape[0]):
    q1 = row.question1
    q2 = row.question2
    label = int(row.is_duplicate)
    if q1 == 'nan' or q2 == 'nan':
        print (q1,q2)
        sys.exit()
    assert q1 != q2
    examples.append({'q1':q1, 'q2':q2, 'label':label})


index = list(range(len(examples)))
random.shuffle(index)

train_index = index[:int(len(examples)*0.9)]
dev_index = index[int(len(examples)*0.9):]

train_examples = [examples[i] for i in train_index]
dev_examples = [examples[i] for i in dev_index]

print ('train data num : ',len(train_examples))
print ('dev data num :', len(dev_examples))

with open('train.json', 'w') as f:
    json.dump(train_examples, f, indent=1, ensure_ascii=False)
with open('dev.json', 'w') as f:
    json.dump(dev_examples, f, indent=1, ensure_ascii=False)
