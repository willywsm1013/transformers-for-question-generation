import json
import spacy
import sys
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

with open(sys.argv[1]) as f:
    data = json.load(f)

def extract_ner(context):
    doc = nlp(context)
    answers = [{'text':ent.text, 'answer_start':ent.start_char} for ent in doc.ents]

    return answers

def extract_noun_phrase(context):
    doc = nlp(context)
    answers = [[{'text':noun.text,'answer_start':noun.start_char}] for noun in doc.noun_chunks]
    return answers

new_data = []
for elem in tqdm(data['data']):
    for para in elem['paragraphs']:
        context = para['context']
        doc = nlp(context)

        answers = extract_ner(context)

        new_data.append({'context':context, 'answers':answers})

with open(sys.argv[2], 'w') as f:
    json.dump(new_data, f, indent=1, ensure_ascii=False)



