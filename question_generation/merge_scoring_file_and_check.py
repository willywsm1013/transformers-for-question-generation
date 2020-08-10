import json
import sys

qlm_scoring_file = sys.argv[1]
qa_scoring_file = sys.argv[2]
output_file = sys.argv[3]

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def dump_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


qlm_data = load_data(qlm_scoring_file)
qa_data = load_data(qa_scoring_file)

assert len(qlm_data) == len(qa_data)

for qlm_elem, qa_elem in zip(qlm_data, qa_data):
    qlm_elem.update(qa_elem)

dump_data(qlm_data, output_file)
