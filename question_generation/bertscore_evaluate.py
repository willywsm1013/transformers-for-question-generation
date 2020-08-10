import sys
import numpy as np
import json
from argparse import ArgumentParser
from bert_score import score

parser = ArgumentParser()
parser.add_argument('--golden', help='golden reference sentences')
parser.add_argument('--generated', help='generted sentences')
parser.add_argument('--output_file', help='output file', default=None)
args = parser.parse_args()

with open(args.golden) as f:
    refs = [line[:-1] for line in f]

with open(args.generated) as f:
    hyps = [ line[:-1] for line in f]

result={}

P, R, F1 = score(hyps, refs, lang='en', verbose=True, rescale_with_baseline=False)
result['bertscore'] = float(np.mean(F1.numpy()))

P, R, F1 = score(hyps, refs, lang='en', verbose=True, rescale_with_baseline=True)
result['bertscore_norm'] = float(np.mean(F1.numpy()))

if args.output_file is not None:
    with open(args.output_file, 'a') as f:
        print(json.dumps(result), file=f)
else:
    print (result)

   
