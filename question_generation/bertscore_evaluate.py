import sys
import numpy as np
from bert_score import score

with open(sys.argv[1]) as f:
    refs = [line[:-1] for line in f]

with open(sys.argv[2]) as f:
    hyps = [ line[:-1] for line in f]

P, R, F1 = score(hyps, refs, lang='en', verbose=True, rescale_with_baseline=False)
print (np.mean(F1.numpy()))

P, R, F1 = score(hyps, refs, lang='en', verbose=True, rescale_with_baseline=True)
print (np.mean(F1.numpy()))
   
