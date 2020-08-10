import json
import sys

result = {}
with open(sys.argv[1]) as f:
    for line in f:
        result.update(json.loads(line))

with open(sys.argv[2], 'w') as f:
    for k, v in result.items():
        print ('%s, %f'%(k,v), file=f)
