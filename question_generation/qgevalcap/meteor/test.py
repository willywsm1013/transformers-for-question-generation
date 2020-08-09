import subprocess
import threading

METEOR_JAR = 'meteor-1.5.jar'
meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
        '-', '-', '-stdio', '-l', 'en', 
        '-norm',
        ]

meteor_p = subprocess.Popen(meteor_cmd, \
        cwd='./', \
        stdin=subprocess.PIPE, \
        stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE,
        universal_newlines=True)
lock = threading.Lock()

test_input = 'SCORE ||| which nfl team represented the nfc at super bowl 50 ? ||| who did the broncos beat to win the super bowl ?'


for i in range(2):
    meteor_p.stdin.write('{}\r\n'.format(test_input))
    meteor_p.stdin.flush()
    score = meteor_p.stdout.readline()

    print (score)
