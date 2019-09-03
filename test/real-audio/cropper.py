# Constant parameters
num_samples = 40
sample_duration = 1000 # works in ms

import sys
from pydub import AudioSegment
import random
import os

filename = sys.argv[1]
exportfolder = 'exp_'+filename[:-4]

audio = AudioSegment.from_wav(filename)

duration = audio.duration_seconds

os.makedirs(exportfolder)
for i in range (num_samples):
    t1 = random.randrange(int(duration - 1)) * 1000 # works in ms
    t2 = t1 + sample_duration
    sample = audio[t1:t2]
    sample.export(exportfolder+'/sample'+ str(i)+'.wav', format = 'wav')