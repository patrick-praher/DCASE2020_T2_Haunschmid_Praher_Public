import sys
import os
import wave
import contextlib


def wav_length(file_path):
    with contextlib.closing(wave.open(file_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


wav_lengths = []

print(sys.argv[1])
for r, d, f in os.walk(sys.argv[1]):
    for file in f:
        if file.endswith(".wav"):
            file_path = os.path.join(r, file)
            wav_lengths.append(wav_length(file_path))            
            #print(file_path + str(wav_lenth(file_path)))

print ("Max: ", max(wav_lengths))
print ("Min: ", min(wav_lengths))

