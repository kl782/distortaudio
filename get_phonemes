import pip
%pip install allosaurus
from allosaurus.app import read_recognizer
model = read_recognizer()
%pip install librosa
from scipy.io import wavfile
%pip install sounddevice
import sounddevice as sd
import numpy as np
import random
import librosa

def get_phonemes(audio_file, other_phonemes={},num_samples = 3):
 #Turns allosaurus' text file into a tuple
    sample = model.recognize(audio_file, timestamp=True)
    samplerows = [x.split() for x in sample.split("\n")]
    samplerows = [(float(start), float(dur),phoneme)for start,dur,phoneme in samplerows]
#create a dictionary with the phoneme as key, and the timestamps as entries
    timestamps_dict = dict()
    for start, dur, phoneme in samplerows:
        if phoneme not in timestamps_dict:
            timestamps_dict[phoneme] = [(start,length)]
    else:
        timestamps_dict[phoneme].append((start,length))

#phoneme dict = to add onto an existing catalogue
    audio, sr = librosa.load(audio_file, sr=None)
    for phoneme, timestamps in timestamps_dict.items():
        # Create an empty list to store the audio clips for the current phoneme
        clips = []
        random_timestamps = random.sample(timestamps, min(num_samples, len(timestamps)))

        # Iterate through the timestamps and split the audio
        for start_time, duration in random_timestamps:
            start_sample = int(start_time * sr)
            end_sample = int((start_time+duration) * sr)
            segment = audio[start_sample:end_sample]
            clips.append(segment)

        other_phonemes[phoneme] = clips

    return other_phonemes
