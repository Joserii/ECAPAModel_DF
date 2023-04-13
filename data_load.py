import pandas as pd
import torch
import soundfile as sf
from torch.utils.data.dataloader import Dataset, DataLoader
import numpy
import random
import glob
import os
from scipy import signal
import soundfile


class train_loader(Dataset):
    def __init__(self, train_list, train_path, num_frames, musan_path, rir_path):
        self.train_list = pd.read_csv(train_list, sep=' ', header=None)
        self.train_path = train_path
        self.num_frames = num_frames
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15],
                         'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))

        for file in augment_files:
            # print(file)
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        # print(self.noiselist.keys())
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))

    def __len__(self):
        return self.train_list.shape[0]

    def __getitem__(self, index):
        data_file_path = self.train_path + self.train_list.iloc[index, 0]
        try:
            audio, _ = sf.read(data_file_path)
        except Exception as e:
            print(f"Error: {e}")

        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        augtype = random.randint(0, 5)
        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')

        #sample = torch.tensor(sample, dtype=torch.float32)
        #sample = torch.unsqueeze(sample, 0)
        label = self.train_list.iloc[index, 1]
        label = get_label(label)
        return torch.FloatTensor(audio[0]), label

    def get_labels(self):
        labels = self.train_list.iloc[:, 1]
        return labels

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(
                random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4)
            noisesnr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(
                10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(
            noises, axis=0), axis=0, keepdims=True)
        return noise + audio

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)

        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]


def get_label(label):
    if label == 'fake':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'genuine':
        label = torch.tensor(1, dtype=torch.int64)
    return label
