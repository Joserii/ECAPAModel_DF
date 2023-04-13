import pandas as pd
import torch
import soundfile as sf
from torch.utils.data.dataloader import Dataset, DataLoader
import numpy
import random

class eval_loader(Dataset):
    def __init__(self, eval_list, eval_path,num_frames):
        self.eval_list = pd.read_csv(eval_list, sep=' ', header=None)
        self.eval_path = eval_path
        self.num_frames = num_frames

    def __len__(self):
        return self.eval_list.shape[0]

    def __getitem__(self, index):
        data_file_path = self.eval_path + self.eval_list.iloc[index, 0]

        audio, _ = sf.read(data_file_path)
        length = self.num_frames * 160 + 240
        if audio.shape[0]<=length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio],axis=0)
        
        #sample = torch.tensor(sample, dtype=torch.float32)
        #sample = torch.unsqueeze(sample, 0)
        label = self.eval_list.iloc[index, 1]
        label = get_label(label)

        return torch.FloatTensor(audio[0]), label
    def get_labels(self):
        labels = self.eval_list.iloc[:, 1]
        return labels

def get_label(label):
    if label == 'fake':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'genuine':
        label = torch.tensor(1, dtype=torch.int64)
    return label


