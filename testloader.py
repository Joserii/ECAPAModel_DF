import pandas as pd
import torch
import soundfile as sf
from torch.utils.data.dataloader import Dataset, DataLoader
import numpy
import random
import os


class test_loader(Dataset):
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __len__(self):
        root_path = '/home/zhaosheng/czy/datasets/ADD2023/test'
        filelist = os.listdir(root_path)
        return len(filelist)

    def __getitem__(self, index):
        root_path = '/home/zhaosheng/czy/datasets/ADD2023/test'
        filelist = os.listdir(root_path)
        filelist.sort()
        data_file_path = os.path.join(root_path, filelist[index])
        print(data_file_path)
        audio, _ = sf.read(data_file_path)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        label = torch.tensor(0, dtype=torch.int64)

        return torch.FloatTensor(audio[0]), label
