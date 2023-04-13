import librosa
import numpy as np
import os
from vad import SigVad
import torch
import torchaudio
_vad = SigVad(top_db=25)
datapath="/lyxx/datasets/raw/ADD2023/train/wav/"
save_path="/lyxx/datasets/raw/ADD2023/train/energy_vad/"
for filename in os.listdir(datapath):
    wav_path = os.path.join(datapath,filename)
    res = _vad.get_speech_endpoint(wav_path=wav_path)
    signal, sr = librosa.load(wav_path, sr=16000)
    predict_speech_save_path=os.path.join(save_path,filename)
    new_signal = np.zeros_like(signal)
    
    for s, e in res:
        start_point = int(s * sr)
        end_point = int(e * sr)
        new_signal[start_point:end_point] = signal[start_point:end_point]
    torchaudio.save(predict_speech_save_path, torch.tensor(new_signal.reshape(1,-1)), 16000)

