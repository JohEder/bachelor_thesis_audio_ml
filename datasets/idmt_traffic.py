import torchaudio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from torchvision.transforms.transforms import Grayscale
import config
from config import AUDIO_DIR, MODEL_TYPES, SAMPLE_RATE, HOP_LENGTH, N_FFT

class IdmtTrafficDataSet(Dataset):

    def __init__(self, annotations_file, target_sample_rate, normal_classes, anomalous_classes, row, mel_bins, on_the_fly=True):
        self.annotations =  annotations_file if isinstance(annotations_file, pd.DataFrame) else pd.read_csv(annotations_file)
        self.audio_dir = AUDIO_DIR #new audio dir with specrtograms
        self.audio_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT, # Frame Size
        hop_length=HOP_LENGTH, #here half the frame size
        n_mels=mel_bins,
        normalized=True #magnitude scaling
        )
        self.image_transformation = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(), #converting to values between 0 and 1
        ])
        self.target_sample_rate = target_sample_rate
        #self.classes = ['None','C','T', 'M', 'B']
        self.normal_classes = normal_classes
        self.anomalous_classes = anomalous_classes
        self.row = row
        self.on_the_fly = on_the_fly


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        item_class = self._get_audio_sample_label(index, self.row) #if self._get_audio_sample_label(index) != None else 'None'
        #print(f"Label: {label}")
        if self.on_the_fly:
            signal, sr = torchaudio.load(audio_sample_path)
            signal  = self._mix_down(signal) #stereo to mono
            signal = self._resample(signal, sr) #adjust sample rates
            # signal -> (num_channels, samples) i.e. (2, 16000)
            signal = self.audio_transformation(signal) #(1, 16000) -> torch.Size([1, 64, 63])
            #print(signal)
            signal = self.image_transformation(signal)
        else:
            raise Exception("Not implemented yet!")
        #label = self.normal_classes.index(label)
        #print(f"normal classes {self.normal_classes}")
        if item_class in self.normal_classes:
            label = 0
        elif item_class in self.anomalous_classes:
            label = 1
        else: 
            raise Exception(f"Class Label {item_class} not in normal and anomalous classes! Wrong Labelling?")
        return signal, label, item_class

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down(self, signal):
        if signal.shape[0] > 1: #(2, 16000)
            #mean operation: aggregating multiple channels
            signal = torch.mean(signal, 0, True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path + '.wav'

    def _get_audio_sample_label(self, index, row):
        return self.annotations.iloc[index, row]