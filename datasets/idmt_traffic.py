import torchaudio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import config
from config import AUDIO_DIR, MODEL_TYPES, SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS

class IdmtTrafficDataSet(Dataset):

    def __init__(self, annotations_file, target_sample_rate, normal_classes, model_type, on_the_fly=True):
        self.annotations =  annotations_file if isinstance(annotations_file, pd.DataFrame) else pd.read_csv(annotations_file)
        self.audio_dir = AUDIO_DIR #new audio dir with specrtograms
        self.model_type = model_type
        self.audio_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT, # Frame Size
        hop_length=HOP_LENGTH, #here half the frame size
        n_mels=N_MELS
        )
        self.image_transformation = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.ToTensor(),
        ])
        self.auto_encoder_transformation = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.RandomCrop(size=[N_MELS, config.NUMBER_OF_FRAMES_AE]), #only train on random slice of the spectogram
        transforms.ToTensor(),
        ])
        self.target_sample_rate = target_sample_rate
        #self.classes = ['None','C','T', 'M', 'B']
        self.normal_classes = normal_classes
        self.on_the_fly = on_the_fly


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        item_class = self._get_audio_sample_label(index) if self._get_audio_sample_label(index) != None else 'None'
        #print(f"Label: {label}")
        if self.on_the_fly:
            signal, sr = torchaudio.load(audio_sample_path)
            signal = self._resample(signal, sr) #adjust sample rates
            # signal -> (num_channels, samples) i.e. (2, 16000)
            signal  = self._mix_down(signal) #stereo to mono
            signal = self.audio_transformation(signal) #(1, 16000) -> torch.Size([1, 64, 63])
            signal = self.image_transformation(signal)#self.auto_encoder_transformation(signal) if self.model_type == MODEL_TYPES.AUTOENCODER else self.image_transformation(signal)
        else:
            raise Exception("Not implemented yet!")
        #label = self.normal_classes.index(label)
        #print(f"normal classes {self.normal_classes}")
        label = 0 if item_class in self.normal_classes else 1
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

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 8]