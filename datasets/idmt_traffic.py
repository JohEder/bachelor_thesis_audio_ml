import torchaudio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from torchvision.transforms.transforms import Grayscale
import config
import numpy as np
import librosa
from config import AUDIO_DIR, MODEL_TYPES, SAMPLE_RATE, HOP_LENGTH, N_FFT

class IdmtTrafficDataSet(Dataset):

    def __init__(self, annotations_file, target_sample_rate, normal_classes, anomalous_classes, row, mel_bins, on_the_fly=True):
        self.annotations =  annotations_file if isinstance(annotations_file, pd.DataFrame) else pd.read_csv(annotations_file)
        self.audio_dir = AUDIO_DIR #new audio dir with specrtograms
        self.audio_transformation = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT, # Frame Size
        win_length=N_FFT//2,
        hop_length=HOP_LENGTH, #here half the frame size
        n_mels=mel_bins,
        #normalized=True #magnitude normalisation
        ),
        torchaudio.transforms.AmplitudeToDB()
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
            signal, sr = torchaudio.load(audio_sample_path, normalize=False)
            signal  = self._mix_down(signal) #stereo to mono
            signal = self._resample(signal, sr) #adjust sample rates
            # signal -> (num_channels, samples) i.e. (2, 16000)
            signal = self.audio_transformation(signal) #(1, 16000) -> torch.Size([1, 64, 63])
            #print(signal.shape) #(1, 64, 87)
            #signal = self.min_max_normalize(signal, 1, 0)
            signal = self.normalize(signal)
            #signal = self._normalize_clipping(signal)
            #print(signal)
            #signal = self.image_transformation(signal)
            #signal = self.image_normalize(signal)
            #signal = self.extract_log_spec(signal)
            #signal = self.min_max_normalize(signal, 1, 0)
            #print(signal.shape)
            #signal = self.spec_to_image(signal)
            #signal = self.normalize_mean(signal)
            #print(f"Max: {signal.max()}, Min: {signal.min()}")

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

    def normalize(self, tensor):
        #Reasonable reconstruction, bad detection  
        # Subtract the mean, and scale to the interval [-1,1]
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean / tensor_minusmean.abs().max()

    def normalize_mean(self, tensor):
        #works really well for reconstruction, and ok for detection        
        return (tensor - tensor.mean()) / (tensor.max() - tensor.min())

    def image_normalize(self, tensor):
        return tensor.float().div(255)

# Let's normalize to the full interval [-1,1]
# waveform = normalize(waveform)


    def spec_to_image(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        #spec_scaled = spec_scaled.numpy()
        #spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    def load_librosa(self, file_path):
        signal, sr = librosa.load(file_path, mono=True, sr=config.SAMPLE_RATE)
        return signal, sr

    def extract_log_spec(self, signal):
        stft = librosa.stft(signal, n_fft=config.N_FFT, hop_length=HOP_LENGTH)[:-1] #(1+ frame_size/2, num_frames)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

    def min_max_normalize(self, signal, max, min):
        norm_array = (signal - signal.min()) / (signal.max() - signal.min())
        norm_array = norm_array * (max - min) + min
        return norm_array
    
    def min_max_denormalize(self, signal, orig_min, orig_max, max, min):
        array = (signal - min) / (max - min)
        array = array * (orig_max - orig_min) + orig_min

    def _normalize_clipping(self, S):
        #funktioniert gar nicht
        return np.clip((S + S.min()) / S.min(), 0, 1)