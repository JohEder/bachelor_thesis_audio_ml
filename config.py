from enum import Enum
import torchaudio
from torchvision import datasets, transforms

CLASSES = ['None','C','T', 'M', 'B'] #Background Noise, Car, Truck, Motorcycle, Bus
NORMAL_CLASSES = ['None', 'C']
ANOMALOUS_CLASSES = ['T', 'M', 'B']

MODEL_TYPES = Enum("MODEL_TYPES", ["TRANSFORMER", "AUTOENCODER", "IDNN"])

NUMBER_REPEAT_EXPERIMENT = 5

#params for mel spectrogram
SAMPLE_RATE = 22500
N_FFT=2048 #is also window size
HOP_LENGTH=1024
N_MELS=128
NUMBER_OF_FRAMES = 2
NUMBER_OF_FRAMES_AE = 4

AUDIO_DIR = "/home/johannes/datasets/IDMT_Traffic/audio"
train_annotations = "/home/johannes/datasets/IDMT_Traffic/annotation/eusipco_2021_train.csv"
test_annotatons = "/home/johannes/datasets/IDMT_Traffic/annotation/eusipco_2021_test.csv"
all_annotations_txt = "/home/johannes/datasets/IDMT_Traffic/annotation/idmt_traffic_all.txt"
RESULT_DIR = '/home/johannes/bachelor_thesis_code/code/results/'

BATCH_SIZE = 32
BATCH_SIZE_VAL = 1

RANDOM_SEED = 42

#params for transformer model architecture
EMBEDDING_SIZE = 128
N_HEADS = 4
N_ENCODER_LAYERS = 4
DROPOUT = 0.0 #is dropout needed for AD? used in most Transformer papers
DIM_FEED_FORWARD = 256
input_dim = 256

model_name = 'transformer_07_adam_d_1e-5_lr_1e-5'