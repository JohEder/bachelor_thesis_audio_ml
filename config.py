from enum import Enum

CLASSES = ['None','C','T', 'M', 'B'] #Background Noise, Car, Truck, Motorcycle, Bus


MODEL_TYPES = Enum("MODEL_TYPES", ["TRANSFORMER", "AUTOENCODER", "IDNN"])
SETUP_TYPES = Enum("SETUP_TYPES", ['VEHICLES', 'WEATHER', 'VELOCITY'])

NUMBER_REPEAT_EXPERIMENT = 5

#params for mel spectrogram
SAMPLE_RATE = 22500
N_FFT=2048 #is also window size
HOP_LENGTH=1024
N_MELS=128
NUMBER_OF_FRAMES = 2
NUMBER_OF_FRAMES_AE = 5
NUMBER_OF_FRAMES_IDNN = 5


Total_steps = 1000
EPOCHS_TF = 20
EPOCHS_AE = 20
EPOCHS_IDNN = 20

AUDIO_DIR = "/home/johannes/datasets/IDMT_Traffic/audio"
train_annotations = "/home/johannes/datasets/IDMT_Traffic/annotation/eusipco_2021_train.txt"
test_annotatons = "/home/johannes/datasets/IDMT_Traffic/annotation/eusipco_2021_test.txt"
all_annotations_txt = "/home/johannes/datasets/IDMT_Traffic/annotation/idmt_traffic_all.txt"
RESULT_DIR = '/home/johannes/bachelor_thesis_code/code/results/'

BATCH_SIZE = 32
BATCH_SIZE_VAL = 1

RANDOM_SEEDS = [42, 9, 17, 4, 7, 32, 29, 20, 34, 12, 35, 89, 36, 67, 89,68]

#params for transformer model architecture
EMBEDDING_SIZE = 128
N_HEADS = 4
N_ENCODER_LAYERS = 4
DROPOUT = 0.0 #is dropout needed for AD? used in most Transformer papers
DIM_FEED_FORWARD = 256

ROW_VEHICLES = 8
ROW_CONDITIONS = 7
ROW_VELOCIIES = 4