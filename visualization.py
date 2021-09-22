

import torch
import torchaudio
from utils import plot_spectrogram
from import_idmt_traffic_dataset import import_idmt_traffic_dataset
from datasets.idmt_traffic import IdmtTrafficDataSet
import config
import matplotlib.pyplot as plt

dataframe = all_data = import_idmt_traffic_dataset(config.all_annotations_txt)
dataframe = dataframe[dataframe.vehicle=='B']
dataset = IdmtTrafficDataSet(dataframe, config.SAMPLE_RATE, ['C', 'M', 'B','T', '30', '50', '70', 'D', 'W', 'None', 'UNK'], [], config.ROW_VEHICLES, config.N_MELS)

print(len(dataset))
sample, label, item_class = dataset[2]

fig, axe = plt.subplots()
fig, axe = plot_spectrogram(torch.squeeze(sample), fig, axe,title=item_class)
fig.savefig(config.RESULT_DIR + str(item_class) + '_melspec.png')
plt.show()

filename = '/home/johannes/datasets/IDMT_Traffic/audio/'
waveform, sample_rate = torchaudio.load(filename)
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))

plt.plot(waveform.t().numpy())
plt.show()