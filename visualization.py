

import torch
from utils import plot_spectrogram
from import_idmt_traffic_dataset import import_idmt_traffic_dataset
from datasets.idmt_traffic import IdmtTrafficDataSet
import config
import matplotlib.pyplot as plt

dataframe = all_data = import_idmt_traffic_dataset(config.train_annotations)
dataframe = dataframe[dataframe.vehicle=='C']
dataset = IdmtTrafficDataSet(dataframe, config.SAMPLE_RATE, ['C', 'M', 'B','T', '30', '50', '70', 'D', 'W', 'None', 'UNK'], [], config.ROW_VEHICLES, config.N_MELS)

print(len(dataset))
sample, label, item_class = dataset[0]

fig, axe = plt.subplots()
plot_spectrogram(torch.squeeze(sample), fig, axe,title=item_class)
plt.show()