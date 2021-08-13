from datasets.idmt_traffic import IdmtTrafficDataSet
from import_idmt_traffic_dataset import import_idmt_traffic_dataset
import config
import matplotlib.pyplot as plt
from utils import plot_spectrogram

def _plot_samples(samples):
  fig, axes = plt.subplots(1, 1)
  for i in range(len(samples)):
    sample, label, orig_class = samples[i]
    sample = sample.squeeze()
    print(sample.shape)
    plot_spectrogram(sample, fig, axes)
    #fig.add_subplot(1, len(samples), i+1)
    #plt.axis('off')
    #plt.imshow(sample)
  plt.show()

def plot_number_of_samples(number_of_samples, data, row, save=False):
    dataset = IdmtTrafficDataSet(data, config.SAMPLE_RATE, ['None', 'C', 'B', 'T', 'M', 'D', 'W', '30', '50', '70'], [], row)
    samples_to_plot = [dataset[j] for j in range(number_of_samples)]
    _plot_samples(samples_to_plot)
    #print(samples_to_plot)
    




data = import_idmt_traffic_dataset(config.all_annotations_txt)
data = data[data.vehicle == 'C']

plot_number_of_samples(1, data, config.ROW_VEHICLES)