from datasets.idmt_traffic import IdmtTrafficDataSet
from import_idmt_traffic_dataset import import_idmt_traffic_dataset
import config
import matplotlib.pyplot as plt
from utils import plot_spectrogram

def _plot_samples(samples, title):
  fig, axes = plt.subplots(1, len(samples), figsize=(8*len(samples), 4))
  for i in range(len(samples)):
    sample, label, orig_class = samples[i]
    sample = sample.squeeze()
    print(sample.shape)
    plot_spectrogram(sample, fig, axes[i], title=orig_class)
  fig.savefig(config.RESULT_DIR + title + '.png')
  plt.show()

def plot_number_of_samples(number_of_samples, data, row,title, save=False):
    dataset = IdmtTrafficDataSet(data, config.SAMPLE_RATE, ['None', 'C', 'B', 'T', 'M', 'D', 'W', '30', '50', '70'], [], row)
    samples_to_plot = [dataset[j] for j in range(number_of_samples)]
    _plot_samples(samples_to_plot, title)
    #print(samples_to_plot)
    




data = import_idmt_traffic_dataset(config.all_annotations_txt)
data = data[data.vehicle == 'T']

plot_number_of_samples(5, data, config.ROW_VEHICLES, 'trucks')