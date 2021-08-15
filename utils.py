from posixpath import join
import matplotlib
from seaborn.distributions import kdeplot
from seaborn.utils import ci
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import librosa

from torch.utils import data
import config


def save_model(model_name, model, epoch):
  model_name += '_' + str(epoch)
  model_name_save = model_name + '.pth'
  torch.save(model, config.RESULT_DIR  + model_name_save)
  return model_name

def load_model(name):
  name +='.pth'
  model = torch.load(config.RESULT_DIR + name)
  return model

def plot_spectrogram(spec, fig, axs, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)

def plot_roc_curve(title, fp_rate, tp_rate, roc_auc, axe):
  axe.plot(fp_rate, tp_rate, color='blue', label=f"ROC_AUC ={roc_auc}")

  axe.set_xlabel('False Positive Rate')
  axe.set_ylabel('True Positive Rate')
  axe.set_title('ROC Curve of ' + title)
  axe.legend(loc="lower right")
  #axe.savefig(config.RESULT_DIR + title + '.jpg')
  #plt.show()

def plot_error_distribution(axe, scores_classes, title):
  test_anom_scores, test_targets = scores_classes
  recons_errros = {'Class' : test_targets, 'Error' : test_anom_scores}
  df_recons_errors = pd.DataFrame(recons_errros, columns=recons_errros.keys())
  print(df_recons_errors.describe())
  print(df_recons_errors.head())
  print(df_recons_errors.tail())
  axe.set_title(title)
  sns.kdeplot(data=df_recons_errors, x='Error', hue='Class', ax=axe)

def plot_and_save_loss_curve(title, losses):
  figure_2 = plt.figure(2)
  figure_2 = plt.plot(losses, label='loss')
  plt.title(title)
  plt.xlabel('steps')
  plt.ylabel('loss')
  #plt.savefig(config.RESULT_DIR + title + '.jpg')
  #plt.show()
  return figure_2

def plot_all_rocs(title, roc_aucs, axe):
  roc_aucs = pd.DataFrame(roc_aucs, columns=roc_aucs.keys())
  sns.pointplot(data=roc_aucs, ax=axe, join=False, palette='inferno')
  axe.set_title(title)

def plot_all_results(df, axe):
  sns.pointplot(data=df, ax=axe, y='ROC_AUC', x='Normal_Data', hue='Model_Type', join=False, palette='inferno', dodge=0.2, ci='sd', capsize=.175)

def plot_mel_filter_experiment(df, axe):
  sns.pointplot(data=df, ax=axe, y='ROC_AUC', x='mel_filters', hue='Model_Type', join=False, palette='inferno', dodge=0.2, ci='sd', capsize=.175)

def convert_to_df(losses):
  losses_map = {}
  losses_map['loss'] = losses
  losses_df = pd.DataFrame(losses_map, columns=losses_map.keys())
  return losses_df

def save_hyperparams(model_type, model_name, training_time, optimizer, learning_rate, epochs, normal_classes, anomalous_classes, roc_auc, summary, weight_decay="", total_steps="", warm_up_steps="", mel_bins=config.N_MELS):
  if model_type == config.MODEL_TYPES.TRANSFORMER:
    with open(config.RESULT_DIR + "hyper_params" + model_name + '_' + str(model_type) + ".txt", 'w') as f:
      f.write(f"Model Name: {model_name}\n" +
          f"Epochs: {epochs}, Training Time: {training_time} Learning Rate: {learning_rate} BatchSize: {config.BATCH_SIZE}, Optimizer: {optimizer}, Weight Decay: {weight_decay} Total Steps: {total_steps}, Warm up Steps: {warm_up_steps}\n" +
          f"SAMPLE_RATE = {config.SAMPLE_RATE}, N_FFT/WINDOW_SIZE = {config.N_FFT}, HOP_LENGTH = {config.HOP_LENGTH}, N_MELS = {mel_bins}\n" + 
          f"NUMBER_OF_FRAMES: {config.NUMBER_OF_FRAMES}, EMBEDDING_SIZE = {config.EMBEDDING_SIZE}, N_HEADS = {config.N_HEADS}, N_ENCODER_LAYERS = {config.N_ENCODER_LAYERS}, DROPOUT = {config.DROPOUT}, DIM_FEED_FORWARD = {config.DIM_FEED_FORWARD}\n"+
          f"Normal Classes: {normal_classes}, Anomalous Classes: {anomalous_classes}, ROC_AUC Score: {roc_auc}  \n\n {summary}")