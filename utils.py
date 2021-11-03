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
import datetime

from torch.utils import data
import config


def save_model(model_name, model):
  model_name_save = model_name + '.pth'
  torch.save(model, config.RESULT_DIR  + model_name_save)
  return model_name

def load_model(name):
  name +='.pth'
  model = torch.load(config.RESULT_DIR + name)
  return model

def plot_and_save_orig_and_recons(orginial_recons, orig_class, ad_score):
  original, recons = orginial_recons
  #torch.set_printoptions(threshold=20000)
  print("\nOriginal")
  print(original)
  print("\nReconstruction")
  print(recons)
  original, recons = original.cpu(), recons.cpu()
  fig, axes = plt.subplots(2, 1, figsize=(4, 6))
  plot_spectrogram(original, fig, axes[0], title='Original and Reconstruction of class:' + orig_class + '\nAD Score: '+str(round(ad_score, 2)))
  plot_spectrogram(recons, fig, axes[1])
  fig.savefig(config.RESULT_DIR + str(datetime.datetime.now()) + '_orig_recons_plot.png')
  #plt.show()

def plot_spectrogram(spec, fig, axs, title=None, ylabel='Mel-band', aspect='auto', xmax=None):
  if title != None:
    axs.set_title(title)
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(spec, origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  return fig, axs

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

def plot_loss_func_experiment(df, axe):
  sns.pointplot(data=df, ax=axe, y='ROC_AUC', x='loss_funcs', hue='Model_Type', join=False, palette='inferno', dodge=0.2, ci='sd', capsize=.175)

def plot_mel_filter_experiment(df, axe):
  sns.pointplot(data=df, ax=axe, y='ROC_AUC', x='mel_filters', hue='Model_Type', join=False, palette='inferno', dodge=0.2, ci='sd', capsize=.175)

def convert_to_df(losses):
  losses_map = {}
  losses_map['loss'] = losses
  losses_df = pd.DataFrame(losses_map, columns=losses_map.keys())
  return losses_df

def save_hyperparams(model_type, model_name, training_time, optimizer, learning_rate, epochs, normal_classes, anomalous_classes, roc_auc, summary, best_model_epoch, weight_decay="", total_steps="", warm_up_steps="", mel_bins=config.N_MELS):
  if model_type == config.MODEL_TYPES.TRANSFORMER:
    with open(config.RESULT_DIR + "hyper_params" + model_name + '_' + str(model_type) + ".txt", 'w') as f:
      try:
        f.write(f"Model Name: {model_name}\n" +
          f"Epochs: {epochs}, Training Time: {training_time} Learning Rate: {learning_rate} BatchSize: {config.BATCH_SIZE}, Optimizer: {optimizer}, Weight Decay: {weight_decay} Total Steps: {total_steps}, Warm up Steps: {warm_up_steps}\n" +
          f"SAMPLE_RATE = {config.SAMPLE_RATE}, N_FFT/WINDOW_SIZE = {config.N_FFT}, HOP_LENGTH = {config.HOP_LENGTH}, N_MELS = {mel_bins}\n" + 
          f"NUMBER_OF_FRAMES: {config.NUMBER_OF_FRAMES}, EMBEDDING_SIZE = {config.EMBEDDING_SIZE}, N_HEADS = {config.N_HEADS}, N_ENCODER_LAYERS = {config.N_ENCODER_LAYERS}, DROPOUT = {config.DROPOUT}, DIM_FEED_FORWARD = {config.DIM_FEED_FORWARD}\n"+
          f"Normal Classes: {normal_classes}, Anomalous Classes: {anomalous_classes}, ROC_AUC Score: {roc_auc}, Best model in epoch: {best_model_epoch}  \n\n {summary}")
      except:
        f.write(f"{summary}")
        print('An error occured while saving the hyper parameters')

  elif model_type == config.MODEL_TYPES.AUTOENCODER:
    with open(config.RESULT_DIR + "hyper_params" + model_name + '_' + str(model_type) + ".txt", 'w') as f:
      try:
        f.write(f"Model Name: {model_name}\n" +
          f"Epochs: {epochs}, Training Time: {training_time} Learning Rate: {learning_rate} BatchSize: {config.BATCH_SIZE}\n" +
          f"SAMPLE_RATE = {config.SAMPLE_RATE}, N_FFT/WINDOW_SIZE = {config.N_FFT}, HOP_LENGTH = {config.HOP_LENGTH}, N_MELS = {mel_bins}\n" + 
          f"NUMBER_OF_FRAMES: {config.NUMBER_OF_FRAMES_AE}\n"+
          f"Normal Classes: {normal_classes}, Anomalous Classes: {anomalous_classes}, ROC_AUC Score: {roc_auc}, Best model in epoch: {best_model_epoch}  \n\n {summary}")
      except:
        f.write(f"{summary}")
        print('An error occured while saving the hyper parameters')
  elif model_type == config.MODEL_TYPES.AUTOENCODER:
    with open(config.RESULT_DIR + "hyper_params" + model_name + '_' + str(model_type) + ".txt", 'w') as f:
      try:
        f.write(f"Model Name: {model_name}\n" +
          f"Epochs: {epochs}, Training Time: {training_time} Learning Rate: {learning_rate} BatchSize: {config.BATCH_SIZE}\n" +
          f"SAMPLE_RATE = {config.SAMPLE_RATE}, N_FFT/WINDOW_SIZE = {config.N_FFT}, HOP_LENGTH = {config.HOP_LENGTH}, N_MELS = {mel_bins}\n" + 
          f"NUMBER_OF_FRAMES: {config.NUMBER_OF_FRAMES_IDNN}\n"+
          f"Normal Classes: {normal_classes}, Anomalous Classes: {anomalous_classes}, ROC_AUC Score: {roc_auc}, Best model in epoch: {best_model_epoch}  \n\n {summary}")
      except:
        f.write(f"{summary}")
        print('An error occured while saving the hyper parameters')



def plot_waveform(waveform, sample_rate, title="Waveform", name='waveform', xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  figure.savefig(config.RESULT_DIR + name)
  plt.show()