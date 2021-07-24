from posixpath import join
import matplotlib
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from torch.utils import data
import config

def save_model_comlicated(experiment_name, scenario, model, model_name, epoch):
  #a valid destination could be ~/results/my_experiment/scenario_1/CTBM/CTBM.pth
  model_name += '_' + str(epoch)
  path = config.RESULT_DIR + experiment_name + "/" + scenario + "/" + model_name
  if not os.path.exists(path):
    os.makedirs(path)
  
  model_name_save = model_name + '.pth'
  torch.save(model, path + "/" + model_name_save)
  return model_name

def save_model(model_name, model, epoch):
  model_name += '_' + str(epoch)
  model_name_save = model_name + '.pth'
  torch.save(model, config.RESULT_DIR  + model_name_save)
  return model_name

def load_model(name):
  name +='.pth'
  model = torch.load(config.RESULT_DIR + name)
  return model

def plot_and_save_roc_curve(title, fp_rate, tp_rate, roc_auc):
  fig = plt.figure(1)
  plt.plot(fp_rate, tp_rate, color='blue', label=f"ROC_AUC ={roc_auc}")

  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve of ' + title)
  plt.legend(loc="lower right")
  plt.savefig(config.RESULT_DIR + title + '.jpg')
  #plt.show()
  return fig

def plot_and_save_loss_curve(title, losses):
  figure_2 = plt.figure(2)
  figure_2 = plt.plot(losses, label='loss')
  plt.title(title)
  plt.xlabel('steps')
  plt.ylabel('loss')
  plt.savefig(config.RESULT_DIR + title + '.jpg')
  #plt.show()
  return figure_2

def plot_all_rocs(title, roc_aucs):
  roc_aucs = pd.DataFrame(roc_aucs, columns=roc_aucs.keys())
  plt.figure(3)
  plot = sns.pointplot(data=roc_aucs, join=False, palette='inferno')
  plt.ylabel('ROC AUC Scores')
  fig_roc = plot.get_figure()
  fig_roc.savefig(config.RESULT_DIR + title + '.jpg')
  #matplotlib.pyplot.show()
  return fig_roc

def save_hyperparams(model_type, model_name, training_time, optimizer, learning_rate, epochs, normal_classes, anomalous_classes, roc_auc, summary, weight_decay="", total_steps="", warm_up_steps=""):
  if model_type == config.MODEL_TYPES.TRANSFORMER:
    with open(config.RESULT_DIR + "hyper_params" + model_name + ".txt", 'w') as f:
      f.write(f"Model Name: {model_name}\n" +
          f"Epochs: {epochs}, Training Time: {training_time} Learning Rate: {learning_rate} BatchSize: {config.BATCH_SIZE}, Optimizer: {optimizer}, Weight Decay: {weight_decay} Total Steps: {total_steps}, Warm up Steps: {warm_up_steps}\n" +
          f"SAMPLE_RATE = {config.SAMPLE_RATE}, N_FFT/WINDOW_SIZE = {config.N_FFT}, HOP_LENGTH = {config.HOP_LENGTH}, N_MELS = {config.N_MELS}\n" + 
          f"NUMBER_OF_FRAMES: {config.NUMBER_OF_FRAMES}, EMBEDDING_SIZE = {config.EMBEDDING_SIZE}, N_HEADS = {config.N_HEADS}, N_ENCODER_LAYERS = {config.N_ENCODER_LAYERS}, DROPOUT = {config.DROPOUT}, DIM_FEED_FORWARD = {config.DIM_FEED_FORWARD}\n"+
          f"Normal Classes: {normal_classes}, Anomalous Classes: {anomalous_classes}, ROC_AUC Score: {roc_auc}  \n\n {summary}")