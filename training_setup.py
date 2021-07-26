import datetime
import math
import models.autoencoder
from torch import nn

from sklearn.utils import shuffle
import models.transformer
from models.transformer import TransformerModel
from torch.nn.modules import module
import config
from config import AUDIO_DIR, all_annotations_txt, BATCH_SIZE, SAMPLE_RATE, BATCH_SIZE_VAL, MODEL_TYPES, EMBEDDING_SIZE, input_dim, N_HEADS, DIM_FEED_FORWARD, N_ENCODER_LAYERS, NUMBER_OF_FRAMES
from import_idmt_traffic_dataset import *
import pandas as pd
from datasets.idmt_traffic import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from utils import load_model, plot_and_save_loss_curve, save_model, plot_roc_curve, save_hyperparams
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AdamW
import pytorch_model_summary as pms
import copy


class TrainingSetup():
    def __init__(self, normal_data, anomalous_data, model_type):
        self.normal_data = normal_data
        self.anomalous_data = anomalous_data
        self.model_type = model_type
        self.setup_name = '_'.join(normal_data)
        #get train, val, test data set with normal data

    def __str__(self):
        return f'Normal classes: {self.normal_data} Anomalous:{self.anomalous_data}'

    def run(self, number_of_runs=1):
      auc_roc_scores = []
      for i in range(number_of_runs):
        model_name = self.setup_name
        print(f"\nrunning {i + 1}. Run of Setup: {self.normal_data}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_type == MODEL_TYPES.TRANSFORMER:
          train_loader, val_loader, test_loader = self.get_normal_and_anomalous_data( all_annotations_txt, BATCH_SIZE, BATCH_SIZE_VAL, MODEL_TYPES.TRANSFORMER)
          #training
          transformer = TransformerModel(EMBEDDING_SIZE, input_dim, N_HEADS, DIM_FEED_FORWARD, N_ENCODER_LAYERS)
          LEARNING_RATE = 0.00001
          WEIGHT_DECAY = 0.0001
          #optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
          optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE) #AdamW(transformer.parameters(), lr=LEARNING_RATE)
          EPOCHS = 20 #later over hundred
          total_steps = len(train_loader) * EPOCHS
          warm_up_steps = math.ceil(total_steps * 0.1)
          scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)
          print(f"Total Steps: {total_steps}, Warm up steps: {warm_up_steps}, Ratio: {warm_up_steps / total_steps}")
          transformer.to(device)
          transformer.train() #mode
          roc_auc_best = -1
          best_model = None
          training_start = datetime.datetime.now()
          losses = []
          for epoch in range(1, EPOCHS + 1):
            losses_epoch = models.transformer.train_epoch(transformer, train_loader, optimizer, epoch, device, scheduler=scheduler)
            val_anom_scores, val_targets = models.transformer.get_anom_scores(transformer, val_loader, device) #batch size in evalution is only one
            roc_auc = roc_auc_score(val_targets, val_anom_scores)
            losses += losses_epoch
            if len(val_loader) > 50:
              if roc_auc > roc_auc_best:
                best_model = copy.deepcopy(transformer)
                #model_name = save_model(model_name, transformer, epoch)
                roc_auc_best = roc_auc
                print(f"Model with best validaton in epoch{epoch}")
            else:
              best_model = copy.deepcopy(transformer)
              roc_auc_best = roc_auc
            print(f"Evaluation ROC Score in epoch {epoch} is {roc_auc}, Best ROC Score is:{roc_auc_best}")
          training_finished = datetime.datetime.now()
          total_training_time = training_finished - training_start
          summary = pms.summary(transformer, torch.ones(BATCH_SIZE, 22, 256).to(device))
        elif self.model_type == MODEL_TYPES.AUTOENCODER:
          train_loader, val_loader, test_loader = self.get_normal_and_anomalous_data( all_annotations_txt, BATCH_SIZE, BATCH_SIZE_VAL, MODEL_TYPES.AUTOENCODER)
          #print(next(iter(train_loader)).shape)
          LEARNING_RATE = 0.001
          EPOCHS = 1
          autoencoder = models.autoencoder.AutoEncoder(input_dim=512)
          optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

          autoencoder.to(device)
          autoencoder.train() #mode
          roc_auc_best = -1
          best_model = None
          training_start = datetime.datetime.now()
          losses = []
          for epoch in range(1, EPOCHS + 1):
            losses_epoch = models.autoencoder.train_epoch(autoencoder, train_loader, optimizer, epoch, device)
            val_anom_scores, val_targets = models.autoencoder.get_anom_scores(autoencoder, val_loader, device) #batch size in evalution is only one
            roc_auc = roc_auc_score(val_targets, val_anom_scores)
            losses += losses_epoch
            if len(val_loader) > 50:
              if roc_auc > roc_auc_best:
                best_model = copy.deepcopy(autoencoder)
                #model_name = save_model(model_name, transformer, epoch)
                roc_auc_best = roc_auc
                #print(f"saved model with best validaton in epoch{epoch}")
            else:
              best_model = copy.deepcopy(autoencoder)
              roc_auc_best = roc_auc
            print(f"Evaluation ROC Score in epoch {epoch} is {roc_auc}, Best ROC Score is:{roc_auc_best}")
          training_finished = datetime.datetime.now()
          total_training_time = training_finished - training_start
          summary = pms.summary(autoencoder, torch.ones(BATCH_SIZE, 512).to(device))

        elif self.model_type == MODEL_TYPES.IDNN:
          raise Exception("not implemented")
        #evaluation
        fp_rate, tp_rate, roc_auc = self.evaluate_model(best_model, test_loader, device)
        print(f"ROC AUC of Model {model_name} is {roc_auc}!")
        auc_roc_scores.append(roc_auc)
        #plot_and_save_loss_curve(self.setup_name, losses)
        save_hyperparams(self.model_type, model_name, total_training_time, optimizer, LEARNING_RATE, EPOCHS, self.normal_data, self.anomalous_data, roc_auc, summary, weight_decay="", total_steps="", warm_up_steps="")
      return auc_roc_scores, losses, fp_rate, tp_rate, roc_auc


    def evaluate_model(self, best_model, test_loader,device):
      #model = load_model(model_name)
      if self.model_type == MODEL_TYPES.TRANSFORMER:
        test_anom_scores, test_targets = models.transformer.get_anom_scores(best_model, test_loader, device)
        fp_rate, tp_rate, _ = roc_curve(test_targets, test_anom_scores, pos_label=1)
        roc_auc = roc_auc_score(test_targets, test_anom_scores)
        #plot_roc_curve(self.setup_name, fp_rate, tp_rate, roc_auc)
        return fp_rate, tp_rate, roc_auc
      elif self.model_type == MODEL_TYPES.AUTOENCODER:
        test_anom_scores, test_targets = models.autoencoder.get_anom_scores(best_model, test_loader, device)
        fp_rate, tp_rate, _ = roc_curve(test_targets, test_anom_scores, pos_label=1)
        roc_auc = roc_auc_score(test_targets, test_anom_scores)
        #plot_roc_curve(self.setup_name, fp_rate, tp_rate, roc_auc)
        return fp_rate, tp_rate, roc_auc
      elif self.model_type == MODEL_TYPES.IDNN:
        raise Exception("Not implemented!")


    def get_normal_and_anomalous_data(self, annotations, batch_size, batch_size_test, model_type):
        if len((set(self.normal_data) & set(self.anomalous_data))) > 0:
          raise Exception("Intersection between normal and anomalous classes should be empty!")

        all_data = import_idmt_traffic_dataset(annotations)

        normal_data = all_data[all_data.vehicle.isin(self.normal_data)]
        #normal_data = self.balance_data_by_vehicle(normal_data)

        anomalous_data = all_data[all_data.vehicle.isin(self.anomalous_data)]
        #anomalous_data = self.balance_data_by_vehicle(anomalous_data)

        train_data, test_data_normal = train_test_split(normal_data, test_size=0.1, shuffle=True, random_state=config.RANDOM_SEED, stratify=normal_data.vehicle)
        train_data, val_data_normal = train_test_split(train_data, test_size=0.02,shuffle=True, random_state=config.RANDOM_SEED, stratify=train_data.vehicle)
        train_data = self.adjust_sample_number_to_batch_size(train_data, batch_size)
        #print(f"training with {len(train_data)} (normal) samples")

        test_data_normal = self.balance_data_by_vehicle(test_data_normal) #balancing test and val data by class to avoid wrong conclusions
        val_data_normal = self.balance_data_by_vehicle(val_data_normal)

        #number_of_normal_test_samples_per_categroy= len(test_data_normal) // len(self.normal_data)
        #print(f"testing with {number_of_normal_test_samples_per_categroy} normal samples")
        #print(f"Validating with {len(val_data_normal)} normal samples")

        #sample same number of anomalous data to test
        #number_anomlous = number_of_normal_test_sampels if number_of_normal_test_sampels < len(anomalous_data) else len(anomalous_data)

        anomalous_test_data = anomalous_data.sample(len(test_data_normal), replace=True, random_state=config.RANDOM_SEED)
        anomalous_val_data = anomalous_data.sample(len(val_data_normal), replace=True, random_state=config.RANDOM_SEED)
        #anomalous_test_data = anomalous_test_data[len(val_data_normal) + 1:] #take first samples from anomalous test data for validation druing trianing
        #print(f"testing with {len(anomalous_data)} anomalous samples")
        #print(f"Validating with {len(anomalous_val_data)} anomalous samples")

        frames = [anomalous_test_data, test_data_normal]
        concatenated_test_data = pd.concat(frames)
        concatenated_test_data.reset_index(drop=True, inplace=True)
        concatenated_test_data = self.adjust_sample_number_to_batch_size(concatenated_test_data, batch_size_test)

        frames_val = [anomalous_val_data, val_data_normal]
        concatenated_val_data = pd.concat(frames_val)
        concatenated_val_data.reset_index(drop=True, inplace=True)
        concatenated_val_data = self.adjust_sample_number_to_batch_size(concatenated_val_data, batch_size_test)

        print(f"Train Data: \n{train_data['vehicle'].value_counts()} \n\nDistribution in Train data: \n{train_data['vehicle'].value_counts(normalize=True)}")
        print(f"Test Data: \n{concatenated_test_data['vehicle'].value_counts()}\n\nDistribution in Test data: \n{concatenated_test_data['vehicle'].value_counts(normalize=True)}")
        print(f"Validation Data: \n{concatenated_val_data['vehicle'].value_counts()} \n\nDistribution in Validation data: \n{concatenated_val_data['vehicle'].value_counts(normalize=True)}")
        normal_train_data = IdmtTrafficDataSet(train_data, SAMPLE_RATE, self.normal_data, model_type=model_type,on_the_fly=True)
        val_data = IdmtTrafficDataSet(concatenated_val_data, SAMPLE_RATE, self.normal_data, model_type=model_type, on_the_fly=True)
        test_data = IdmtTrafficDataSet(concatenated_test_data, SAMPLE_RATE, self.normal_data, model_type=model_type, on_the_fly=True)

        train_loader = torch.utils.data.DataLoader(normal_train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE_VAL, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE_VAL, shuffle=False)

        return train_loader, val_loader, test_loader


    def adjust_sample_number_to_batch_size(self, data, batch_size):
      if len(data) % batch_size == 0:
        print("no data discarded.")
        return data
      else:
        remainder = len(data) % batch_size
        print(str(remainder + 1) + " samples discarded.")
        return data.iloc[remainder + 1:,:]


    def balance_data_by_vehicle(self, pd_data_frame):
      g = pd_data_frame.groupby('vehicle')
      g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
      return g
    
    #def balance_data_normal_anomalous(self, normal_data):


#Klassse ist verantwortlich die Daten zu holen, das model zu trainieren
#output: trainiertes model und loss curve