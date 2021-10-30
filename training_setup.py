
import datetime
import math
from operator import pos
from os import replace

from scipy.sparse.construct import rand
import models.autoencoder
import models.idnn
from torch import nn

from sklearn.utils import shuffle
import models.transformer
from models.transformer import TransformerModel
from torch.nn.modules import module
import config
from config import AUDIO_DIR, NUMBER_OF_FRAMES_AE, NUMBER_OF_FRAMES_IDNN, N_MELS, all_annotations_txt, BATCH_SIZE, SAMPLE_RATE, BATCH_SIZE_VAL, MODEL_TYPES, EMBEDDING_SIZE, N_HEADS, DIM_FEED_FORWARD, N_ENCODER_LAYERS, NUMBER_OF_FRAMES
from import_idmt_traffic_dataset import *
import pandas as pd
from datasets.idmt_traffic import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from utils import save_hyperparams, save_model
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AdamW
import pytorch_model_summary as pms
import copy
import random
import numpy as np


class TrainingSetup():
    def __init__(self, normal_data, anomalous_data, setup_type=config.SETUP_TYPES.VEHICLES):
        self.normal_data = normal_data
        self.anomalous_data = anomalous_data
        self.setup_name = '_'.join(normal_data) +'_anom_'+ '_'.join(anomalous_data)
        self.annotations = config.all_annotations_txt
        self.setup_type = setup_type

    def __str__(self):
        return f'Normal classes: {self.normal_data} Anomalous:{self.anomalous_data}'

    def run(self, model_type, number_of_runs=1, number_mel_bins=config.N_MELS, loss_function='l2', model_save=True):
      auc_roc_scores = []
      for i in range(number_of_runs):
        current_seed = config.RANDOM_SEEDS[i]
        random.seed(current_seed)
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        model_name = self.setup_name
        roc_auc_best = -1
        best_model = None
        training_start = datetime.datetime.now()
        losses = []
        print(f"\nrunning {i + 1}. Run of Setup: {self.normal_data} : {model_type}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == MODEL_TYPES.TRANSFORMER:
          train_loader, val_loader, test_loader = self.get_normal_and_anomalous_data( self.annotations, BATCH_SIZE, BATCH_SIZE_VAL, current_seed, number_mel_bins)
          #training
          transformer = TransformerModel(EMBEDDING_SIZE, number_mel_bins*config.NUMBER_OF_FRAMES, N_HEADS, DIM_FEED_FORWARD, N_ENCODER_LAYERS)
          LEARNING_RATE = 0.0001
          WEIGHT_DECAY = 0.0001
          optimizer = AdamW(transformer.parameters(), lr=LEARNING_RATE) #torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE) 
          EPOCHS = config.EPOCHS_TF
          total_steps = len(train_loader) * EPOCHS
          warm_up_steps = math.ceil(total_steps * 0.1)
          scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)
          print(f"Total Steps: {total_steps}, Warm up steps: {warm_up_steps}, Ratio: {warm_up_steps / total_steps}")
          transformer.to(device)
          transformer.train() #mode
          for epoch in range(1, EPOCHS + 1):
            losses_epoch = models.transformer.train_epoch(transformer, train_loader, optimizer, epoch, device, scheduler=scheduler, loss_func=loss_function)
            val_anom_scores, val_targets, _, _ = models.transformer.get_anom_scores(transformer, val_loader, device, loss_func=loss_function) #batch size in evalution is only one
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
          summary = pms.summary(transformer, torch.ones(BATCH_SIZE, 43, config.N_MELS*2).to(device))
        elif model_type == MODEL_TYPES.AUTOENCODER:
          train_loader, val_loader, test_loader = self.get_normal_and_anomalous_data( self.annotations, BATCH_SIZE, BATCH_SIZE_VAL, current_seed, number_mel_bins)
          #print(next(iter(train_loader)).shape)
          LEARNING_RATE = 0.001
          EPOCHS = config.EPOCHS_AE
          INPUT_DIM = number_mel_bins * NUMBER_OF_FRAMES_AE
          print(f"Input dim: {INPUT_DIM}")
          autoencoder = models.autoencoder.AutoEncoder(input_dim=INPUT_DIM)
          optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
          total_steps = len(train_loader) * EPOCHS
          warm_up_steps = 0

          autoencoder.to(device)
          autoencoder.train() #mode
          for epoch in range(1, EPOCHS + 1):
            losses_epoch = models.autoencoder.train_epoch(autoencoder, train_loader, optimizer, epoch, device)
            val_anom_scores, val_targets, _, _ = models.autoencoder.get_anom_scores(autoencoder, val_loader, device) #batch size in evalution is only one
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
          summary = pms.summary(autoencoder, torch.ones(BATCH_SIZE, INPUT_DIM).to(device))

        elif model_type == MODEL_TYPES.IDNN:
          train_loader, val_loader, test_loader = self.get_normal_and_anomalous_data( self.annotations, BATCH_SIZE, BATCH_SIZE_VAL, current_seed, number_mel_bins)
          input_sample, label, _  = (next(iter(train_loader)))
          input_shape = input_sample.shape #[32, 1, 128, 44]
          #print(input_shape)
          LEARNING_RATE = 0.001
          EPOCHS = config.EPOCHS_IDNN
          idnn = models.idnn.Idnn(input_dim=number_mel_bins * (config.NUMBER_OF_FRAMES_IDNN - 1), mel_bins=number_mel_bins)
          optimizer = torch.optim.Adam(idnn.parameters(), lr=LEARNING_RATE)
          total_steps = len(train_loader) * EPOCHS
          warm_up_steps = 0
          idnn.to(device)
          idnn.train()
          for epoch in range(1, EPOCHS + 1):
            losses_epoch = models.idnn.train_epoch(idnn, train_loader, optimizer, epoch, device)
            val_anom_scores, val_targets, _, _ = models.idnn.get_anom_scores(idnn, val_loader, device, mel_bins=number_mel_bins)
            roc_auc = roc_auc_score(val_targets, val_anom_scores)
            losses += losses_epoch
            if len(val_loader) > 50 and roc_auc > roc_auc_best:
              best_model = copy.deepcopy(idnn)
              roc_auc_best = roc_auc
            else: 
              best_model = copy.deepcopy(idnn)
              roc_auc_best = roc_auc
            print(f"Evaluation ROC Score in epoch {epoch} is {roc_auc}, Best ROC Score is:{roc_auc_best}")
          training_finished = datetime.datetime.now()
          total_training_time = training_finished - training_start
          summary = pms.summary(idnn, torch.ones(BATCH_SIZE, number_mel_bins, NUMBER_OF_FRAMES_IDNN).to(device))
        #evaluation
        fp_rate, tp_rate, roc_auc, scores_classes, orig_recons = self.evaluate_model(best_model, test_loader, device, model_type, number_mel_bins)
        if model_save:
          save_model(self.setup_name + '_' + str(model_type), best_model)
        print(f"ROC AUC of Model {model_name} is {roc_auc}!")
        auc_roc_scores.append(roc_auc)
        #plot_and_save_loss_curve(self.setup_name, losses)
        save_hyperparams(model_type, model_name, total_training_time, optimizer, LEARNING_RATE, EPOCHS, self.normal_data, self.anomalous_data, roc_auc, summary, weight_decay="", total_steps=total_steps, warm_up_steps=warm_up_steps, mel_bins=number_mel_bins)
      return auc_roc_scores, losses, fp_rate, tp_rate, roc_auc, scores_classes, orig_recons


    def evaluate_model(self, best_model, test_loader, device, model_type, mel_bins):
      #model = load_model(model_name)
      if model_type == MODEL_TYPES.TRANSFORMER:
        test_anom_scores, test_targets, original_class_labels, orig_recons = models.transformer.get_anom_scores(best_model, test_loader, device)
        fp_rate, tp_rate, _ = roc_curve(test_targets, test_anom_scores, pos_label=1)
        roc_auc = roc_auc_score(test_targets, test_anom_scores)
        #plot_roc_curve(self.setup_name, fp_rate, tp_rate, roc_auc)
        return fp_rate, tp_rate, roc_auc, (test_anom_scores, original_class_labels), orig_recons
      elif model_type == MODEL_TYPES.AUTOENCODER:
        test_anom_scores, test_targets, orig_class_labels, orig_recons = models.autoencoder.get_anom_scores(best_model, test_loader, device)
        fp_rate, tp_rate, _ = roc_curve(test_targets, test_anom_scores, pos_label=1)
        roc_auc = roc_auc_score(test_targets, test_anom_scores)
        #plot_roc_curve(self.setup_name, fp_rate, tp_rate, roc_auc)
        return fp_rate, tp_rate, roc_auc, (test_anom_scores, orig_class_labels), orig_recons
      elif model_type == MODEL_TYPES.IDNN:
        test_anom_scores, test_targets, orig_class_labels, orig_recons = models.idnn.get_anom_scores(best_model, test_loader, device, mel_bins=mel_bins)
        fp_rate, tp_rate, _ = roc_curve(test_targets, test_anom_scores, pos_label=1)
        roc_auc = roc_auc_score(test_targets, test_anom_scores)
        return fp_rate, tp_rate, roc_auc, (test_anom_scores, orig_class_labels), orig_recons


    def get_normal_and_anomalous_data(self, annotations, batch_size, batch_size_test, current_seed, mel_bins):
        if len((set(self.normal_data) & set(self.anomalous_data))) > 0:
          raise Exception("Intersection between normal and anomalous classes should be empty!")

        all_data = import_idmt_traffic_dataset(annotations)
        all_data = all_data[all_data.is_background | all_data.vehicle != 'None'] #filter sounds that are not background and not vehicles
        high_quality = all_data[all_data.microphone == 'SE']
        print(f"High quality total data: {len(high_quality)}")


        if self.setup_type == config.SETUP_TYPES.WEATHER:
          type = 'weather'
          high_quality = high_quality[high_quality.weather != 'None']
          row = config.ROW_CONDITIONS
        elif self.setup_type == config.SETUP_TYPES.VELOCITY:
          type = 'speed_kmh'
          high_quality = high_quality[high_quality.speed_kmh != 'UNK']
          row = config.ROW_VELOCIIES
        elif self.setup_type == config.SETUP_TYPES.VEHICLES:
          type = 'vehicle'
          row = config.ROW_VEHICLES


        normal_data = high_quality[high_quality[type].isin(self.normal_data)]
        anomalous_data = high_quality[high_quality[type].isin(self.anomalous_data)]
        train_data, test_data_normal = train_test_split(normal_data, test_size=0.2, shuffle=True, random_state=current_seed, stratify=normal_data[type])
        train_data, val_data_normal = train_test_split(train_data, test_size=0.1,shuffle=True, random_state=current_seed, stratify=train_data[type])

        #anomalous_data = self.balance_data_by_vehicle(anomalous_data)
        train_data = self.adjust_sample_number_to_batch_size(train_data, batch_size)
        #print(f"training with {len(train_data)} (normal) samples")

        #test_data_normal = self.balance_data_by_vehicle(test_data_normal) #balancing test and val data by class to avoid wrong conclusions
        #val_data_normal = self.balance_data_by_vehicle(val_data_normal)

        #number_of_normal_test_samples_per_categroy= len(test_data_normal) // len(self.normal_data)
        #print(f"testing with {number_of_normal_test_samples_per_categroy} normal samples")
        #print(f"Validating with {len(val_data_normal)} normal samples")

        #sample same number of anomalous data to test
        #number_anomlous = number_of_normal_test_sampels if number_of_normal_test_sampels < len(anomalous_data) else len(anomalous_data)
        #test_data_normal = self.balance_data_by_vehicle(test_data_normal)

        #print(anomalous_data[type].value_counts())
        min_class = min(anomalous_data[type].value_counts())
        anomalous_test_data = anomalous_data.groupby(type).sample(min_class, random_state=current_seed)
        #anomalous_data = anomalous_data.drop(anomalous_test_data.index)
        anomalous_val_data = anomalous_data.groupby(type).sample(min_class, random_state=current_seed)
        #anomalous_test_data = anomalous_test_data[len(val_data_normal) + 1:] #take first samples from anomalous test data for validation druing trianing
        print(f"testing with {len(anomalous_test_data)} anomalous samples")
        print(f"length of normal test data {len(test_data_normal)}")
        #print(f"Validating with {len(anomalous_val_data)} anomalous samples")
        if len(test_data_normal) > len(anomalous_test_data):
          test_data_normal = test_data_normal.groupby(type).sample(len(anomalous_test_data) // len(self.normal_data),random_state=current_seed )
        else:
          anomalous_test_data = anomalous_test_data.groupby(type).sample(len(test_data_normal) // len(self.anomalous_data), random_state=current_seed)

        if len(val_data_normal) > len(anomalous_val_data):
          val_data_normal = val_data_normal.groupby(type).sample(len(anomalous_val_data) // len(self.normal_data),random_state=current_seed, replace=True )
        else:
          anomalous_val_data = anomalous_val_data.groupby(type).sample(len(val_data_normal) // len(self.anomalous_data), random_state=current_seed)

        frames = [anomalous_test_data, test_data_normal]
        concatenated_test_data = pd.concat(frames)
        concatenated_test_data.reset_index(drop=True, inplace=True)
        concatenated_test_data = self.adjust_sample_number_to_batch_size(concatenated_test_data, batch_size_test)

        frames_val = [anomalous_val_data, val_data_normal]
        concatenated_val_data = pd.concat(frames_val)
        concatenated_val_data.reset_index(drop=True, inplace=True)
        concatenated_val_data = self.adjust_sample_number_to_batch_size(concatenated_val_data, batch_size_test)

        #concatenated_test_data = self.balance_data_by_vehicle(concatenated_test_data)
        #concatenated_val_data = self.balance_data_by_vehicle(concatenated_val_data)

        print(f"Train Data: \n{train_data[type].value_counts()} \n\nDistribution in Train data: \n{train_data[type].value_counts(normalize=True)}")
        print(f"Test Data: \n{concatenated_test_data[type].value_counts()}\n\nDistribution in Test data: \n{concatenated_test_data[type].value_counts(normalize=True)}")
        print(f"Validation Data: \n{concatenated_val_data[type].value_counts()} \n\nDistribution in Validation data: \n{concatenated_val_data[type].value_counts(normalize=True)}")
        normal_train_data = IdmtTrafficDataSet(train_data, SAMPLE_RATE, self.normal_data, self.anomalous_data,row, mel_bins, on_the_fly=True)
        val_data = IdmtTrafficDataSet(concatenated_val_data, SAMPLE_RATE, self.normal_data,self.anomalous_data,row, mel_bins, on_the_fly=True)
        test_data = IdmtTrafficDataSet(concatenated_test_data, SAMPLE_RATE, self.normal_data,self.anomalous_data,row, mel_bins, on_the_fly=True)

        train_loader = torch.utils.data.DataLoader(normal_train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
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
        print(len(data.iloc[remainder + 1:,:]))
        #assert len(data.iloc[remainder + 1:,:]) % batch_size == 0
        return data.iloc[remainder + 1:,:]


    def balance_data_by_vehicle(self, pd_data_frame):
      g = pd_data_frame.groupby('vehicle')
      g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
      return g
    