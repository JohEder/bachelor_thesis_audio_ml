from ast import dump
from matplotlib.pyplot import cla
import torch
from torch import nn
from config import NUMBER_OF_FRAMES_AE
import config
import random

class AutoEncoder(nn.Module):
  def __init__(self, input_dim):
    super().__init__()

    self.encoder = nn.Sequential(
        #Dense Layer 1
        nn.Linear(in_features=input_dim, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Dense Layer 2
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Dense Layer 3
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Dense Layer 4
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #BottleNeck Layer
        nn.Linear(in_features=128, out_features=8),
        nn.ReLU(),
        nn.BatchNorm1d(8)
    )
    self.decoder = nn.Sequential(
        #Dense Layer 5
        nn.Linear(in_features=8, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Dense Layer 6
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Dense Layer 7
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Dense Layer 8
        nn.Linear(in_features=128, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(128),

        #Output Layer
        nn.Linear(in_features=128, out_features=input_dim),
    )

  def forward(self, input_data):
    z = self.encoder(input_data)
    output = self.decoder(z)
    return output

def patch_batch_ae(input_batch, stride=NUMBER_OF_FRAMES_AE):
  #input of shape (batch_size, channels, mel_filters, frames)
  unfold = nn.Unfold(kernel_size=(input_batch.shape[2], NUMBER_OF_FRAMES_AE), stride=stride) #patching the spectogram
  unfolded_batch = unfold(input_batch) #(batch_size, features, number_of_patches)
  unfolded_batch = unfolded_batch.transpose(1, 2) #(batch_size, number_of_patches, features)
  return unfolded_batch

def train_epoch(model, train_loader, optimizer, epoch, device):
    print(f"Starting Epoch {epoch}")
    model.train()
    epoch_loss = []
    for batch_index, (data_batch, _, _) in enumerate(train_loader):
        #data_batch = patch_batch_ae(data_batch)
        #print(data_batch.shape)
        #print(batch_index)
        sliced_batch = torch.split(data_batch, 5, dim=3)
        sliced_batch = sliced_batch[:-1]
        #print(sliced_batch[-1].shape)
        #print(f"len sliced batch: {len(sliced_batch)}")
        patch_num = random.randint(0, len(sliced_batch)-1)
        #print(f"sliced batch[patch_num]: {sliced_batch[patch_num].shape}")
        data_batch = sliced_batch[patch_num]
        data_batch = torch.reshape(data_batch, (config.BATCH_SIZE, config.N_MELS*config.NUMBER_OF_FRAMES_AE))
        #print(data_batch.shape)
        #data_batch = data_batch[:,1,:]
        #print(data_batch.shape)
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        output = model(data_batch)
        assert output.shape == data_batch.shape
        # Calculate loss
        loss = mse_loss(output, data_batch)
        loss.backward()                 
        optimizer.step()
        epoch_loss.append(loss.item())
        if batch_index % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index * len(data_batch), len(train_loader.dataset),100. * batch_index / len(train_loader), loss.item()))
    return epoch_loss

def get_anom_scores(model, val_loader, device, number_of_batches_eval=None):
  anom_scores = []
  targets = []
  class_labels = []
  reconstructions = []
  model.to(device)
  model.eval()
  with torch.no_grad():
    for batch_number, data in enumerate(val_loader, 0):
        if (number_of_batches_eval != None) and (batch_number > number_of_batches_eval):
            break
        if (batch_number % 50 == 0):
            print(f"Progress: {batch_number}/{len(val_loader)}")
        inputs, target, class_label = data
        #print(f"Inputs Shape: {inputs.shape}")
        sliced_batch = torch.split(inputs, config.NUMBER_OF_FRAMES_AE, dim=3)
        sliced_batch = sliced_batch[:-1]
        #print(f"len sliced batch: {len(sliced_batch)}")
        #print(f"sliced batch 0: {sliced_batch[0].shape}")
        #print(inputs.shape) #torch.Size([1, 1, 128, 88])
        #inputs_patched = patch_batch_ae(inputs)
        #print(inputs.shape)
        loss_total_current_spec = 0
        reconstructed_frames = []
        for i in range(len(sliced_batch)): #iterate through patches #inputs_patched.shape[1]
          input_frames = sliced_batch[i]
          input_frames_flat = torch.reshape(input_frames, (1, config.N_MELS*config.NUMBER_OF_FRAMES_AE))
          #print(input_frames.shape)
          input_frames_flat, input_frames = input_frames_flat.to(device), input_frames.to(device)
          output= model(input_frames_flat) #ith frame gets propagated
          output = torch.reshape(output, (1, 1, config.N_MELS, config.NUMBER_OF_FRAMES_AE))
          reconstructed_frames.append(output)
          #print(index)
          #print(f"input shape: {input_frames.shape}\nOutput Shape: {output.shape}")
          assert input_frames.shape == output.shape
          loss = mse_loss(input_frames, output)
          #print(class_label)
          #print(loss)
          loss_total_current_spec += loss.item()
        

        reconstructed_spec = torch.cat(reconstructed_frames, dim=3)
        #print(f"Reconstructed Spec Shape: {reconstructed_spec.shape}")
        #reconstructed_spec = reconstructed_spec.transpose(2, 1)

        inputs, reconstructed_spec = torch.squeeze(inputs), torch.squeeze(reconstructed_spec)
        #assert reconstructed_spec.shape == inputs_patched.shape
        #inputs, reconstructed_spec = torch.reshape(inputs, (config.N_MELS, -1)), torch.reshape(reconstructed_spec, (config.N_MELS, -1))
        #print(inputs.shape)
        #print(reconstructed_spec.shape)
        #assert reconstructed_spec.shape == inputs.shape
        reconstructions.append((inputs, reconstructed_spec))

        loss_total_current_spec /= inputs.shape[1] #divide by number of patches
        #print(loss_total_current_spec)
        anom_scores.append(loss_total_current_spec)
        targets.append(target)
        class_labels.append(class_label[0])
    return anom_scores, targets, class_labels, reconstructions

def mse_loss(input, output):
    loss = nn.MSELoss(reduction='mean')
    return loss(output, input)