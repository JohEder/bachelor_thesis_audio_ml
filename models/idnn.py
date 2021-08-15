from math import radians
import torch
from torch import nn
import random
from torch.nn.modules.fold import Unfold

from torch.utils import data
import config


class Idnn(nn.Module):
    def __init__(self, input_dim, mel_bins):
        super(Idnn, self).__init__()
        self.input = input_dim
        self.mel_bins = mel_bins
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, mel_bins)
        )

    def preprocess_input(self, x):
        print(x.shape)
        left = x[:, :, 0:(self.n_segments//2)]
        middle = x[:, :, (self.n_segments//2)]
        right = x[:, :, (self.n_segments//2)+1:]
        context = torch.cat((left, right), dim=-1)
        return context.transpose(1, 2), middle

    def forward(self, x):
        context, middle_frame = self.preporcess(x)
        z = self.encoder(context)
        output = self.decoder(z)
        #print(f"Output shape: {output.shape}")
        return output, middle_frame

    def test_loss(self, x):  # no aggregation as apposed to torch.mse_loss(...) - per "spectrogram" anomaly scores
        x_hat, y = self.forward(x)
        return torch.sum((x_hat - y) ** 2, dim=tuple(range(1, y.dim())))
    
    def preporcess(self, random_frames):
        #print(f"5 Frames: {random_frames.shape}")

        left = random_frames[:, :, 0:(random_frames.shape[2]//2)]
        #print(f"Left: {left.shape}")

        middle_frame = random_frames[:, :, (random_frames.shape[2]//2)]
        #print(f"Middel Frame: {middle_frame.shape}")

        right = random_frames[:, :, (random_frames.shape[2]//2)+1:]
        #print(f"right: {right.shape}")

        context = torch.cat((left, right), dim=-1)
        #print(f"context: {context.shape}")

        context = context.reshape(context.shape[0], -1)
        #print(f"input: {context.shape}")
        return context, middle_frame

def train_epoch(model, train_loader, optimizer, epoch, device):
    print(f"Starting Epoch {epoch}")
    model.train()
    epoch_loss = []
    for batch_index, (data_batch, _, _) in enumerate(train_loader):
        first_frame = random.randint(0, data_batch.shape[3] - (config.NUMBER_OF_FRAMES_IDNN + 1))
        first_frame = first_frame if first_frame > 0 else 0
        last_frame = first_frame + config.NUMBER_OF_FRAMES_IDNN
        data_batch = torch.squeeze(data_batch)
        data_batch = data_batch[:, :,first_frame:last_frame]
        #print(f"Data Batch Shape: {data_batch.shape}") #(32, 128, 5)
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        output_frame, middle_frame = model(data_batch)
        assert middle_frame.shape == output_frame.shape
        loss = mse_loss(middle_frame, output_frame)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if batch_index % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index * len(data_batch), len(train_loader.dataset),100. * batch_index / len(train_loader), loss.item()))
    return epoch_loss


def mse_loss(input, output):
    loss = nn.MSELoss()
    return loss(output, input)

def get_anom_scores(model, val_loader, device, number_of_batches_eval=None, mel_bins=config.N_MELS):
    anom_scores = []
    targets = []
    orig_class_labels = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_number, data in enumerate(val_loader, 0):
            if (number_of_batches_eval != None) and (batch_number > number_of_batches_eval):
                break
            if (batch_number % 50 == 0):
                print(f"Progress: {batch_number}/{len(val_loader)}")
            inputs, target, class_label = data
            inputs = inputs.to(device)
            #print(inputs.shape)
            inputs = patch_batch_framewise(inputs)
            #print(inputs.shape)
            loss_total_current_spec = 0
            for i in range(0, inputs.shape[1] - mel_bins, mel_bins): #iterate through samples
                input_frames = inputs[:,i:i+mel_bins,:]
                #print(input_frames.shape)
                output, middle_frame = model(input_frames) #ith frame gets propagated
                #print(output.shape)
                #print(index)
                assert middle_frame.shape == output.shape
                loss = mse_loss(middle_frame, output)
                loss_total_current_spec += loss.item()

            loss_total_current_spec /= inputs.shape[1] #divide by number of patches
            #print(loss_total_current_spec)
            anom_scores.append(loss_total_current_spec)
            targets.append(target)
            orig_class_labels.append(class_label[0])
    return anom_scores, targets, orig_class_labels

def patch_batch_framewise(data_batch):
    #input of shape (batch_size, channels, mel_filters, frames)
    unfold = Unfold(kernel_size=(data_batch.shape[1], 5), stride=1, padding=0)
    data_batch = unfold(data_batch)
    data_batch = data_batch.transpose(2, 1) #(batch_size, samples, frames_per_sample)
    return data_batch