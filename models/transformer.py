import torch
from torch import nn
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from config import EMBEDDING_SIZE, N_HEADS, DIM_FEED_FORWARD, N_ENCODER_LAYERS, NUMBER_OF_FRAMES
import math
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import config

class TransformerModel(nn.Module):
  def __init__(self, d_model, input_dim, n_heads, dim_feedforward, n_encoder_layers, dropout=0.5):
    super(TransformerModel, self).__init__()
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoder_layers)
    self.patch_embedding = PatchEmbedding(input_dim, d_model)
    self.input_dim = input_dim
    self.d_model = d_model
    self.decoder = Decoder(d_model, 128) #43*128

    self.mask_token = nn.Parameter(torch.randn(d_model, requires_grad=True))

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    #self.patch_embedding.weight.data.uniform_(-initrange, initrange)
    #self.decoder.bias.data.zero_()
    #self.decoder.weight.data.uniform_(-initrange, initrange)
  
  def forward(self, input, mask_index=None):
    #print("Input")
    #print(input)
    embedded = self.patch_embedding(input)
    #print("Embedded")
    #print(embedded)
    embedded_masked, mask_idxs = self.mask_embedded_tokens(embedded, mask_index)
    #print("Embedded masked")
    #print(embedded_masked[0, mask_idxs[0], :])
    pos_encoded_embedded = self.pos_encoder(embedded_masked * math.sqrt(self.d_model))
    #print("Pos encoded")
    #print(pos_encoded_embedded[0, mask_idxs[0], :])
    transformer_out = self.transformer_encoder(pos_encoded_embedded)
    #print("transformer out")
    #print(transformer_out[0, mask_idxs[0], :])
    #masked_part = transformer_out[:, mask_idxs[], :]
    masked_part = self.get_masked_frames(transformer_out, mask_idxs)
    #avg = torch.mean(transformer_out, dim=2, keepdim=True)
    output = self.decoder(masked_part)
    #print("Output")
    #print(output)
    return output, mask_idxs
  
  def get_masked_frames(self, input, mask_idxs):
    #print(f"Input shape {input.shape}")
    masked_frames = []
    for i in range(len(mask_idxs)):
      current_masked = input[i, mask_idxs[i], :]
      masked_frames.append(current_masked)
    masked_frames = torch.stack(masked_frames, dim=0)
    #print(f"Masked frames shape: {masked_frames.shape}")
    return masked_frames

  def mask_embedded_tokens(self, input, specific_mask_idx=None):
    if specific_mask_idx != None:
      assert specific_mask_idx < input.shape[1]
    number_of_specs = input.shape[0]
    input_masked = []
    masks_index_list = []
    for i in range(number_of_specs):
      if specific_mask_idx != None:
        input[i, specific_mask_idx, :] = self.mask_token
        masks_index_list.append([specific_mask_idx])
      else:
        number_of_tokens_tobe_masked = 1 #math.ceil(input.shape[1] * config.MASK_RATIO)
        #print(f"Number of tokens to be masked: {number_of_tokens_tobe_masked}")
        input, mask_idx_for_spec_list = self._mask_token_training(input,number_of_tokens_tobe_masked,i)
        masks_index_list.append(mask_idx_for_spec_list)
      #assert torch.equal(input[i, mask_idx, :], self.mask_token)
      input_masked.append(input[i,:,:]) #maybe just tuples (current_spec_masked, mask_idx)
    input_masked, mask_idxs = torch.stack(input_masked), masks_index_list
    assert input.shape == input_masked.shape
    return input_masked, mask_idxs

  def _mask_token_training(self, input, number_of_tokens, spec_number):
    mask_idxs_for_spec = [random.randint(0, input.shape[1]-1) for i in range(number_of_tokens)]

    for i in range(len(mask_idxs_for_spec)):
      input[spec_number, mask_idxs_for_spec[i], :] = self.mask_token #random.choices([self.mask_token, torch.zeros(self.d_model)], weights=[0.75, 0.25])[0]
    
    return input, mask_idxs_for_spec


class Decoder(nn.Module):
  def __init__(self, transformer_out, out_put_total):
    super(Decoder, self).__init__()
    self.input_dim = transformer_out
    self.mlp = nn.Sequential(
        nn.Linear(in_features=transformer_out, out_features=2*transformer_out),  #evtl 2*d_model
        nn.GELU(),
        nn.Linear(in_features=2*transformer_out, out_features=transformer_out),
        nn.GELU(),
        nn.Linear(in_features=transformer_out, out_features=out_put_total),)
    
  def forward(self, input):
    x = self.mlp(input)
    return x


class PositionalEncoding(nn.Module):
  def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, embedding_dim)
    #print(f"Shape: {pe.shape}")
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    #print(f"Position shape: {position.shape}")
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class PatchEmbedding(nn.Module):
  def __init__(self, input_dim, embedding_dimension):
    super().__init__()
    self.input_dim = input_dim
    self.embedding_layer = nn.Linear(input_dim, embedding_dimension)
  
  def forward(self, input_data):
    embedding = self.embedding_layer(input_data)
    return embedding



def patch_batch(input_batch, number_of_frames):
  #input of shape (batch_size, channels, mel_filters, frames)
  unfold = nn.Unfold(kernel_size=(input_batch.shape[2], number_of_frames), stride=number_of_frames) #patching the spectogram
  unfolded_batch = unfold(input_batch) #(batch_size, features, number_of_patches)
  unfolded_batch = unfolded_batch.transpose(1, 2) #(batch_size, number_of_patches, features)
  return unfolded_batch

def unpatch_batch(input_batch, number_of_frames):
  #(channels, patches, features) (1, 44, 256)
  print(input_batch.shape)
  input_batch = input_batch.transpose(2,1)
  fold = nn.Fold(kernel_size=(input_batch.shape[1], 2), stride=number_of_frames)
  folded_batch = fold(input_batch)
  print(folded_batch.shape)
  return folded_batch

def calculate_l1_loss_masked(input_batch, output_batch, mask_idxs):
  loss_per_batch = []
  for i in range(len(mask_idxs)):
    input_at_masked = input_batch[i, mask_idxs[i], :]
    output_at_masked = output_batch[i, mask_idxs[i], :]
    assert input_at_masked.shape == output_at_masked.shape
    scores = torch.mean(abs((input_at_masked - output_at_masked)))
    loss_per_batch.append(scores)
  loss_per_batch = torch.stack(loss_per_batch, dim=0)
  return loss_per_batch.mean()

def calculate_loss_masked(input_batch, output_batch, mask_idxs):
  #print(input_batch.shape)
  loss_per_batch = []
  for i in range(len(mask_idxs)):
    loss_for_spec = calculate_loss_masked_one_spec(input_batch, output_batch, i, mask_idxs[i])
    loss_per_batch.append(loss_for_spec)
  loss_per_batch = torch.stack(loss_per_batch, dim=0)
  return loss_per_batch.mean()


def calculate_loss_masked_one_spec(input_batch, output_batch, spec_number, spec_list):
  loss_per_spec = []
  for i in range(len(spec_list)):
    input_at_masked = input_batch[spec_number, spec_list[i], :]
    output_at_masked = output_batch[spec_number, spec_list[i], :]
    assert input_at_masked.shape == output_at_masked.shape
    mse = nn.MSELoss(reduction='mean')
    scores = mse(output_at_masked, input_at_masked)
    loss_per_spec.append(scores)
  loss_per_spec = torch.stack(loss_per_spec, dim=0)
  return loss_per_spec.sum()


def train_epoch(model, train_loader, optimizer, epoch, device, scheduler=None, loss_func='l2'):
  print(f"Starting Epoch {epoch}")
  model.train()
  epoch_loss = []
  for batch_index, (data_batch, _, _) in enumerate(train_loader):
    #data_batch = patch_batch(data_batch, NUMBER_OF_FRAMES)
    #print(f"Data Batch shape: {data_batch.shape}") #(32, 1, 64, 87)
    data_batch_patched = torch.split(data_batch, config.NUMBER_OF_FRAMES, dim=3)
    data_batch_patched = data_batch_patched[:-1]
    number_of_patches = len(data_batch_patched)
    #print(f"len sliced batch: {number_of_patches}") #43
    #print(f"sliced batch 0: {data_batch_patched[0].shape}") #(32, 1, 64, 2)
    data_batch_patched = torch.stack(data_batch_patched, dim=1)
    #print(f"Stacked: {data_batch_patched.shape}") #(32, 43, 1, 64, 2)
    data_batch_patched = torch.reshape(data_batch_patched, (config.BATCH_SIZE, number_of_patches, config.N_MELS * config.NUMBER_OF_FRAMES))
    #print(f"Reshaped: {data_batch_patched.shape}") #(32, 43, 128)
    data_batch_patched = data_batch_patched.to(device)
    optimizer.zero_grad()
    output, mask_idxs = model(data_batch_patched) #Mask indexes is a list of lists: Each spectrogram has a list of mask indexes
    #print(f"Mask indexes: {mask_idxs}")
    output = torch.reshape(output, (config.BATCH_SIZE, number_of_patches, config.N_MELS*config.NUMBER_OF_FRAMES))
    #print(f"Output: {output.shape}")

    assert output.shape == data_batch_patched.shape
    if loss_func == 'l2':
      loss = calculate_loss_masked(data_batch_patched, output, mask_idxs)
    else:
      loss = calculate_l1_loss_masked(data_batch_patched, output, mask_idxs)
    #loss_total = calculate_loss_total(data_batch, output)
    #print(f"Loss patches: {loss}\nLoss total: {loss_total}")
    epoch_loss.append(loss.item())
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    current_lr = scheduler.get_last_lr()
    scheduler.step()
    if batch_index % 30 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(epoch, batch_index * len(data_batch), len(train_loader.dataset),100. * batch_index / len(train_loader), loss.item(), current_lr))
  return epoch_loss


def get_anom_scores(model, data_loader, device, number_of_batches_eval=None, loss_func='l2'):
  #currently the batch size for evaluation needs to be 1
  total_anom_scores = []
  total_targets = []
  original_class_labels = []
  reconstructions = []
  model.to(device)
  model.eval()
  with torch.no_grad():
    for batch_number, data in enumerate(data_loader, 0):
      #print(f"Batch Number: {batch_number}")
      if (number_of_batches_eval != None) and (batch_number > number_of_batches_eval):
        break
      if (batch_number % 30 == 0):
        print(f"Progress: {batch_number}/{len(data_loader)}")
      input, target, class_label = data
      #print(f"Data Batch shape: {input.shape}") #(1, 1, 64, 87)
      input_patched = torch.split(input, config.NUMBER_OF_FRAMES, dim=3)
      input_patched = input_patched[:-1]
      number_of_patches = len(input_patched)
      #print(f"len sliced batch: {number_of_patches}") #43
      #print(f"sliced batch 0: {input_patched[0].shape}") #(1, 1, 64, 2)
      input_patched = torch.stack(input_patched, dim=1)
      #print(f"Stacked: {input_patched.shape}") #(1, 43, 1, 64, 2)
      input_patched = torch.reshape(input_patched, (config.BATCH_SIZE_VAL, number_of_patches, config.N_MELS * config.NUMBER_OF_FRAMES))
      #print(f"Reshaped: {input_patched.shape}") #(1, 43, 128)
      input_patched = input_patched.to(device)
      
      #print(patched_input.shape) #(n_spectograms, n_patches, features)
      #every patch needs to be masked once and the masked loss calculated added and divided by number of patches
      loss_total_current_spec = 0
      reconstructed_patches = []
      for i in range(input_patched.shape[1]): #iterate through patches
        output, index = model(input_patched, i) #patch i gets masked

        assert index[0][0] == i
        output = torch.reshape(output, (1, number_of_patches, config.N_MELS*config.NUMBER_OF_FRAMES))
        #print(f"Output Shape: {output.shape}")
        input_at_masked = input_patched[0, i, :]
        output_at_masked = output[0, i, :] #batch size 1
        #print(output_at_masked)
        if loss_func == 'l2':
          loss = calculate_loss_masked(input_patched, output, index) # last argument (sum) does not make a difference for batch size 1
        else:
          loss = calculate_l1_loss_masked(input_patched, output, index)
        loss_total_current_spec += loss.item()

        output_at_masked = torch.reshape(output_at_masked, (1, 1, config.N_MELS, config.NUMBER_OF_FRAMES))
        reconstructed_patches.append(output_at_masked)
      
      #print(reconstructed_patches[3])
      #print(reconstructed_patches[20])
      #assert torch.equal(reconstructed_patches[6], reconstructed_patches[20])
      #print(loss_total_current_spec)

      #loss_total_current_spec /= input_patched.shape[1] #divide by number of patches
      #print(loss_total_current_spec)
      total_anom_scores.append(loss_total_current_spec) #coverting to numpy for processing with scikit
      total_targets.append(target)
      original_class_labels.append(class_label[0])
      #print(len(reconstructed_patches))
      #print(reconstructed_patches[0].shape)
      reconstruction = torch.cat(reconstructed_patches, dim=3)
      #print(f"Reconstruction: {reconstruction.shape}")
      #reconstruction = torch.reshape(reconstruction, (1, 1, config.N_MELS, -1))
      #print(reconstruction.shape)

      input, reconstruction = torch.squeeze(input), torch.squeeze(reconstruction)
      reconstructions.append((input, reconstruction))
      #print(input.shape)
      #print(reconstruction.shape)
      #assert input.shape == reconstruction.shape
    #assert (not torch.equal(reconstruction[0], reconstruction[2]))
    return total_anom_scores, total_targets, original_class_labels, reconstructions


def get_reconstruction(model, dataloader):
  reconstructed_input = torch.zeros(128, 88)
  return