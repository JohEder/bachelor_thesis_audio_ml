import datetime
import torch
from torch import nn
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from config import EMBEDDING_SIZE, input_dim, N_HEADS, DIM_FEED_FORWARD, N_ENCODER_LAYERS, NUMBER_OF_FRAMES
import math
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class TransformerModel(nn.Module):
  def __init__(self, d_model, input_dim, n_heads, dim_feedforward, n_encoder_layers, dropout=0.5):
    super(TransformerModel, self).__init__()
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoder_layers)
    self.patch_embedding = PatchEmbedding(input_dim, d_model)
    self.input_dim = input_dim
    self.d_model = d_model
    self.decoder = Decoder(d_model, input_dim)

    self.mask_token = nn.Parameter(torch.randn(d_model, requires_grad=True))

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    #self.patch_embedding.weight.data.uniform_(-initrange, initrange)
    #self.decoder.bias.data.zero_()
    #self.decoder.weight.data.uniform_(-initrange, initrange)
  
  def forward(self, input, mask_index=None):
    embedded = self.patch_embedding(input) #* math.sqrt(self.input_dim) #is scaling necessary? yes, otherwise values are incredibly small
    embedded_masked, mask_idxs = self.mask_embedded_tokens(embedded)
    pos_encoded_embedded = self.pos_encoder(embedded_masked)
    transformer_out = self.transformer_encoder(pos_encoded_embedded)
    output = self.decoder(transformer_out)
    return output, mask_idxs

  def mask_embedded_tokens(self, input, specific_mask_idx=None):
    if specific_mask_idx != None:
      assert specific_mask_idx < input.shape[1]
    number_of_specs = input.shape[0]
    input_masked = []
    masks_index_list = []
    for i in range(number_of_specs):
      mask_idx = specific_mask_idx if specific_mask_idx != None else random.randint(0, input.shape[1]-1)

      input[i, mask_idx, :] = self.mask_token

      input_masked.append(input[i,:,:]) #maybe just tuples (current_spec_masked, mask_idx)
      masks_index_list.append(torch.as_tensor(mask_idx))

    return torch.stack(input_masked), torch.stack(masks_index_list)


class Decoder(nn.Module):
  def __init__(self, transformer_out, out_put_total):
    super(Decoder, self).__init__()
    self.input_dim = transformer_out
    self.mlp = nn.Sequential(
        nn.Linear(in_features=transformer_out, out_features=transformer_out),  #evtl 2*d_model
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

def calculate_loss_masked(input_batch, output_batch, mask_idxs, sum_up):
  #print(input_batch.shape)
  loss_per_batch = []
  for i in range(len(mask_idxs)):
    input_at_masked = input_batch[i, mask_idxs[i], :]
    output_at_masked = output_batch[i, mask_idxs[i], :]
    scores = torch.mean((input_at_masked - output_at_masked) ** 2, dim=tuple(range(1, output_at_masked.dim()))) #brauche ich die dimension? ist glaube ich immer 1
    loss_per_batch.append(scores)
  loss_per_batch = torch.stack(loss_per_batch, dim=0)
  return loss_per_batch.mean()


def train_epoch(model, train_loader, optimizer, epoch, device, scheduler=None):
  print(f"Starting Epoch {epoch}")
  model.train()
  epoch_loss = []
  for batch_index, (data_batch, _) in enumerate(train_loader):
    #print(data_batch.shape)
    data_batch = patch_batch(data_batch, NUMBER_OF_FRAMES)
    data_batch = data_batch.to(device)

    optimizer.zero_grad()
    output, mask_idxs = model(data_batch)
    loss = calculate_loss_masked(data_batch, output, mask_idxs, True)
    #loss_total = calculate_loss_total(data_batch, output)
    #print(f"Loss patches: {loss}\nLoss total: {loss_total}")
    epoch_loss.append(loss.item())
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    current_lr = scheduler.get_last_lr()
    scheduler.step()
    if batch_index % 100 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(epoch, batch_index * len(data_batch), len(train_loader.dataset),100. * batch_index / len(train_loader), loss.item(), current_lr))
  return epoch_loss


def get_anom_scores(model, data_loader, device, number_of_batches_eval=None):
  #currently the batch size for evaluation needs to be 1
  total_anom_scores = []
  total_targets = []
  model.to(device)
  model.eval()
  with torch.no_grad():
    for batch_number, data in enumerate(data_loader, 0):
      if (number_of_batches_eval != None) and (batch_number > number_of_batches_eval):
        break
      if (batch_number % 30 == 0):
        print(f"Progress: {batch_number}/{len(data_loader)}")
      inputs, target = data
      inputs = inputs.to(device)
      #print(inputs.shape)
      inputs = patch_batch(inputs, NUMBER_OF_FRAMES)
      #print(inputs.shape) #(n_spectograms, n_patches, features)
      #every patch needs to be masked once and the masked loss calculated added and divided by number of patches
      loss_total_current_spec = 0
      for i in range(inputs.shape[1]): #iterate through patches
        output, index = model(inputs, i) #patch i gets masked
        #print(output)
        #print(index)
        loss = calculate_loss_masked(inputs, output, index, True) # last argument (sum) does not make a difference for batch size 1
        loss_total_current_spec += loss.item()
      
      loss_total_current_spec /= inputs.shape[1] #divide by number of patches
      #print(loss_total_current_spec)
      total_anom_scores.append(loss_total_current_spec) #coverting to numpy for processing with scikit
      total_targets.append(target)
    return total_anom_scores, total_targets