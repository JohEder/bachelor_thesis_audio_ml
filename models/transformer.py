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
    #print("Input")
    #print(input)
    embedded = self.patch_embedding(input) #* math.sqrt(self.input_dim) #is scaling necessary? yes, otherwise values are incredibly small
    #print("Embedded")
    #print(embedded)
    embedded_masked, mask_idxs = self.mask_embedded_tokens(embedded, mask_index)
    #print("Embedded masked")
    #print(embedded_masked[0, mask_idxs[0], :])
    pos_encoded_embedded = self.pos_encoder(embedded_masked)
    #print("Pos encoded")
    #print(pos_encoded_embedded[0, mask_idxs[0], :])
    transformer_out = self.transformer_encoder(pos_encoded_embedded)
    #print("transformer out")
    #print(transformer_out[0, mask_idxs[0], :])
    output = self.decoder(transformer_out)
    #print("Output")
    #print(output)
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
      assert torch.equal(input[i, mask_idx, :], self.mask_token)
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

def calculate_loss_masked(input_batch, output_batch, mask_idxs, sum_up):
  #print(input_batch.shape)
  loss_per_batch = []
  for i in range(len(mask_idxs)):
    input_at_masked = input_batch[i, mask_idxs[i], :]
    output_at_masked = output_batch[i, mask_idxs[i], :]
    assert input_at_masked.shape == output_at_masked.shape
    scores = torch.mean((input_at_masked - output_at_masked) ** 2, dim=tuple(range(1, output_at_masked.dim()))) #brauche ich die dimension? ist glaube ich immer 1
    loss_per_batch.append(scores)
  loss_per_batch = torch.stack(loss_per_batch, dim=0)
  return loss_per_batch.mean()


def train_epoch(model, train_loader, optimizer, epoch, device, scheduler=None, loss_func='l2'):
  print(f"Starting Epoch {epoch}")
  model.train()
  epoch_loss = []
  for batch_index, (data_batch, _, _) in enumerate(train_loader):
    data_batch = patch_batch(data_batch, NUMBER_OF_FRAMES)
    data_batch = data_batch.to(device)
    optimizer.zero_grad()
    output, mask_idxs = model(data_batch)
    if loss_func == 'l2':
      loss = calculate_loss_masked(data_batch, output, mask_idxs, True)
    else:
      loss = calculate_l1_loss_masked(data_batch, output, mask_idxs)
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


def train_epoch_new(model, train_loader, optimizer, epoch, device, scheduler=None):
  print(f"Starting Epoch {epoch}")
  model.train()
  epoch_loss = []
  for batch_index, (data_batch, _, _) in enumerate(train_loader):
    print(data_batch.shape)
    data_batch = torch.squeeze(data_batch)
    print(data_batch.shape)
    data_batch = torch.stack(torch.split(data_batch, NUMBER_OF_FRAMES, dim=2))
    print(data_batch.shape)
    data_batch = torch.reshape(data_batch, (32, 44, 128, 2))
    print(data_batch.shape)
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
    if batch_index % 30 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(epoch, batch_index * len(data_batch), len(train_loader.dataset),100. * batch_index / len(train_loader), loss.item(), current_lr))
  return epoch_loss

def get_anom_scores_new(model, data_loader, device, number_of_batches_eval=None):
  #currently the batch size for evaluation needs to be 1
  total_anom_scores = []
  total_targets = []
  original_class_labels = []
  reconstructions = []
  model.to(device)
  model.eval()
  with torch.no_grad():
    for batch_number, data in enumerate(data_loader, 0):
      if (number_of_batches_eval != None) and (batch_number > number_of_batches_eval):
        break
      if (batch_number % 30 == 0):
        print(f"Progress: {batch_number}/{len(data_loader)}")
      input, target, class_label = data
      input = input.to(device)
      #print(f"inputs shape: {input.shape}")
      #print(inputs.shape)
      patched_input = torch.stack(torch.split(input, NUMBER_OF_FRAMES))
      #print(patched_input.shape) #(n_spectograms, n_patches, features)
      #every patch needs to be masked once and the masked loss calculated added and divided by number of patches
      loss_total_current_spec = 0
      reconstructed_patches = []
      original_inputs = []
      for i in range(patched_input.shape[0]): #iterate through patches
        output, index = model(patched_input, i) #patch i gets masked
        assert index == i
        #print(f"Output Shape: {output.shape}") #(1, 22, 32)
        output_at_masked = output[i, :, :]
        #input_at_masked = torch.squeeze(patched_input[0, i, :])
        #print(output_at_masked.shape) #(1,1,32)
        #output_at_masked = torch.squeeze(output_at_masked)
        reconstructed_patches.append(output_at_masked)
        #original_inputs.append(input_at_masked)
        loss = calculate_loss_masked(patched_input, output, index, True) # last argument (sum) does not make a difference for batch size 1
        loss_total_current_spec += loss.item()
      
      loss_total_current_spec /= patched_input.shape[1] #divide by number of patches
      #print(loss_total_current_spec)
      total_anom_scores.append(loss_total_current_spec) #coverting to numpy for processing with scikit
      total_targets.append(target)
      original_class_labels.append(class_label[0])
      #print(len(reconstructed_patches))
      #print(reconstructed_patches[0].shape)
      reconstruction = torch.stack(reconstructed_patches)
      print(reconstruction.shape)
      reconstruction = reconstruction.reshape(1, 1, 128, 88)
      reconstruction = torch.reshape(reconstruction, (config.N_MELS, -1))
      #orig_input = torch.reshape(torch.stack(original_inputs), (config.N_MELS, -1))
      input = torch.squeeze(input)
      reconstructions.append((input, reconstruction))
      #print(input.shape)
      #print(reconstruction.shape)
      assert input.shape == reconstruction.shape
    return total_anom_scores, total_targets, original_class_labels, reconstructions



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
      input = input.to(device)
      #print(f"inputs shape: {input.shape}")
      #print(inputs.shape)
      patched_input = patch_batch(input, NUMBER_OF_FRAMES)
      #print(patched_input.shape) #(n_spectograms, n_patches, features)
      #every patch needs to be masked once and the masked loss calculated added and divided by number of patches
      loss_total_current_spec = 0
      reconstructed_patches = []
      original_inputs = []
      for i in range(patched_input.shape[1]): #iterate through patches
        output, index = model(patched_input, i) #patch i gets masked
        assert index == i
        #print(f"Output Shape: {output.shape}")
        output_at_masked = output[0, i, :]
        #print(output_at_masked)
        input_at_masked = torch.squeeze(patched_input[0, i, :])
        #print(output_at_masked.shape)
        output_at_masked = torch.squeeze(output_at_masked)
        reconstructed_patches.append(output_at_masked)
        original_inputs.append(input_at_masked)
        if loss_func == 'l2':
          loss = calculate_loss_masked(patched_input, output, index, True) # last argument (sum) does not make a difference for batch size 1
        else:
          loss = calculate_l1_loss_masked(patched_input, output, index)
        loss_total_current_spec += loss.item()
      
      #print(reconstructed_patches[3])
      #print(reconstructed_patches[20])
      #assert torch.equal(reconstructed_patches[6], reconstructed_patches[20])
      #print(loss_total_current_spec)

      loss_total_current_spec /= patched_input.shape[1] #divide by number of patches
      #print(loss_total_current_spec)
      total_anom_scores.append(loss_total_current_spec) #coverting to numpy for processing with scikit
      total_targets.append(target)
      original_class_labels.append(class_label[0])
      #print(len(reconstructed_patches))
      #print(reconstructed_patches[0].shape)
      reconstruction = torch.stack(reconstructed_patches)
      #print(reconstruction.shape)
      reconstruction = torch.reshape(reconstruction, (1, config.N_MELS//2, -1))
      #print(reconstruction.shape)
      #print(patched_input.shape)
      #assert reconstruction.shape == patched_input.shape
      reconstruction = reconstruction.transpose(2,1)
      #print(reconstruction.shape)
      reconstruction = reconstruction.reshape(1, 1, config.N_MELS, -1)
      input, reconstruction = torch.squeeze(input), torch.squeeze(reconstruction)
      reconstruction = torch.reshape(reconstruction, (config.N_MELS, -1))
      reconstructions.append((input, reconstruction))
      #print(input.shape)
      #print(reconstruction.shape)
      #assert input.shape == reconstruction.shape
    assert (not torch.equal(reconstruction[0], reconstruction[2]))
    return total_anom_scores, total_targets, original_class_labels, reconstructions


def get_reconstruction(model, dataloader):
  reconstructed_input = torch.zeros(128, 88)
  return