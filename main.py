import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import time

import pickle

from data.dataset import make_dataset_vocab, get_train_test_data, get_train_test_iterator
from data.lexical_analyzer import LexicalAnalyzer
from model.seq2seq import Encoder, Decoder, Seq2Seq
from model.utils import count_parameters, initialize_weights, epoch_time
from model.train import train
from model.test import evaluate

from config import config

SEED = config['SEED']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

lexer = LexicalAnalyzer()

src_params = config['src_params']
trg_params = config['trg_params']
filename = config['datafile']
src_min_freq = config['src_min_freq']
trg_min_freq = config['trg_min_freq']

dataset, SRC, TRG = make_dataset_vocab(filename, src_params, trg_params, lexer, src_min_freq = src_min_freq, trg_min_freq = trg_min_freq)

src_vocab_file = config['src_vocab_file']
src_vocab = open('src_vocab.pickle', 'ab') 
pickle.dump(SRC, src_vocab)                      
src_vocab.close() 

trg_vocab_file = config['trg_vocab_file']
trg_vocab = open('trg_vocab.pickle', 'ab') 
pickle.dump(SRC, trg_vocab)                      
trg_vocab.close() 

train_split = config['train_split']
sort = config['sort']
batch_size = config['batch_size']
device = config['device']
if device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print('cuda not available. device switched to cpu ... ')

train_data, test_data = get_train_test_data(dataset, train_split, SEED)
train_iterator, test_iterator = get_train_test_iterator(train_data, test_data, sort = sort, batch_size = batch_size, device = device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = config['HID_DIM']
ENC_LAYERS = config['ENC_LAYERS']
DEC_LAYERS = config['DEC_LAYERS']
ENC_HEADS = config['ENC_HEADS']
DEC_HEADS = config['DEC_HEADS']
ENC_PF_DIM = config['ENC_PF_DIM']
DEC_PF_DIM = config['DEC_PF_DIM']
ENC_DROPOUT = config['ENC_DROPOUT']
DEC_DROPOUT = config['DEC_DROPOUT']

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

LEARNING_RATE = config['LEARNING_RATE']
N_EPOCHS = config['N_EPOCHS']
CLIP = config['CLIP']

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


# TRAINING 

best_valid_loss = float('inf')
best_train_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'min_val_loss_model.pt')
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), 'min_train_loss_model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


print()
from metrics import calculate_bleu

score = calculate_bleu(test_data, SRC, TRG, model, device)
print(f'BLEU score = {score*100:.2f}')

