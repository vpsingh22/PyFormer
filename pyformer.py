import torch
import torch.nn as nn
import torch.optim as optim

import random
import spacy
import numpy as np
import time

import pickle

from data.dataset import make_dataset_vocab
from data.lexical_analyzer import LexicalAnalyzer
from model.seq2seq import Encoder, Decoder, Seq2Seq



def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 500):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]   
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention



class PyFormer:
    def __init__(self, src_vocab_file, trg_vocab_file):
        self.SRC = self.load_vocab(src_vocab_file)
        self.TRG = self.load_vocab(trg_vocab_file)

    def load_model(self, weights, device):
        INPUT_DIM = len(self.SRC.vocab)
        OUTPUT_DIM = len(self.TRG.vocab)
        enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
        dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
        SRC_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        TRG_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]
        model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
        model.load_state_dict(torch.load(weights))
        return model

    def load_vocab(self, vocab_file):
        file = open(vocab_file, 'rb')      
        vocab = pickle.load(file) 
        file.close()
        return vocab
    

    def post_process(self, translation):
        output = ""
        for i in range(len(translation)):
            if translation[i] not in [' ', '\n', '\t']:
                if (i + 1 < len(translation) and translation[i + 1] in ['.', '(', ')', ',']) or translation[i] in ['.', '(', ')']:
                    output += translation[i]
                else: 
                    output += translation[i] + ' '
            else:
                output += translation[i]
        return output



if __name__ == '__main__':
    from colorama import Fore, Back, Style
    from config import config

    HID_DIM = config['HID_DIM']
    ENC_LAYERS = config['ENC_LAYERS']
    DEC_LAYERS = config['DEC_LAYERS']
    ENC_HEADS = config['ENC_HEADS']
    DEC_HEADS = config['DEC_HEADS']
    ENC_PF_DIM = config['ENC_PF_DIM']
    DEC_PF_DIM = config['DEC_PF_DIM']
    ENC_DROPOUT = config['ENC_DROPOUT']
    DEC_DROPOUT = config['DEC_DROPOUT']
    weights = 'rmodel.pt'
    src_vocab_file = 'src_vocab.pickle'
    trg_vocab_file = 'trg_vocab.pickle'

    src_vocab_file = config['src_vocab_file']
    trg_vocab_file = config['trg_vocab_file']
    weights = config['weights']
    device = config['device']

    pyformer = PyFormer(src_vocab_file, trg_vocab_file)
    model = pyformer.load_model(weights, device)
    ex_file = config['examples_file']
    with open(ex_file, 'r') as ex:
        input_sentences = ex.readlines()
        for input_sentence in input_sentences:
            print(Back.LIGHTYELLOW_EX + input_sentence + '\n')

            translation, attention = translate_sentence(input_sentence, pyformer.SRC, pyformer.TRG, model, device)
            translation = translation[:-1]
            output = pyformer.post_process(translation)
            print(Back.LIGHTCYAN_EX + output + '\n')
            print('\n\n')

