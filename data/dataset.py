import torchtext
from torchtext.data import Field, BucketIterator, Example
from torchtext import data

import spacy
import numpy as np

import random
import math
import time

from .lexical_analyzer import LexicalAnalyzer
lexer = LexicalAnalyzer()

def tokenize_english(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    text = text.lower()
    return text.split()

def tokenize_python(text, lexer):
    """
    Tokenizes Python code from a string into a list of strings
    """
    res = []
    token, lexeme, _, __ = lexer.tokenize(text)
    for i in range(len(token)):
        if token[i] not in ['S_MULTILINECOMMENT', 'D_MULTILINECOMMENT', 'COMMENT', 'SPACE']:
            res.append(lexeme[i])
    del res[500:]
    return res

def parse(filename):
    with open(filename, 'r') as datafile:
        data = datafile.read()
        data = data.replace(r'    ', '``')
        data = data.replace(r'   ', '``')
        data = data.replace(r'``', '\t')
        lines = data.split('\n')
        dataitems = []
        sent, code = '', ''
        for line in lines:
            if line[0] == '#':
                dataitems.append((sent, code))
                sent = line[1:]
                code = ''
            else:
                code = code + line
        dataitems.append((sent, code))        
        dataitems = dataitems[1:]
    
    return dataitems
    

def make_dataset_vocab(filename, src_params, trg_params, lexer, src_min_freq, trg_min_freq):
    dataitems = parse(filename)
    SRC = Field(tokenize = tokenize_english, 
        init_token = '<sos>', 
        eos_token = '<eos>', 
        lower = True, 
        batch_first = True)

    TRG = Field(tokenize = tokenize_python, 
        init_token = '<sos>', 
        eos_token = '<eos>', 
        lower = False,
        batch_first = True)
    
    fields = (('src', SRC), ('trg', TRG))
    example = [Example.fromlist([dataitems[i][0], dataitems[i][1]], fields) for i in range(len(dataitems))] 
    dataset = data.Dataset(example, fields)
    SRC.build_vocab(dataset, min_freq = src_min_freq)
    TRG.build_vocab(dataset, min_freq = trg_min_freq)
    return dataset, SRC, TRG

def get_train_test_iterator(dataset, train_split, SEED, sort = False, BATCH_SIZE = 32, device = 'cpu'):
    (train_data, test_data) = dataset.split(split_ratio=[train_split, 1 - train_split], random_state=random.seed(SEED))
    train_iterator, test_iterator = BucketIterator.splits(
            (train_data, test_data),
            sort=False,
            batch_size = BATCH_SIZE,
            device = device
        )
    return train_iterator, test_iterator

    



