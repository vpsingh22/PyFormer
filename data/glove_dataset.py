import torch
import torch.nn as nn

import numpy as np

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class GloveDataset:

    def __init__(self, corpus, lexer, TRG, n_words=200000, window_size=5):
        self._window_size = window_size
        text = ""


        _, lexe, __, ___ = lexer.tokenize(text)
        tokens = []
        for i in range(len(_)):
            if _[i] not in ['S_MULTILINECOMMENT', 'D_MULTILINECOMMENT', 'COMMENT', 'SPACE']:
                tokens.append(lexe[i])

        self._tokens = tokens[:n_words]
        word_counter = Counter()
        word_counter.update(self._tokens)
        #   self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._word2id = vars(TRG.vocab)['stoi']
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        self._id_tokens = [self._word2id[w] for w in self._tokens]

        self._create_coocurrence_matrix()

        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()


    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]

