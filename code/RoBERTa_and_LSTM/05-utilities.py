#!/usr/bin/python3
import torch

def to_gpu(x):
  if torch.cuda.is_available():
    return x.to('cuda')
  
  return x.to('cpu')

class Vocab: 

  """
  Class that handles mapping to and from 
  words to vocab indices
  """

  def __init__(self, stream = 'tokens', target = False):
    self._target = target
    if self._target:
      self._word2idx = {}
    else:
      self._word2idx = {'__UNK__': 0, '': 1}
    self._idx2word = {}
    self._stream = stream

  def train(self, raw_ds):
    # generate vocab list
    for sample in raw_ds:
      for token in sample[self._stream]:
        t = token.lower()
        if t not in self._word2idx:
          self._word2idx[t] = len(self._word2idx)

    # rebuild reverse lookup
    self._idx2word = {v: k for k, v in self._word2idx.items()}

  def encode(self, word):
    if self._target:
      return self._word2idx[word]
    else:
      return self._word2idx.get(word, self._word2idx['__UNK__'])

  def decode(self, idx):
    if self._target:
      return self._idx2word[idx]
    else:
      return self._idx2word.get(idx, '__UNK__')