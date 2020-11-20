import torch
import numpy as np
import math
import time

class Data_utility(object):

  def __init__(self, file_name, train, valid, cuda, horizon, input_length, normalize = 2):
    self.cuda = cuda
    self.h = horizon
    self.input_length = input_length
    fin = open(file_name)
    self.rawdat = np.loadtxt(fin, delimiter = ',')
    self.dat = np.zeros(self.rawdat.shape)
    self.n, self.m = self.dat.shape
    self.normalize = 2
    self.scale = np.ones(self.m)
    self._normalized(normalize)
    self._split(int(train * self.n), int((train + valid) * self.n), self.n)

    self.scale = torch.from_numpy(self.scale).float()
    tmp = self.test[2] * self.scale.expand(self.test[2].size(0), self.m)

    if self.cuda:
      self.scale = self.scale.cuda()

    self.rse = tmp.std()
    #self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

  def _normalized(self, normalize):

    if normalize == 0:
      self.dat = self.rawdat

    if normalize == 1:
      self.dat = self.rawdat / np.max(self.rawdat)

    if normalize == 2:
      for i in range(self.m):
        self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
        self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

  def _split(self, train, valid, test):

    train_set = range(24 * 8 + self.h - 1, train)
    valid_set = range(train, valid)
    test_set = range(valid, self.n)
    self.train = self._batchify(train_set, self.h)
    self.valid = self._batchify(valid_set, self.h)
    self.test = self._batchify(test_set, self.h)

  def _batchify(self, idx_set, horizon):

    n = len(idx_set)
    X = torch.zeros((n, 24 * 8, self.m))
    Y = torch.zeros((n, self.m))

    for i in range(n):
      end = idx_set[i] - self.h + 1
      start = end - 24 * 8
      X[i, :, :] = torch.from_numpy(self.dat[start: end, :])
      #X = X.view(-1, 8, 24, self.m)
      #MEMORY = X[:,: 7, :self.input_length, :]
      #INPUT = X[:, 7, :self.input_length, :]
      Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
    X = X.view(-1, 8, 24, self.m)
    MEMORY = X[:,: 7, :self.input_length, :]
    INPUT = X[:, -1, :self.input_length, :]

    return [MEMORY, INPUT, Y]

  def get_batches(self, memorys, inputs, targets, batch_size, shuffle = True):

    length = len(inputs)
    if shuffle:
      index = torch.randperm(length)
    else:
      index = torch.LongTensor(range(length))

    start_idx = 0
    while start_idx < length:
      end_idx = min(length, start_idx + batch_size)
      excerpt = index[start_idx:end_idx]
      MEMORY = memorys[excerpt]
      INPUT = inputs[excerpt]
      Y = targets[excerpt]
      if self.cuda:
        MEMORY = MEMORY.cuda()
        INPUT = INPUT.cuda()
        Y = Y.cuda()
      yield MEMORY, INPUT, Y
      del MEMORY, INPUT, Y
      torch.cuda.empty_cache()
      start_idx += batch_size

def evaluate(data, MEMORY, INPUT, Y, model, evaluateL2, evaluateL1, batch_size):
  model.eval()
  total_loss = 0
  total_loss_l1 = 0
  n_samples = 0
  predict = None
  test = None

  for MEMORY, INPUT, Y in data.get_batches(MEMORY, INPUT, Y, batch_size, False):
    output = model(MEMORY, INPUT)
    if predict is None:
      predict = output
      test = Y
    else:
      predict = torch.cat((predict, output))
      test = torch.cat((test, Y))

    scale = data.scale.expand(output.size(0), data.m)
    l2 = evaluateL2(output * scale, Y * scale)
    #l1 = evaluateL1(output * scale, Y * scale)
    total_loss += l2
    #total_loss_l1 += l1
    n_samples += (output.size(0) * data.m)

    del MEMORY, INPUT, Y, output, scale, l2
    torch.cuda.empty_cache()

  rse = math.sqrt(total_loss / n_samples) / data.rse
  #rae = (total_loss_l1 / n_samples) / data.rae.cuda()

  predict = predict.data.cpu().numpy()
  Ytest = test.data.cpu().numpy()
  sigma_p = predict.std(axis = 0)
  sigma_g = Ytest.std(axis = 0)
  mean_p = predict.mean(axis = 0)
  mean_g = Ytest.mean(axis = 0)
  index = (sigma_g != 0)
  correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
  correlation = (correlation[index]).mean()
  del predict, test, Ytest, sigma_p, sigma_g, mean_p, mean_g, index
  torch.cuda.empty_cache()
  return rse, correlation

def train(data, MEMORY, INPUT, Y, model, criterion, optim, batch_size):
  model.train()
  total_loss = 0
  n_samples = 0
  for MEMORY, INPUT, Y in data.get_batches(MEMORY, INPUT, Y, batch_size, True):
    model.zero_grad()
    output = model(MEMORY, INPUT)
    scale = data.scale.expand(output.size(0), data.m)
    loss = criterion(output * scale, Y * scale)
    loss.backward()
    grad_norm = optim.step()
    total_loss += loss
    n_samples += (output.size(0) * data.m)
    del MEMORY, INPUT, Y, output, scale, loss, grad_norm
    torch.cuda.empty_cache()
  return total_loss / n_samples
