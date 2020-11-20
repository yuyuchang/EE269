from utils import *
from argument import *
from model import MEMRNN
import time
import math
import torch
import torch.nn as nn
import numpy as np
import Optim

args.cuda = args.gpu is not None
if args.cuda:
  torch.cuda.set_device(args.gpu)


Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.input_length, args.normalize)
print(Data.rse)

model = eval(args.model).Model(args, Data)
print(model)

if args.cuda:
  model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
  criterion = nn.L1Loss(size_average = False)
else:
  criterion = nn.MSELoss(size_average = False)

evaluateL2 = nn.MSELoss(size_average = False)
evaluateL1 = nn.L1Loss(size_average = False)

if args.cuda:
  criterion = criterion.cuda()
  #evaluateL1 = evaluateL1.cuda()
  evaluateL2 = evaluateL2.cuda()

best_val_loss = 100000000
best_val_corr = -100000000
optim = Optim.Optim(model.parameters(), args.optim, args.lr, args.clip)

try:
  print('begin training')
  for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train_loss = train(Data, Data.train[0], Data.train[1], Data.train[2], model, criterion, optim, args.batch_size)
    torch.cuda.empty_cache()
    val_loss, val_corr = evaluate(Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1, args.batch_size)
    print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_corr))

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_val_corr = val_corr

    del train_loss, val_loss, val_corr
    torch.cuda.empty_cache()

except KeyboardInterrupt:
  print('-' * 89)

print('test rse {:5.4f} | test corr {:5.4f}'.format(best_val_loss, best_val_corr))

f = open(args.output, "a+")
f.write(str(args.horizon) + "," + str(args.input_length) + "," + str(args.CNN_kernel) + "," + str(args.CNN_unit) + "," + str(args.GRU_unit) + "," + str(best_val_loss.item()) + "," + str(best_val_corr) + "\n")
