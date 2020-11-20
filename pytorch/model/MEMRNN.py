import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, args, data):
    super(Model, self).__init__()
    self.use_cuda = args.cuda
    self.input_length = args.input_length
    self.m = data.m
    self.CNN_kernel = args.CNN_kernel
    self.CNN_unit = args.CNN_unit
    self.GRU_unit = args.GRU_unit
    self.AR = args.AR
    self.attn = args.attn

    self.input_encoded_m_conv = nn.Conv2d(1, self.CNN_unit, kernel_size = (self.CNN_kernel, self.m))
    self.input_encoded_m_attn = nn.Linear(self.input_length - self.CNN_kernel + 1, self.input_length - self.CNN_kernel + 1)
    self.input_encoded_m_gru = nn.GRU(self.CNN_unit, self.GRU_unit)

    self.question_encoded_conv = nn.Conv2d(1, self.CNN_unit, kernel_size = (self.CNN_kernel, self.m))
    self.question_encoded_attn = nn.Linear(self.input_length - self.CNN_kernel + 1, self.input_length - self.CNN_kernel + 1)
    self.question_encoded_gru = nn.GRU(self.CNN_unit, self.GRU_unit)

    self.input_encoded_c_conv = nn.Conv2d(1, self.CNN_unit, kernel_size = (self.CNN_kernel, self.m))
    self.input_encoded_c_attn = nn.Linear(self.input_length - self.CNN_kernel + 1, self.input_length - self.CNN_kernel + 1)
    self.input_encoded_c_gru = nn.GRU(self.CNN_unit, self.GRU_unit)
    self.dropout = nn.Dropout(p = args.dropout)

    self.linear1 = nn.Linear(self.GRU_unit * 8, self.m)

    self.output = None
    if args.output_fun == 'sigmoid':
      self.output = F.sigmoid
    if args.output_fun == 'tanh':
      self.output = F.tanh

    if args.AR > 0:
      self.linear_ar = nn.Linear(self.AR, 1)

  def forward(self, MEMORY, INPUT):
    batch_size = MEMORY.size(0)

    input_encoded_m = MEMORY.view(-1, 1, self.input_length, self.m)
    input_encoded_m = F.relu(self.input_encoded_m_conv(input_encoded_m))
    input_encoded_m = self.dropout(input_encoded_m)
    input_encoded_m = torch.squeeze(input_encoded_m, 3)
    if self.attn:
      input_encoded_m_w = F.softmax(self.input_encoded_m_attn(input_encoded_m), dim = 2)
      input_encoded_m = input_encoded_m * input_encoded_m_w
    input_encoded_m = input_encoded_m.permute(2, 0, 1)
    _, input_encoded_m = self.input_encoded_m_gru(input_encoded_m)
    input_encoded_m = torch.squeeze(input_encoded_m, 0)
    input_encoded_m = input_encoded_m.view(batch_size, -1, input_encoded_m.size(1))

    question_encoded = INPUT.view(-1, 1, self.input_length, self.m)
    question_encoded = F.relu(self.question_encoded_conv(question_encoded))
    question_encoded = self.dropout(question_encoded)
    question_encoded = torch.squeeze(question_encoded, 3)
    if self.attn:
      question_encoded_w = F.softmax(self.question_encoded_attn(question_encoded), dim = 2)
      question_encoded = question_encoded * question_encoded_w
    question_encoded = question_encoded.permute(2, 0, 1)
    _, question_encoded = self.question_encoded_gru(question_encoded)
    question_encoded = torch.squeeze(question_encoded, 0)
    question_encoded = question_encoded.view(batch_size, -1, question_encoded.size(1))

    match = torch.matmul(input_encoded_m, question_encoded.permute(0, 2, 1))
    match = torch.squeeze(match, 2)
    match = F.softmax(match, dim = 1)
    match = match.view(match.size(0), match.size(1), 1)

    input_encoded_c = MEMORY.view(-1 ,1, self.input_length, self.m)
    input_encoded_c = F.relu(self.input_encoded_c_conv(input_encoded_c))
    input_encoded_c = self.dropout(input_encoded_c)
    input_encoded_c = torch.squeeze(input_encoded_c, 3)
    if self.attn:
      input_encoded_c_w = F.softmax(self.input_encoded_c_attn(input_encoded_c), dim = 2)
      input_encoded_c = input_encoded_c * input_encoded_c_w
    input_encoded_c = input_encoded_c.permute(2, 0, 1)
    _, input_encoded_c = self.input_encoded_c_gru(input_encoded_c)
    input_encoded_c = torch.squeeze(input_encoded_c, 0)
    input_encoded_c = input_encoded_c.view(batch_size, -1, input_encoded_c.size(1))

    match = match.expand_as(input_encoded_c)
    o = torch.mul(match, input_encoded_c)
    o = o.view(batch_size, -1)

    question_encoded = torch.squeeze(question_encoded, 1)
    a_hat = torch.cat((question_encoded, o), 1)
    res = self.linear1(a_hat)

    if self.AR > 0:
      z = torch.squeeze(INPUT)
      z = z[:, -self.AR:, :]
      z = z.permute(0, 2, 1).contiguous()
      z = z.view(-1, self.AR)
      z = self.linear_ar(z)
      z = z.view(-1, self.m)
      res = res + z
    return res
